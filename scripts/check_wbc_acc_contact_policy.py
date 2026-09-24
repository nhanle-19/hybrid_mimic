"""Offline standing/slow-lift QP checks using manual actions; not a simulator rollout."""
import ast
import json
import sys
import re
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_gpu_control import WBCACCGPUQP
from wbc_acc_contact_policy import FEET, sole_vertices, available_vertices, finite_difference


def main():
    torch.set_num_threads(4)
    path = ROOT/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/controller_cfg.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    cfg = SimpleNamespace(**{n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)})
    model = WBCACCTorchModel()
    robot_cfg = ast.parse((ROOT/'source/whole_body_tracking/whole_body_tracking/robots/t1.py').read_text())
    robot_call = next(n.value for n in robot_cfg.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BOOSTER_T1_CFG' for t in n.targets))
    actuators = next(k.value for k in robot_call.keywords if k.arg == 'actuators')
    joint_limits = {}
    for actuator in actuators.values:
        params = {k.arg: k.value for k in actuator.keywords}
        patterns = ast.literal_eval(params['joint_names_expr'])
        effort = ast.literal_eval(params['effort_limit_sim'])
        for name in model.joint_names:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                joint_limits[name] = next(value for pattern, value in effort.items() if re.fullmatch(pattern, name)) if isinstance(effort, dict) else effort
    configured_limits = torch.tensor([joint_limits[name] for name in model.joint_names], dtype=torch.float64)
    count, dt = 201, .02
    q = torch.zeros((count, 30), dtype=torch.float64); q[:, 6] = 1.; q[:, 2] = .72
    v = torch.zeros((count, 29), dtype=torch.float64)
    initial = model.state(q[:1], v[:1])
    q[:, 2] -= initial['frames'][FEET[0]]['position'][0, 2]+sole_vertices()[0, :, 2].min()
    report = {'validation_kind': 'prescribed kinematic snapshots, not closed-loop simulation', 'torque_limit_source': 'BOOSTER_T1_CFG effort_limit_sim', 'scenarios': {}}
    for scenario in ('standing', 'slow_right_foot_lift'):
        trajectory = q.clone()
        if scenario != 'standing':
            phase = torch.linspace(0, 1, count, dtype=torch.float64)
            amount = .8*(3*phase.square()-2*phase.pow(3))
            for name, scale in [('Right_Hip_Pitch', -.5), ('Right_Knee_Pitch', 1.), ('Right_Ankle_Pitch', -.5)]:
                trajectory[:, 7+model.joint_names.index(name)] += scale*amount
        velocities = v.clone(); velocities[:, 6:] = finite_difference(trajectory[:, 7:], dt)
        state = model.state(trajectory, velocities)
        for frame in state['frames'].values(): frame['acceleration'] = finite_difference(frame['velocity'], dt)
        actions = torch.zeros((count, 14), dtype=torch.float64); actions[:, 0] = 1.; actions[:, 7] = 1.
        if scenario != 'standing': actions[:, 7] = (1.-torch.linspace(0, 4, count)).clamp_min(0)
        geometry = available_vertices(state, cfg)
        limits = configured_limits.expand(count, -1)
        result = WBCACCGPUQP(cfg).solve(state, state, actions, geometry, limits)
        unavailable = ~geometry.any(-1) | (actions[:, [0, 7]] == 0)
        assert result['wrenches'][unavailable].count_nonzero() == 0
        assert not result['failed'].any()
        report['scenarios'][scenario] = {
            'samples': count, 'solver_failures': int(result['failed'].sum()),
            'max_physical_residual': float(result['residual'].max()),
            'max_torque_utilization': float((result['torque'].abs()/limits).max()),
            'final_right_foot_lift_m': float(state['frames'][FEET[1]]['position'][-1, 2]-state['frames'][FEET[1]]['position'][0, 2]),
            'unavailable_or_disabled_wrench_max': float(result['wrenches'][unavailable].abs().max()) if unavailable.any() else 0.,
        }
    output = ROOT/'eval_data/wbc_acc/reference_contact_validation/manual_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
