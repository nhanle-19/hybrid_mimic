"""Controller-only numerical audit. Separate from a closed-loop PhysX rollout."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/whole_body_tracking/whole_body_tracking'
sys.path.insert(0, str(SOURCE/'utils'))
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_contact_policy import available_vertices, sole_vertices, FEET
from wbc_authority_gpu_control import WBCAuthorityGPUQP, encode_authorities


def controller_defaults():
    # Read literal config defaults without importing Isaac Sim. Refuse nonliteral
    # values rather than silently substituting a separate test configuration.
    values = {}
    for directory, name in [('t1_wbc_acc', 'T1WBCACCControllerCfg'),
                            ('t1_wbc_authority', 'T1WBCAuthorityControllerCfg')]:
        tree = ast.parse((SOURCE/f'tasks/tracking/config/{directory}/controller_cfg.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        values.update({node.target.id: ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.AnnAssign)})
    return SimpleNamespace(**values)


def torque_limits(model):
    tree = ast.parse((SOURCE/'robots/t1.py').read_text())
    cfg = next(node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'BOOSTER_T1_CFG' for target in node.targets))
    actuators = next(keyword.value for keyword in cfg.keywords if keyword.arg == 'actuators')
    limits = {}
    for actuator in actuators.values:
        args = {keyword.arg: keyword.value for keyword in actuator.keywords}
        patterns = ast.literal_eval(args['joint_names_expr'])
        bound = ast.literal_eval(args['effort_limit_sim'])
        for name in model.joint_names:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                limits[name] = (next(value for pattern, value in bound.items() if re.fullmatch(pattern, name))
                                if isinstance(bound, dict) else bound)
    return model.tensor([limits[name] for name in model.joint_names])


def audit(device='cpu'):
    cfg = controller_defaults()
    model = WBCACCTorchModel(device)
    cases, fractions, masks = [], [], []
    def add(name, c, mask=None):
        cases.append(name)
        fractions.append(torch.as_tensor(c, dtype=torch.float64, device=device).expand(2, 3).clone())
        masks.append(torch.ones((2, 4), dtype=torch.bool, device=device) if mask is None else mask)
    for value in (0., 1e-12, 1e-8, 1e-4, .01, .1, .5, 1.):
        add(f'uniform_{value}', value)
    for axis in range(3):
        for value in (0., 1e-10, .01):
            c = torch.ones((2, 3), dtype=torch.float64, device=device)
            c[0, axis] = value
            add(f'left_axis_{axis}_{value}', c)
    # Mixed geometry in one batch, including no support, corners, edges, feet.
    for bits in range(256):
        mask = ((bits >> torch.arange(8, device=device)) & 1).bool().reshape(2, 4)
        add(f'geometry_{bits}', .5, mask)
    n = len(cases)
    q = model.tensor([[0.]*30]).expand(n, -1).clone()
    q[:, 6] = 1.
    v = model.tensor([[0.]*29]).expand(n, -1).clone()
    state = model.state(q, v)
    q[:, 2] -= state['frames'][FEET[0]]['position'][:, 2]+sole_vertices(device)[0, :, 2].min()
    state = model.state(q, v)
    for frame in state['frames'].values():
        frame['acceleration'] = torch.zeros_like(frame['velocity'])
    support = available_vertices(state, cfg) & torch.stack(masks)
    actions = encode_authorities(torch.stack(fractions))
    limits = torque_limits(model)[None].expand(n, -1)
    report = {'description': 'Static analytical QPs; not simulator stability certification',
              'device': device, 'cases': n, 'backends': {}}
    solutions = {}
    for backend in ('batched', 'osqp'):
        result = WBCAuthorityGPUQP(cfg, backend=backend).solve(state, state, actions, support, limits)
        generalized = sum(mv(state['frames'][foot]['J'].transpose(-1, -2), result['wrenches'][:, i])
                          for i, foot in enumerate(FEET))
        inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
        inverse[:, 6:] -= result['torque']
        wrench = result['wrenches']
        errors = torch.stack((inverse.abs().amax(-1),
                              (wrench[..., 2]-result['normal_force_limits']).clamp_min(0).amax(-1),
                              (-wrench[..., 2]).clamp_min(0).amax(-1),
                              (wrench[..., :2].norm(dim=-1)-.6*wrench[..., 2]).clamp_min(0).amax(-1),
                              (result['torque'].abs()-limits).clamp_min(0).amax(-1)), -1)
        failed = result['failed'] | ~torch.isfinite(errors).all(-1) | (errors.amax(-1) > 2e-5)
        report['backends'][backend] = {'failures': [cases[i] for i in torch.where(failed)[0].tolist()],
                                      'maximum_physical_error': float(errors.max()),
                                      'maximum_qp_residual': float(result['residual'].max())}
        solutions[backend] = result
    both_valid = ~(solutions['batched']['failed'] | solutions['osqp']['failed'])
    difference = (solutions['batched']['acceleration']-solutions['osqp']['acceleration']).abs().amax(-1)
    report['maximum_acceleration_difference_on_valid_cases'] = float(difference[both_valid].max())
    report['passed'] = all(not value['failures'] for value in report['backends'].values()) and bool((difference[both_valid] < .01).all())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('eval_data/wbc_authority/pretraining/numerical.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = audit(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
