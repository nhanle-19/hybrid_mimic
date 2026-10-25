"""Solve the eight-contact FORCE QP along the crawl reference (offline snapshots).

This audits constraints and manual labels; it does not integrate a simulator or
claim that the resulting torques track the crawl in closed loop.
"""
import argparse
import ast
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from crawl_force_schedule import ROOT, digest, load_manual_schedule, save_schedule
from crawl_manual_nn_outputs import DEFAULT, load_manual_nn_outputs, save_manual_nn_outputs
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_gpu_control import select_state
from wbc_crawl_contact import CRAWL_REGIONS, CrawlContactGeometry, WBCCrawlForceQP


def controller_config(specification):
    path = ROOT/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/controller_cfg.py'
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    values = {node.target.id: ast.literal_eval(node.value) for node in cls.body if isinstance(node, ast.AnnAssign)}
    values.update(contact_force_max=tuple(specification['force_max_n']), swing_weight=0.,
                  contact_gap_tolerance=specification['contact_gap_tolerance_m'],
                  contact_patch_height_tolerance=specification['contact_patch_height_tolerance_m'],
                  max_contact_penetration=specification['max_contact_penetration_m'])
    return SimpleNamespace(**values)


def torque_limits(joint_names):
    """Read the actual robot actuator limits without importing Isaac Sim."""
    path = ROOT/'source/whole_body_tracking/whole_body_tracking/robots/t1.py'
    matches = {name: [] for name in joint_names}
    tree = ast.parse(path.read_text())
    robot = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'BOOSTER_T1_CFG' for target in node.targets))
    for node in ast.walk(robot):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'ImplicitActuatorCfg':
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        patterns = ast.literal_eval(keywords['joint_names_expr'])
        effort = ast.literal_eval(keywords['effort_limit_sim'])
        for name in joint_names:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                values = [value for pattern, value in effort.items() if re.fullmatch(pattern, name)] if isinstance(effort, dict) else [effort]
                matches[name].extend(values)
    if any(not values or len(set(values)) != 1 for values in matches.values()):
        raise ValueError('Missing or inconsistent robot actuator limits')
    return np.asarray([matches[name][0] for name in joint_names], dtype=float)


def motion_state(model, motion):
    """Convert portable XYZW poses and finite-difference velocities to USD DOFs."""
    order = [motion['joint_names'].tolist().index(name) for name in model.joint_names]
    joint = motion['dof_pos'][:, order]
    rotation = Rotation.from_quat(motion['root_rot'])
    fps = float(motion['fps'])
    linear = rotation.inv().apply(np.gradient(motion['root_pos'], 1/fps, axis=0))
    # Spatial angular velocity from neighboring relative rotations; map back
    # into the LOCAL root twist convention of the analytical dynamics model.
    increment = (rotation[1:]*rotation[:-1].inv()).as_rotvec()*fps
    angular_world = np.r_[increment[:1], .5*(increment[:-1]+increment[1:]), increment[-1:]]
    angular = rotation.inv().apply(angular_world)
    q = np.c_[motion['root_pos'], motion['root_rot'], joint]
    v = np.c_[linear, angular, np.gradient(joint, 1/fps, axis=0)]
    return model.state(model.tensor(q), model.tensor(v))


def run(schedule_path, output, device='cpu', batch_size=24, manual_outputs_path=DEFAULT, friction=.2,
        backend='batched', solver_tolerance=1e-12):
    torch.set_num_threads(4)
    specification, motion, annotations, schedule = load_manual_schedule(schedule_path)
    output.mkdir(parents=True, exist_ok=True)
    save_schedule(output, schedule)
    manual_outputs = load_manual_nn_outputs(manual_outputs_path, ROOT/specification['motion'])
    save_manual_nn_outputs(output/'manual_nn_outputs.npz', manual_outputs)
    model = WBCACCTorchModel(device)
    cfg = controller_config(specification)
    geometry = CrawlContactGeometry(ROOT/specification['geometry'], device)
    solver = WBCCrawlForceQP(cfg, geometry, friction=friction, backend=backend,
                            solver_tolerance=solver_tolerance)
    states = motion_state(model, motion)
    count = len(schedule['time_s'])
    labels = torch.as_tensor(schedule['contact_labels'], device=device)
    # This is a reference-frame snapshot audit, not the live control-rate replay.
    # Align each snapshot to its first policy sample when the table runs faster.
    multiple = json.loads(str(manual_outputs['metadata_json'])).get('sample_rate_multiple', 1)
    actions = model.tensor(manual_outputs['actions'][::multiple])
    limit = model.tensor(torque_limits(model.joint_names))
    columns = [annotations['surface_names'].tolist().index(name) for name in CRAWL_REGIONS]
    # Independent MuJoCo FK was saved with the labels, whereas the solver uses
    # the exported USD tree. Check every body before using its mesh vertices.
    fk_error = 0.
    for body, frame in states['frames'].items():
        i = annotations['body_names'].tolist().index(body)
        fk_error = max(fk_error, float((frame['position']-model.tensor(annotations['body_position_w'][:, i])).abs().max()),
                       float((frame['rotation']-model.tensor(annotations['body_rotation_w'][:, i])).abs().max()))
    if fk_error > 1e-6:
        raise ValueError(f'Crawl dynamics/annotation body frames disagree: {fk_error}')
    keys = ('torque', 'acceleration', 'rho', 'point_forces', 'wrenches', 'normal_force_limits', 'available',
            'actual_available', 'clearance', 'point_position_w', 'point_ids', 'residual', 'failed', 'wrench_origin_w')
    chunks = {key: [] for key in keys}
    gap_error = inverse_error = friction_violation = 0.
    for start in range(0, count, batch_size):
        end = min(start+batch_size, count)
        state = select_state(states, slice(start, end))
        result = solver.solve(state, state, actions[start:end], limit.expand(end-start, -1))
        gap_error = max(gap_error, float((result['clearance']-model.tensor(annotations['surface_clearance_m'][start:end, columns])).abs().max()))
        if not torch.equal(result['actual_available'].any(-1), labels[start:end]):
            raise ValueError(f'Actual-pose contact geometry differs from manual labels in frames {start}:{end}')
        inactive = result["activation"] <= 0
        if torch.count_nonzero(result['point_forces'][inactive]) or torch.count_nonzero(result['normal_force_limits'][inactive]):
            raise ValueError('Non-contact region produced nonzero force/capacity')
        # Reconstruct generalized forces independently from saved point IDs.
        generalized = torch.zeros_like(state['bias'])
        for region in range(8):
            for slot in range(4):
                ids = result['point_ids'][:, region, slot]
                valid = ids >= 0
                safe_ids = ids.clamp_min(0).cpu().numpy()
                bodies = annotations['point_body_index'][safe_ids]
                for body_id in np.unique(bodies):
                    frame = state['frames'][str(annotations['body_names'][body_id])]
                    force = torch.where((valid & model.tensor(bodies == body_id).bool())[:, None],
                                        result['point_forces'][:, region, slot], 0.)
                    offset = result['point_position_w'][:, region, slot]-frame['position']
                    wrench = torch.cat((force, torch.cross(offset, force, dim=-1)), -1)
                    generalized += mv(frame['J'].transpose(-1, -2), wrench)
        inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
        inverse[:, 6:] -= result['torque']
        inverse_error = max(inverse_error, float(inverse.abs().max()))
        forces = result['point_forces']
        friction_violation = max(friction_violation, float((forces[..., :2].norm(dim=-1)-friction*forces[..., 2]).clamp_min(0).max()))
        for key in keys:
            chunks[key].append(result[key].detach().cpu().numpy())
        print(f'Solved {end}/{count}; failures={int(result["failed"].sum())}; max residual={float(result["residual"].max()):.3g}', flush=True)
    results = {key: np.concatenate(values) for key, values in chunks.items()}
    # The batched audit compares against OSQP on all contact patterns. The
    # optional OSQP-only audit checks every snapshot independently, without
    # pretending that solving OSQP twice establishes cross-solver parity.
    sample = torch.tensor([0, 70, 95, 104, 106, 115, 329, 453, 497, 513, 540, 598], device=device)
    state = select_state(states, sample)
    if backend == 'batched':
        osqp_result = WBCCrawlForceQP(cfg, geometry, backend='osqp', friction=friction).solve(
            state, state, actions[sample], limit.expand(len(sample), -1))
        parity = float(np.max(np.abs(osqp_result['torque'].cpu().numpy()-results['torque'][sample.cpu().numpy()])))
        oracle_failed = sample[osqp_result['failed']].cpu().tolist()
    else:
        parity, oracle_failed = None, []
    # Positive raw outputs cannot create contact when the robot lifts 20 mm.
    q = states['q'][:1].clone()
    q[:, 2] += .02
    raised = model.state(q, states['v'][:1])
    lifted = solver.solve(raised, raised, torch.full((1, 8), 3., device=device, dtype=model.dtype),
                          limit[None])
    # One disabled region and a fully disabled schedule both remove forces exactly.
    zero_state = select_state(states, slice(0, 1))
    negative_checks = {}
    for name, disabled in [('left_hand_output_off', [0]), ('all_outputs_off', list(range(8)))]:
        manual = actions[:1].clone()
        manual[:, disabled] = -3.
        result = solver.solve(zero_state, zero_state, manual, limit[None])
        negative_checks[name] = bool(not result['failed'].any() and result['point_forces'][:, disabled].count_nonzero() == 0
                                     and result['normal_force_limits'][:, disabled].count_nonzero() == 0)
    negative_checks['lifted_20mm_positive_outputs_zero_forces'] = bool(
        not lifted['failed'].any() and lifted['point_forces'].count_nonzero() == 0 and lifted['normal_force_limits'].count_nonzero() == 0)
    normal = results['point_forces'][..., 2].sum(-1)
    cap_violation = float(np.maximum(normal-results['normal_force_limits'], 0).max())
    torque_violation = float(np.maximum(np.abs(results['torque'])-limit.cpu().numpy(), 0).max())
    unilateral_violation = float(np.maximum(-results['point_forces'][..., 2], 0).max())
    failed_frames = np.flatnonzero(results['failed']).tolist()
    passed = (not failed_frames and not oracle_failed and all(negative_checks.values())
              and max(inverse_error, cap_violation, torque_violation, friction_violation, unilateral_violation) < 2e-5
              and gap_error < 1e-6 and (parity is None or parity < 2e-3))
    report = dict(status='passed' if passed else 'failed', validation_type='offline_reference_snapshots',
                  physical_rollout_verified=False, solver_backend=backend, solver_tolerance=solver_tolerance,
                  cross_solver_parity_checked=backend == 'batched',
                  frames=count, fps=float(motion['fps']), region_order=list(CRAWL_REGIONS),
                  policy_sampling='first policy sample of each reference frame',
                  manual_sample_rate_multiple=multiple,
                  normal_world=[0., 0., 1.], inequality='0 <= sum_p f[region,p,z] <= decoded_action * force_max (live geometry available)',
                  force_max_n=specification['force_max_n'], friction_coefficient=friction,
                  contact_frames={name: int(labels[:, i].sum()) for i, name in enumerate(CRAWL_REGIONS)},
                  normal_bound_count=8, point_slots_per_region=4, qp_variable_count=model.nv+8*4*4,
                  failed_frames=failed_frames, max_qp_residual=float(results['residual'].max()),
                  independent_inverse_dynamics_error=inverse_error, max_torque_bound_violation_nm=torque_violation,
                  max_normal_bound_violation_n=cap_violation, max_unilateral_violation_n=unilateral_violation,
                  max_friction_violation_n=friction_violation, dynamics_vs_label_fk_error=fk_error,
                  max_clearance_disagreement_m=gap_error,
                  osqp_frames=sample.cpu().tolist() if backend == 'batched' else list(range(count)),
                  osqp_failed_frames=oracle_failed if backend == 'batched' else failed_frames,
                  max_osqp_torque_difference_nm=parity,
                  negative_checks=negative_checks, controller_config=vars(cfg), joint_names=model.joint_names,
                  joint_torque_limits_nm=limit.cpu().tolist(), review=specification['review'],
                  limitations=['Manual intervals describe the repaired robot reference, not measured human support forces.',
                               'Four actual mesh vertices per region approximate its support patch conservatively.',
                               'Friction is bounded, but the QP does not enforce a kinetic-friction/slip law.',
                               'Existing Isaac FORCE tasks/checkpoints remain two-foot tasks; this eight-region solver is exercised by this offline runner.',
                               'PhysX collision shapes/sensors and closed-loop crawl tracking are not validated by these snapshot solves.'])
    np.savez_compressed(output/'qp_diagnostics.npz', **results, contact_labels=schedule['contact_labels'],
                        actions=actions.cpu().numpy(), time_s=schedule['time_s'], region_names=schedule['region_names'])
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in
                [Path(schedule_path), Path(manual_outputs_path), ROOT/specification['motion'], ROOT/specification['geometry'],
                 Path(__file__), ROOT/'scripts/crawl_force_schedule.py',
                 ROOT/'source/whole_body_tracking/whole_body_tracking/utils/wbc_crawl_contact.py',
                 ROOT/'source/whole_body_tracking/whole_body_tracking/utils/wbc_acc_gpu_control.py',
                 ROOT/'source/whole_body_tracking/whole_body_tracking/utils/wbc_force_gpu_control.py',
                 ROOT/'source/whole_body_tracking/whole_body_tracking/assets/booster/t1/wbc_acc_dynamics.json',
                 ROOT/'source/whole_body_tracking/whole_body_tracking/robots/t1.py',
                 *sorted(output.glob('*.npz')), output/'manual_schedule.csv', output/'validation.json']}
    (output/'artifact_sha256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError('Crawl FORCE validation failed; inspect validation.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, default=ROOT/'configs/crawl_force_schedule.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/crawl_force_checks')
    parser.add_argument('--manual-outputs', type=Path, default=DEFAULT)
    parser.add_argument('--friction', type=float, default=.2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=['batched', 'osqp'], default='batched')
    parser.add_argument('--solver-tolerance', type=float, default=1e-12)
    parser.add_argument('--batch-size', type=int, default=24)
    args = parser.parse_args()
    if not np.isfinite(args.solver_tolerance) or args.solver_tolerance <= 0:
        parser.error('Solver tolerance must be finite and positive')
    run(args.schedule.resolve(), args.output.resolve(), args.device, args.batch_size,
        args.manual_outputs.resolve(), args.friction, args.backend, args.solver_tolerance)
