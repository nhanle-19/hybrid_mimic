"""Run eight-region FORCE torques in Isaac/PhysX using manually specified NN outputs.

The robot is initialized once. Subsequent states come only from physics; the
reference supplies feedback targets and manual outputs, never root/joint teleports.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schedule', type=Path, default=ROOT/'configs/crawl_force_schedule.json')
parser.add_argument('--manual-outputs', type=Path, default=ROOT/'eval_data/reference_crawl_forward/force/planned_labels.npz',
                    help='Fixed raw NN output table supplied manually in place of a learned policy; supports reference or control rate.')
parser.add_argument('--asset-dir', type=Path, default=ROOT/'eval_data/reference_crawl_forward/physics_asset')
parser.add_argument('--output', type=Path, default=ROOT/'eval_data/reference_crawl_forward/force')
parser.add_argument('--root-error-limit', type=float, default=.5,
                    help='Preliminary rollout stop threshold in metres; tracking errors remain recorded.')
parser.add_argument('--joint-error-limit', type=float, default=.6,
                    help='Joint tracking RMSE stop threshold in radians.')
parser.add_argument('--frames', type=int, help='Reference frames to execute; default is the full 599-frame motion.')
parser.add_argument('--substeps', type=int, default=16)
parser.add_argument('--qp-every', type=int, default=4, help='Physics steps between actual-state QP updates.')
parser.add_argument('--qp-backend', choices=['batched', 'osqp'], default='batched')
parser.add_argument('--record-contact-substeps', action='store_true',
                    help='Save measured contacts after every physics step for impact audits; controller rate is unchanged.')
parser.add_argument('--posture-kp', type=float, default=1600.)
parser.add_argument('--posture-kd', type=float, default=80.)
parser.add_argument('--posture-weight', type=float, default=3.)
parser.add_argument('--pelvis-weight', type=float, help='Diagnostic override of the body-orientation tracking weight.')
parser.add_argument('--max-penetration', type=float, default=.03,
                    help='Actual contact penetration allowance, matching the existing WBC physics gate.')
parser.add_argument('--com-kp', type=float, default=160.)
parser.add_argument('--momentum-kv', type=float, default=25.)
parser.add_argument('--vertical-momentum-weight', type=float, default=10.,
                    help='Weight of vertical centroidal momentum tracking; leaves horizontal tracking weights unchanged.')
parser.add_argument('--friction', type=float, default=.2, help='Same coefficient in the QP and both collision materials.')
parser.add_argument('--force-max', type=float, help='Diagnostic override of every ON-region force ceiling in newtons.')
parser.add_argument('--hand-force-max', type=float, help='Diagnostic override of the two hand force ceilings only.')
parser.add_argument('--reference-speed', type=float, default=1., help='Time scaling; changes reference velocities and label timing together.')
parser.add_argument('--hold-frame', type=int, help='Diagnostic static pose: hold this reference frame and its contact labels.')
parser.add_argument('--contact-separation-velocity', type=float,
                    help='Diagnostic gate: disable above-floor points moving upward faster than this m/s.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.substeps < 1 or args.qp_every < 1 or args.substeps % args.qp_every or (args.frames is not None and args.frames < 1):
    parser.error('Positive substeps/qp-every are required; qp-every must divide substeps')
if any(not math.isfinite(value) or value <= 0 for value in (args.posture_kp, args.posture_kd,
        args.posture_weight, args.max_penetration, args.com_kp, args.momentum_kv, args.friction, args.reference_speed,
        args.root_error_limit, args.joint_error_limit)):
    parser.error('Controller gains, weight and penetration allowance must be finite and positive')
if any(value is not None and (not math.isfinite(value) or value <= 0)
       for value in (args.force_max, args.hand_force_max, args.pelvis_weight,
                     args.contact_separation_velocity, args.vertical_momentum_weight)):
    parser.error('Optional controller overrides must be finite and positive')
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
(args.output/'replay_command.json').write_text(json.dumps(sys.argv, indent=2)+'\n')
# Application flags are already in args; do not pass experiment flags to Kit.
sys.argv = [sys.argv[0]]
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from whole_body_tracking.robots.t1 import BOOSTER_T1_CFG
from check_crawl_force_qp import controller_config, motion_state, torque_limits
from crawl_force_schedule import digest, load_manual_schedule
from crawl_manual_nn_outputs import load_manual_nn_outputs, save_manual_nn_outputs, save_replay_labels
from build_crawl_physics_asset import build
from wbc_acc_torch_model import WBCACCTorchModel, rotation_xyzw, mv
from wbc_crawl_contact import CrawlContactGeometry, CRAWL_REGIONS
from wbc_crawl_control import CrawlForceController, classify_contact_points


def main():
    torch.set_num_threads(4)
    specification, motion, annotations, schedule = load_manual_schedule(args.schedule)
    manual_outputs = load_manual_nn_outputs(args.manual_outputs, ROOT/specification['motion'])
    save_manual_nn_outputs(args.output/'planned_labels.npz', manual_outputs)
    asset_dir = args.asset_dir.resolve()
    if not (asset_dir/'collision_manifest.json').is_file():
        build(asset_dir, ROOT/specification['geometry'])
    collision = json.loads((asset_dir/'collision_manifest.json').read_text())
    if collision['source_geometry_sha256'] != specification['geometry_sha256']:
        raise ValueError('Physics asset belongs to different contact annotations; rebuild it')
    for key in ('asset', 'geometry', 'source_asset'):
        if digest(collision[key]) != collision[key+'_sha256']:
            raise ValueError(f'Stale crawl physics asset: {key}')
    frames = len(motion['root_pos']) if args.frames is None else args.frames
    if frames > len(motion['root_pos']):
        raise ValueError('Cannot execute beyond the supplied reference')
    if args.hold_frame is not None and not 0 <= args.hold_frame < len(motion['root_pos']):
        raise ValueError('Held frame must index the supplied reference')
    fps = float(motion['fps'])
    dt = 1/(fps*args.substeps)
    cfg = controller_config(specification)
    cfg.posture_position_gain, cfg.posture_velocity_gain = args.posture_kp, args.posture_kd
    cfg.posture_weight = args.posture_weight
    if args.pelvis_weight is not None:
        cfg.pelvis_weight = args.pelvis_weight
    cfg.max_contact_penetration = args.max_penetration
    if args.contact_separation_velocity is not None:
        cfg.contact_separation_velocity = args.contact_separation_velocity
    cfg.com_position_gain, cfg.linear_momentum_gain = args.com_kp, args.momentum_kv
    if args.vertical_momentum_weight is not None:
        cfg.momentum_weights = (*cfg.momentum_weights[:2], args.vertical_momentum_weight,
                                *cfg.momentum_weights[3:])
    if args.force_max is not None:
        cfg.contact_force_max = (args.force_max,)*8
    if args.hand_force_max is not None:
        cfg.contact_force_max = (args.hand_force_max,)*2+tuple(cfg.contact_force_max[2:])
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    # SimulationContext disables reports by default. Direct tensor views need
    # the same opt-in that Isaac Lab's ContactSensor constructor performs.
    sim.carb_settings.set_bool('/physics/disableContactProcessing', False)
    ground = sim_utils.CuboidCfg(size=(200., 200., .1), collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=.001, rest_offset=0.), physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=args.friction, dynamic_friction=args.friction, restitution=0.))
    ground.func('/World/Ground', ground, translation=(0., 0., -.05))
    ground_shapes = [str(prim.GetPath()) for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/Ground'))
                     if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(ground_shapes) != 1:
        raise ValueError(f'Expected one physical ground shape, got {ground_shapes}')
    robot_cfg = BOOSTER_T1_CFG.copy()
    robot_cfg.prim_path = '/World/Robot'
    robot_cfg.spawn.usd_path = collision['asset']
    robot_cfg.soft_joint_pos_limit_factor = 1.
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 32
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 8
    for actuator in robot_cfg.actuators.values():
        actuator.stiffness = actuator.damping = actuator.armature = actuator.friction = 0.
        actuator.dynamic_friction = actuator.viscous_friction = 0.
    robot = Articulation(robot_cfg)
    material_paths = []
    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/Robot')):
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI(prim)
            material.CreateStaticFrictionAttr().Set(args.friction)
            material.CreateDynamicFrictionAttr().Set(args.friction)
            material_paths.append(str(prim.GetPath()))
    if not material_paths:
        raise ValueError('No robot collision materials found for friction override')
    sim.reset()
    # The view also exposes disabled legacy shapes with default materials.
    # Set all shape slots explicitly, then verify the actual PhysX readback.
    live_material_tensor = robot.root_physx_view.get_material_properties()
    live_material_tensor[..., :2] = args.friction
    robot.root_physx_view.set_material_properties(live_material_tensor, torch.tensor([0], dtype=torch.int32))
    live_materials = robot.root_physx_view.get_material_properties().cpu().numpy()
    np.testing.assert_allclose(live_materials[..., :2], args.friction, atol=1e-6, rtol=0)
    model = WBCACCTorchModel(args.device)
    references = motion_state(model, motion)
    references['v'] *= args.reference_speed
    order = [robot.joint_names.index(name) for name in model.joint_names]
    limits = robot.data.joint_effort_limits[:, order].double()
    torch.testing.assert_close(limits[0], model.tensor(torque_limits(model.joint_names)), atol=1e-6, rtol=0)
    geometry = CrawlContactGeometry(collision['geometry'], args.device)
    controller = CrawlForceController(model, cfg, geometry, references['q'], references['v'],
        model.tensor(manual_outputs['actions']), fps*args.reference_speed, limits,
        backend=args.qp_backend, friction=args.friction, hold_frame=args.hold_frame,
        action_fps=float(manual_outputs['fps'])*args.reference_speed,
        reference_contact_labels=torch.as_tensor(schedule['contact_labels'], device=args.device))
    initial_index = args.hold_frame or 0
    initial = references['q'][initial_index:initial_index+1]
    initial_v = references['v'][:1] if args.hold_frame is None else torch.zeros_like(references['v'][:1])
    root_pose = torch.cat((initial[:, :3], initial[:, [6, 3, 4, 5]]), -1)
    root_velocity = torch.cat((mv(rotation_xyzw(initial[:, 3:7]), initial_v[:, :3]),
                               mv(rotation_xyzw(initial[:, 3:7]), initial_v[:, 3:6])), -1)
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
    joint_pos[:, order], joint_vel[:, order] = initial[:, 7:].float(), initial_v[:, 6:].float()
    robot.write_root_link_pose_to_sim(root_pose.float())
    robot.write_root_link_velocity_to_sim(root_velocity.float())
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    sim.forward()
    robot.update(dt)

    def actual_state():
        quat = robot.data.root_link_quat_w[:, [1, 2, 3, 0]].double()
        rotation = rotation_xyzw(quat)
        q = torch.cat((robot.data.root_link_pos_w.double(), quat, robot.data.joint_pos[:, order].double()), -1)
        velocity = robot.data.root_link_vel_w.double()
        v = torch.cat((mv(rotation.transpose(-1, -2), velocity[:, :3]),
                       mv(rotation.transpose(-1, -2), velocity[:, 3:]), robot.data.joint_vel[:, order].double()), -1)
        return model.state(q, v)

    state = actual_state()
    # Before the first torque: validate the live articulation against the QP.
    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    coms = robot.root_physx_view.get_coms()[0].cpu().numpy()
    inertias = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
    max_pose_error = 0.
    for name, body in model.description['bodies'].items():
        index = robot.body_names.index(name)
        np.testing.assert_allclose(masses[index], body['mass'], atol=1e-5)
        np.testing.assert_allclose(coms[index, :3], body['com'], atol=1e-5)
        r = Rotation.from_quat(body['principal_axes_xyzw']).as_matrix()
        np.testing.assert_allclose(inertias[index], r@np.diag(body['inertia_diagonal'])@r.T, atol=1e-5)
        max_pose_error = max(max_pose_error, float((robot.data.body_link_pos_w[:, index]-state['frames'][name]['position']).abs().max()))
    if max_pose_error > 2e-4:
        raise ValueError(f'Physics/analytical body-frame mismatch: {max_pose_error}')
    if bool((robot.data.joint_stiffness != 0).any() | (robot.data.joint_damping != 0).any()):
        raise ValueError('Force torques require zero actuator PD gains')
    surface_names = annotations['surface_names'].tolist()
    views = {body: sim.physics_sim_view.create_rigid_contact_view('/World/Robot/'+body,
             filter_patterns=ground_shapes, max_contact_data_count=128) for body in model.body_names}
    if any(view.sensor_count != 1 or view.filter_count != 1 for view in views.values()):
        raise RuntimeError('Ground sensors must resolve one body and one floor shape each')
    body_entries = {body: [entry for entry in collision['entries'] if entry['body'] == body] for body in model.body_names}

    def measured_forces():
        measured = model.tensor(np.zeros((len(surface_names), 3)))
        for body, view in views.items():
            normal, points, normals, _, counts, starts = view.get_contact_data(dt=dt)
            counts, starts = counts.clone(), starts.clone()
            friction, friction_points, friction_counts, friction_starts = view.get_friction_data(dt=dt)
            index = robot.body_names.index(body)
            rotation = rotation_xyzw(robot.data.body_link_quat_w[0, index, [1, 2, 3, 0]].double())
            origin = robot.data.body_link_pos_w[0, index].double()
            for forces, positions, count, start in ((normal.reshape(-1, 1)*normals, points, counts, starts),
                (friction, friction_points, friction_counts, friction_starts)):
                size, offset = int(count.reshape(-1)[0]), int(start.reshape(-1)[0])
                if not size:
                    continue
                if offset+size > len(forces):
                    raise RuntimeError('Contact reporting buffer overflow')
                local = (positions[offset:offset+size].double()-origin)@rotation
                indices = classify_contact_points(local, body_entries[body], surface_names)
                measured.index_add_(0, indices, forces[offset:offset+size].double())
        return measured

    records, trajectory_q, trajectory_t = [], [state['q'][0].cpu().numpy().copy()], [0.]
    substep_contact_forces = []
    applied = torch.zeros_like(robot.data.joint_pos)
    reason, result, steps = '', None, 0
    max_effort_error = 0.
    print(f'CRAWL_PHYSICS_READY: {len(views)} ground sensors, {collision["collider_count"]} collision patches, dt={dt}', flush=True)
    try:
        for step in range(math.ceil(frames*args.substeps/args.reference_speed)):
            time_s = step*dt
            if step % args.qp_every == 0:
                state = actual_state()
                if not bool(torch.isfinite(state['q']).all() & torch.isfinite(state['v']).all()):
                    reason = 'nonfinite_simulator_state'
                    break
                result = controller.compute(state, time_s)
                record_keys = ('torque', 'point_forces', 'wrenches', 'normal_force_limits', 'contact_labels',
                               'actual_available', 'near_floor_available', 'available', 'clearance', 'residual', 'failed',
                               'raw_actions', 'activation', 'requested_normal_force_limits', 'joint_rmse', 'root_error', 'reference_q',
                               'solver_converged', 'solver_optimality_residual', 'solver_complementarity',
                               'acceleration', 'contact_acceleration', 'point_position_w', 'point_velocity_w')
                record = {key: result[key][0].detach().cpu().numpy().copy() for key in record_keys}
                record.update(time_s=time_s, reference_frame=result['reference_frame'],
                              policy_sample=result['policy_sample'],
                              actual_q=state['q'][0].cpu().numpy().copy(), actual_v=state['v'][0].cpu().numpy().copy())
                if bool(result['failed'].any()):
                    reason = 'qp_failed'
                    record['measured_forces'] = measured_forces().cpu().numpy()
                    records.append(record)
                    break
                if (float(result['joint_rmse'].max()) > args.joint_error_limit
                        or float(result['root_error'].max()) > args.root_error_limit):
                    reason = ('joint_tracking_error' if float(result['joint_rmse'].max()) > args.joint_error_limit
                              else 'root_tracking_error')
                    record['measured_forces'] = measured_forces().cpu().numpy()
                    records.append(record)
                    break
                applied[:, order] = result['torque'].float()
            robot.set_joint_effort_target(applied)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            steps += 1
            live_q = torch.cat((robot.data.root_link_pos_w.double(), robot.data.root_link_quat_w[:, [1, 2, 3, 0]].double(),
                                robot.data.joint_pos[:, order].double()), -1)
            trajectory_q.append(live_q[0].cpu().numpy().copy())
            trajectory_t.append(steps*dt)
            max_effort_error = max(max_effort_error, float((robot.data.applied_torque-applied).abs().max()))
            if args.record_contact_substeps or step % args.qp_every == 0:
                measured = measured_forces().cpu().numpy().copy()
                if args.record_contact_substeps:
                    substep_contact_forces.append(measured)
            if step % args.qp_every == 0:
                record['measured_forces'] = measured
                records.append(record)
            if steps % (30*args.substeps) == 0:
                print(f'CRAWL_PHYSICS_STEP {steps/args.substeps:.0f}/{frames}: '
                      f't={steps*dt:.3f}s, joint_rmse={float(result["joint_rmse"][0]):.4f}, '
                      f'root_error={float(result["root_error"][0]):.4f}', flush=True)
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    complete = not reason and steps == math.ceil(frames*args.substeps/args.reference_speed)
    diagnostics = {key: np.asarray([record[key] for record in records]) for key in records[0]} if records else {}
    diagnostics.update(trajectory_q=np.asarray(trajectory_q), trajectory_time_s=np.asarray(trajectory_t),
                       joint_names=np.asarray(model.joint_names), region_names=np.asarray(CRAWL_REGIONS),
                       measured_region_names=np.asarray(surface_names), physics_dt=dt, qp_dt=dt*args.qp_every,
                       torque_limits=limits[0].cpu().numpy(), reference_speed=args.reference_speed)
    if args.record_contact_substeps:
        diagnostics.update(physics_measured_forces=np.asarray(substep_contact_forces),
                           physics_force_time_s=np.asarray(trajectory_t[1:]))
    np.savez_compressed(args.output/'diagnostics.npz', **diagnostics)
    if records and steps*dt > .2 and np.max(np.abs(diagnostics['measured_forces'])) < 1e-6:
        reason, complete = (reason+'; ' if reason else '')+'ground_contact_measurement_missing', False
    inactive_force = float(np.max(np.abs(diagnostics['point_forces'][diagnostics['activation'] <= 0]), initial=0)) if records else None
    report = dict(physics_rollout_ran=steps > 0, simulator='Isaac Sim / PhysX', device=args.device,
                  requested_frames=frames, total_reference_frames=len(motion['root_pos']),
                  physics_steps=steps, simulated_seconds=steps*dt, requested_prefix_completed=complete,
                  reference_speed=args.reference_speed, reference_seconds=steps*dt*args.reference_speed if args.hold_frame is None else 0.,
                  hold_frame=args.hold_frame, root_error_limit_m=args.root_error_limit,
                  joint_error_limit_rad=args.joint_error_limit, evaluation_stage='preliminary',
                  friction_coefficient=args.friction, robot_material_paths=material_paths,
                  verified_collision_friction_range=[float(live_materials[..., :2].min()), float(live_materials[..., :2].max())],
                  full_motion_completed=complete and frames == len(motion['root_pos']) and args.hold_frame is None,
                  failure_reason=reason, control_updates=len(records), physics_dt=dt, qp_dt=dt*args.qp_every,
                  qp_backend=args.qp_backend, qp_solver_tolerance=controller.solver.solver_tolerance,
                  reference_resets_during_rollout=0, state_source='actual articulated body state',
                  actuation='joint effort targets from eight-region FORCE QP; actuator PD, armature and friction zero',
                  max_applied_vs_commanded_torque_error_nm=max_effort_error, initial_model_pose_error_m=max_pose_error,
                  inactive_predicted_force_max_n=inactive_force,
                  max_joint_tracking_rmse_rad=float(diagnostics['joint_rmse'].max()) if records else None,
                  max_root_tracking_error_m=float(diagnostics['root_error'].max()) if records else None,
                  qp_failed_updates=int(diagnostics['failed'].sum()) if records else 0,
                  manual_schedule_sha256=digest(args.schedule), motion_sha256=specification['motion_sha256'],
                  motion_file=str((ROOT/specification['motion']).resolve()),
                  manual_nn_outputs_sha256=digest(args.manual_outputs), reference_contact_mask_applied=False,
                  physics_asset_sha256=collision['asset_sha256'], controller_config=vars(cfg),
                  physics_geometry=collision['geometry'],
                  ground_collision_paths=ground_shapes,
                  contact_measurement='Ground-filtered PhysX normal/friction contacts, assigned by nearest authored body-local convex hull surface.',
                  contact_substeps_recorded=args.record_contact_substeps,
                  contact_substep_samples=len(substep_contact_forces),
                  note='A predicted force bound does not directly clamp measured collision impulses. No reference poses are written after initialization.')
    (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'controller_profile.json').write_text(json.dumps(dict(controller_config=vars(cfg),
        friction=args.friction, root_error_limit_m=args.root_error_limit,
        joint_error_limit_rad=args.joint_error_limit), indent=2)+'\n')
    save_replay_labels(args.output, manual_outputs, diagnostics, report)
    print('CRAWL_PHYSICS_RESULT '+json.dumps(report), flush=True)
    return complete


if __name__ == '__main__':
    success = False
    try:
        success = main()
    except Exception as error:
        (args.output/'startup_failure.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2)+'\n')
        traceback.print_exc()
    finally:
        # On this installation Kit's close() imports Replicator, then hangs
        # initializing graphics on a machine without an NVIDIA device. All
        # artifacts above are synchronously closed; preserve the result code
        # while avoiding that unrelated headless CPU shutdown path.
        if args.headless and args.device == 'cpu' and not torch.cuda.is_available():
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0 if success else 1)
        app.close(wait_for_replicator=False)
    sys.exit(0 if success else 1)
