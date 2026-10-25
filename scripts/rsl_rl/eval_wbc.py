"""Evaluate WBC-ACC, WBC-FORCE, or WBC-AUTHORITY and save controller/physics diagnostics."""
import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Tracking-WBC-ACC-T1-Eval-v0')
parser.add_argument('--motion_file', required=True)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--qp_backend', choices=('batched', 'osqp'), default='batched',
                    help='Batched torch QP (CUDA or explicit CPU evaluation); osqp is the CPU reference solver.')
parser.add_argument('--steps', type=int, help='Default: full clip for scheduled actions, otherwise 141.')
parser.add_argument('--nominal_standing', action='store_true', help='QP-only WBC-AUTHORITY standing validation with a grounded symmetric pose instead of the supplied first frame.')
parser.add_argument('--track_motion', action='store_true', help='Toy tasks: advance through the supplied reference once instead of holding frame zero.')
parser.add_argument('--force_max', type=float, nargs=2, metavar=('LEFT_N', 'RIGHT_N'),
                    help='Toy FORCE experiment: independent per-foot maximum force capacities in newtons.')
parser.add_argument('--fail_on_reset', action='store_true', help='Fail validation on the first episode termination or timeout; save partial diagnostics.')
parser.add_argument('--video', action='store_true', help='Record the rollout to MP4; works with --headless.')
parser.add_argument('--record_motion_states', action='store_true',
                    help='Save actual/reference generalized positions during scheduled playback for offline rendering.')
parser.add_argument('--video_dir', help='Defaults to <checkpoint run>/videos/play; QP-only uses eval_data/<controller>/videos.')
parser.add_argument('--output', help='Defaults to <checkpoint run>/diagnostics.npz; QP-only uses eval_data/<controller>/diagnostics.npz.')
parser.add_argument('--manual_support', type=float, nargs=2, default=(1., 1.), metavar=('LEFT', 'RIGHT'),
                    help='Manual per-foot fractions: AUTH xyz cost authority, otherwise force limits; geometry gates support.')
parser.add_argument('--manual_authority', type=float, nargs=6, metavar='C',
                    help='QP-only WBC-AUTHORITY fractions: left xyz, then right xyz (cost only).')
parser.add_argument('--manual_inputs', type=Path,
                    help='JSON with force/authority raw NN output arrays; bypasses manual fraction encoding.')
schedule_group = parser.add_mutually_exclusive_group()
schedule_group.add_argument('--manual_schedule', type=Path, help='Play a per-reference-frame manual action NPZ once.')
schedule_group.add_argument('--build_manual_schedule', type=Path,
                            help='Build a reference-derived action NPZ and CSV, then play it once; requires --qp_only.')
parser.add_argument('--manual_weight', type=float, default=50., help='All six manual motion weights per foot for --qp_only.')
parser.add_argument('--qp_only', '--zero_policy', dest='zero_policy', action='store_true',
                    help='Simulate reference tracking with manual support activations and motion weights.')
parser.add_argument('--force_full_contact', action='store_true',
                    help='QP-only diagnostic: allow all vertices on both feet regardless of actual contact geometry.')
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra = parser.parse_known_args()
toy_task = args.task.startswith('Toy-Standing-WBC-')
if args.track_motion and not toy_task:
    parser.error('--track_motion is only for Toy-Standing-WBC tasks')
if args.force_max is not None and (not toy_task or 'WBC-FORCE' not in args.task or
        not all(math.isfinite(x) and x > 0 for x in args.force_max)):
    parser.error('--force_max requires a toy FORCE task and two positive finite maxima')
scheduled = args.manual_schedule is not None or args.build_manual_schedule is not None
if args.record_motion_states and not scheduled:
    parser.error('--record_motion_states requires a manual schedule')
schedule_controller = 'authority' if 'WBC-AUTHORITY' in args.task else 'force' if 'WBC-FORCE' in args.task else ''
if scheduled:
    if (not args.zero_policy or (toy_task and not args.track_motion)
            or ('Standing' in args.task and not toy_task) or not schedule_controller
            or args.manual_inputs is not None or args.nominal_standing or args.force_full_contact
            or args.num_envs != 1 or any(item.split('=')[0] in
                ('--manual_support', '--manual_authority', '--manual_weight') for item in sys.argv[1:])):
        parser.error('Schedules require a tracking FORCE/AUTHORITY --qp_only task (toy: --track_motion), one environment, and no manual/geometry overrides')
    if args.build_manual_schedule is not None and args.build_manual_schedule.exists():
        parser.error('Schedule output already exists; use --manual_schedule to replay it or choose a new path')
    if (args.manual_schedule or args.build_manual_schedule).suffix != '.npz':
        parser.error('Schedule path must end in .npz')
    args.fail_on_reset = True
elif args.steps is None:
    args.steps = 141
manual_values = None
if toy_task and (not args.zero_policy or (args.manual_inputs is None and not scheduled) or args.nominal_standing):
    parser.error('Toy tasks require --qp_only and manual inputs or a tracking schedule with the supplied reference')
if args.manual_inputs is not None:
    if not args.zero_policy or any(item.split('=')[0] in
            ('--manual_support', '--manual_authority', '--manual_weight') for item in sys.argv[1:]):
        parser.error('--manual_inputs requires --qp_only and cannot be mixed with manual fraction/weight flags')
    from manual_wbc_inputs import load_manual_inputs
    manual_controller = ('authority' if 'WBC-AUTHORITY' in args.task else
                         'force' if 'WBC-FORCE' in args.task else '')
    try:
        manual_values = load_manual_inputs(args.manual_inputs, manual_controller)
    except (ValueError, OSError) as error:
        parser.error(str(error))
if toy_task:
    args.fail_on_reset = True
if not args.zero_policy and args.load_run is None:
    parser.error('Provide --qp_only for controller-only simulation, or --load_run for a learned WBC policy')
if args.zero_policy and args.load_run is not None:
    parser.error('--qp_only/--zero_policy cannot be combined with --load_run')
if args.force_full_contact and not args.zero_policy:
    parser.error('--force_full_contact requires --qp_only')
if args.manual_authority is not None and (not args.zero_policy or not all(math.isfinite(x) and 0 <= x <= 1 for x in args.manual_authority)):
    parser.error('--manual_authority requires --qp_only and six fractions in [0,1]')
if args.steps is not None and args.steps < 1:
    parser.error('--steps must be positive')
if not all(math.isfinite(x) and 0 <= x <= 1 for x in args.manual_support):
    parser.error('--manual_support must contain two values in [0,1]')
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0]]+hydra
# Keep the launcher alive and preserve Python errors/exit status during cleanup.
app_launcher = AppLauncher(args, fast_shutdown=False)
app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import whole_body_tracking.tasks


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.motion.motion_file = args.motion_file
    if toy_task and args.steps is not None:
        # Finish the requested hold before the environment's timeout boundary.
        env_cfg.episode_length_s = args.steps * env_cfg.sim.dt * env_cfg.decimation + 1.
    action_name = next(name for name in ('wbc_authority', 'wbc_force', 'wbc_acc') if hasattr(env_cfg.actions, name))
    authority_policy = action_name == 'wbc_authority'
    if args.manual_authority is not None and not authority_policy:
        raise ValueError('--manual_authority requires a WBC-AUTHORITY task')
    force_only = action_name == 'wbc_force'
    action_cfg = getattr(env_cfg.actions, action_name)
    if args.force_max is not None:
        action_cfg.controller.contact_force_max = tuple(args.force_max)
    if args.track_motion:
        import numpy as np
        from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.motion import WBCACCSinglePassMotion
        with np.load(args.motion_file, allow_pickle=False) as archive:
            if not np.isclose(float(archive['fps'].item())*env_cfg.sim.dt*env_cfg.decimation, 1.):
                raise ValueError('Toy tracking requires reference fps to match the control frequency')
            if not len(archive['joint_pos']) or (args.steps is not None and args.steps > len(archive['joint_pos'])):
                raise ValueError('Toy tracking steps must fit within the nonempty reference clip')
        env_cfg.commands.motion.class_type = WBCACCSinglePassMotion
        action_cfg.standing_only = False
    if args.nominal_standing:
        if not (authority_policy and args.zero_policy and action_cfg.standing_only):
            raise ValueError('--nominal_standing requires a QP-only Standing-WBC-AUTHORITY task')
        from whole_body_tracking.tasks.tracking.config.t1_wbc_authority.motion import WBCAuthorityNominalStandingMotion
        env_cfg.commands.motion.class_type = WBCAuthorityNominalStandingMotion
    action_cfg.record_diagnostics = True
    action_cfg.force_full_contact = args.force_full_contact
    action_cfg.backend = args.qp_backend
    # Deliberate evaluation on CPU uses the same batched algorithm; training
    # keeps its CUDA requirement and never silently falls back.
    action_cfg.allow_cpu_batched_evaluation = args.qp_backend == 'batched' and torch.device(args.device).type == 'cpu'
    controller = action_cfg.controller
    schedule = None
    if scheduled:
        import numpy as np
        from manual_wbc_schedule import load_schedule
        from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.motion import WBCACCSinglePassMotion
        from whole_body_tracking.tasks.tracking.tracking_env_cfg import TerminationsCfg
        from isaaclab.managers import TerminationTermCfg
        from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.rewards import qp_failed
        with np.load(args.motion_file, allow_pickle=False) as archive:
            frame_count = len(archive['joint_pos'])
            reference_fps = float(archive['fps'].item())
        if not np.isclose(reference_fps*env_cfg.sim.dt*env_cfg.decimation, 1.):
            raise ValueError('Scheduled playback requires reference fps to match the control frequency')
        if args.steps is None:
            args.steps = frame_count
        if args.steps > frame_count:
            raise ValueError('Scheduled playback cannot extend beyond the reference clip')
        env_cfg.commands.motion.class_type = WBCACCSinglePassMotion
        env_cfg.episode_length_s = args.steps*env_cfg.sim.dt*env_cfg.decimation+1.
        # Evaluation defaults are permissive; this experiment must detect falls.
        env_cfg.terminations = TerminationsCfg()
        env_cfg.terminations.ee_body_pos.params['body_names'] = list(env_cfg.commands.motion.eef_names)
        env_cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed, params={'action_name': action_name})
        if args.manual_schedule:
            schedule = load_schedule(args.manual_schedule, schedule_controller, args.motion_file,
                                     env_cfg.sim.dt*env_cfg.decimation, frame_count,
                                     controller_config=controller.to_dict())
    # Resolve the checkpoint before creating the environment or video recorder.
    # An absolute root avoids duplicated prefixes in Isaac Lab's path resolver.
    path = None
    if args.zero_policy:
        output_root = (Path('eval_data') / action_name).resolve()
        default_video_dir = output_root / 'videos'
    else:
        log_root = (Path('logs/rsl_rl') / agent_cfg.experiment_name).resolve()
        path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        output_root = Path(path).parent
        default_video_dir = output_root / 'videos' / 'play'
    output = str(Path(args.output).resolve() if args.output else output_root / 'diagnostics.npz')
    video_dir = str(Path(args.video_dir).resolve() if args.video_dir else default_video_dir)
    print(f'[INFO] Evaluation diagnostics: {output}', flush=True)
    if not (force_only or authority_policy) and not controller.contact_weight_min < args.manual_weight < controller.contact_weight_max:
        raise ValueError('Manual weight must be strictly between configured bounds')
    contact_description = ('diagnostic override enables all contact vertices on both feet'
                           if args.force_full_contact else 'actual geometry gates support')
    print(f'[INFO] {action_name}: {contact_description}.'
          + (' Cost-only translational contact authority; fixed normal-force caps.' if authority_policy else
             ' No foot-acceleration objectives.' if force_only else ' Soft foot-acceleration objectives.'), flush=True)
    env_cfg.sim.device = args.device
    env = gym.make(args.task, cfg=env_cfg, render_mode='rgb_array' if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir, step_trigger=lambda step: step == 0,
            video_length=args.steps, name_prefix=f'{action_name}-qp' if args.zero_policy else f'{action_name}-policy',
            fps=round(1 / env.unwrapped.step_dt), disable_logger=True)
        print(f'[INFO] Recording {action_name} video to: {Path(video_dir).resolve()}', flush=True)
    env = RslRlVecEnvWrapper(env)
    reset_count, reset_steps, step_count = 0, [], 0
    termination_counts = {name: 0 for name in env.unwrapped.termination_manager.active_terms}
    executed_frames, executed_actions = [], []
    recorded_actual_q, recorded_reference_q = [], []
    ee_height_errors, anchor_height_errors = [], []
    anchor_position_errors, anchor_orientation_errors, sole_heights = [], [], []

    def save_run(completed, failure_reason=''):
        import json
        import numpy as np
        term = env.unwrapped.action_manager.get_term(action_name)
        term.save_diagnostics(output, completed=completed, failure_reason=failure_reason)
        with np.load(output, allow_pickle=False) as archive:
            data = dict(archive)
        data.update(episode_reset_count=reset_count, episode_reset_steps=np.asarray(reset_steps, dtype=np.int64),
                    policy_steps_completed=step_count, termination_counts_json=np.asarray(json.dumps(termination_counts)))
        # Preserve the actual robot materials and model coefficient so a replay
        # can audit friction overrides instead of assuming task defaults.
        data['qp_friction'] = np.asarray(action_cfg.friction)
        data['robot_physics_materials'] = (
            env.unwrapped.scene['robot'].root_physx_view.get_material_properties()
            .detach().cpu().numpy().copy())
        preparation = getattr(env.unwrapped.command_manager.get_term('motion'), 'reference_preparation', None)
        if preparation is not None:
            data['reference_preparation_json'] = np.asarray(json.dumps(preparation))
        if manual_values is not None:
            data.update(manual_nn_outputs=np.asarray(manual_values),
                        manual_inputs_file=np.asarray(str(args.manual_inputs.resolve())))
        if toy_task:
            data['toy_track_motion'] = np.asarray(args.track_motion)
        if scheduled:
            data.update(manual_schedule_file=np.asarray(str((args.manual_schedule or args.build_manual_schedule).resolve())),
                        manual_schedule_metadata=schedule['metadata_json'],
                        manual_schedule_frames=np.asarray(executed_frames, dtype=np.int64),
                        manual_schedule_actions=np.asarray(executed_actions, dtype=np.float32),
                        manual_actual_contact_gate=np.asarray(False),
                        pre_step_ee_height_error_m=np.asarray(ee_height_errors),
                        pre_step_anchor_height_error_m=np.asarray(anchor_height_errors),
                        pre_step_anchor_position_error_m=np.asarray(anchor_position_errors),
                        pre_step_anchor_orientation_error_rad=np.asarray(anchor_orientation_errors),
                        pre_step_sole_height_m=np.asarray(sole_heights),
                        pre_step_sole_foot_names=np.asarray(FEET),
                        ee_height_error_body_names=np.asarray(env_cfg.commands.motion.eef_names),
                        complete_motion=bool(completed and step_count == len(schedule['actions'])))
        if args.record_motion_states:
            data.update(recorded_actual_q=np.asarray(recorded_actual_q),
                        recorded_reference_q=np.asarray(recorded_reference_q),
                        recorded_motion_time_s=np.asarray(executed_frames)*env.unwrapped.step_dt,
                        recorded_motion_source=np.asarray('Pre-step PhysX states; reference is only a feedback target.'))
        np.savez_compressed(output, **data)

    try:
        obs, _ = env.reset()
        if scheduled:
            from manual_wbc_schedule import build_schedule, save_schedule, reference_digest, load_schedule
            from whole_body_tracking.utils.wbc_acc_contact_policy import FEET, sole_vertices
            from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw
            from isaaclab.utils.math import quat_error_magnitude
            term = env.unwrapped.action_manager.get_term(action_name)
            motion_command = env.unwrapped.command_manager.get_term('motion')
            motion = motion_command.motion
            array = lambda value: value.detach().cpu().numpy()
            digest = reference_digest(array(motion.joint_pos), array(motion._body_pos_w), array(motion._body_quat_w))
            if args.build_manual_schedule:
                term._reference_states()
                ref = term.references
                schedule = build_schedule(schedule_controller, float(motion.fps.item()),
                    np.stack([array(ref['frames'][foot]['position']) for foot in FEET], 1),
                    np.stack([array(ref['frames'][foot]['rotation']) for foot in FEET], 1),
                    np.stack([array(ref['frames'][foot]['velocity']) for foot in FEET], 1),
                    array(ref['com']), float(term.dynamics.mass.sum()), array(sole_vertices()),
                    controller.to_dict(), args.motion_file, digest)
                save_schedule(args.build_manual_schedule, schedule)
                print(f'[INFO] Saved manual {schedule_controller} schedule: {args.build_manual_schedule}', flush=True)
            schedule = load_schedule(args.manual_schedule or args.build_manual_schedule, schedule_controller,
                                     args.motion_file, env.unwrapped.step_dt, motion.time_step_total,
                                     digest, controller.to_dict())
            schedule_actions = torch.as_tensor(schedule['actions'], device=env.unwrapped.device, dtype=torch.float32)
            scheduled_robot = env.unwrapped.scene['robot']
            scheduled_foot_ids = [scheduled_robot.body_names.index(name) for name in FEET]
            scheduled_sole_points = sole_vertices(env.unwrapped.device, torch.float32)
            if args.record_motion_states:
                term._reference_states()
            if int(motion_command.time_steps[0]) != 0:
                raise RuntimeError('Scheduled playback must start at reference frame zero')
        if args.zero_policy:
            actions = torch.zeros((args.num_envs, env.unwrapped.action_manager.total_action_dim),
                                  device=env.unwrapped.device)
            if scheduled:
                actions[:] = schedule_actions[0]
            elif manual_values is not None:
                if len(manual_values) != actions.shape[1]:
                    raise ValueError('Manual NN output count does not match the task action dimension')
                actions[:] = actions.new_tensor(manual_values)
            elif authority_policy:
                from whole_body_tracking.utils.wbc_authority_gpu_control import encode_authorities
                values = args.manual_authority or [args.manual_support[0]]*3+[args.manual_support[1]]*3
                actions[:] = encode_authorities(actions.new_tensor(values).reshape(1, 2, 3))
            elif force_only:
                from whole_body_tracking.utils.wbc_force_gpu_control import encode_force_capacities
                actions[:] = encode_force_capacities(actions.new_tensor([args.manual_support]))
            else:
                actions[:, 0], actions[:, 7] = args.manual_support
                probability = (args.manual_weight-controller.contact_weight_min)/(controller.contact_weight_max-controller.contact_weight_min)
                logits = math.log(probability/(1-probability))
                actions[:, 1:7] = logits
                actions[:, 8:14] = logits
            policy = lambda _: actions
            print(f'[INFO] QP-only tracking: reference feedback + {action_name}; manual policy actions.', flush=True)
        else:
            from rsl_rl.runners import OnPolicyRunner
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
            runner.load(path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f'[INFO] Loaded {action_name} policy: {path}', flush=True)
        completed = True
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    completed = False
                    break
                if scheduled:
                    frame = int(motion_command.time_steps[0])
                    if frame != step:
                        raise RuntimeError(f'Reference/action alignment lost: expected {step}, got {frame}')
                    actions[:] = schedule_actions[frame]
                    executed_frames.append(frame)
                    executed_actions.append(schedule['actions'][frame].copy())
                    if args.record_motion_states:
                        recorded_actual_q.append(array(term._states()['q'][0]).copy())
                        recorded_reference_q.append(array(term.references['q'][frame]).copy())
                    ee_indices = [motion_command.cfg.body_names.index(name) for name in env_cfg.commands.motion.eef_names]
                    ee_height_errors.append(array((motion_command.body_pos_w[:, ee_indices, 2]
                                                   -motion_command.robot_body_pos_w[:, ee_indices, 2]).abs()[0]))
                    anchor_height_errors.append(float((motion_command.anchor_pos_w[:, 2]
                                                       -motion_command.robot_anchor_pos_w[:, 2]).abs()[0]))
                    anchor_position_errors.append(array((motion_command.robot_anchor_pos_w
                                                          -motion_command.anchor_pos_w)[0]))
                    anchor_orientation_errors.append(float(quat_error_magnitude(
                        motion_command.anchor_quat_w, motion_command.robot_anchor_quat_w)[0]))
                    foot_position = (scheduled_robot.data.body_link_pos_w[0, scheduled_foot_ids]
                                     -env.unwrapped.scene.env_origins[0])
                    foot_rotation = rotation_xyzw(
                        scheduled_robot.data.body_link_quat_w[0, scheduled_foot_ids][:, [1, 2, 3, 0]])
                    vertices = scheduled_sole_points@foot_rotation.transpose(-1, -2)+foot_position[:, None]
                    sole_heights.append(array(vertices[..., 2]-controller.ground_height))
                obs, _, dones, _ = env.step(policy(obs))
                step_count = step+1
                resets = int(dones.sum().item())
                reset_count += resets
                if resets:
                    reset_steps.append(step_count)
                    manager = env.unwrapped.termination_manager
                    for name in termination_counts:
                        termination_counts[name] += int(manager.get_term(name).sum().item())
                    if args.fail_on_reset:
                        raise RuntimeError(f'Pre-training validation failed: episode reset at step {step_count}; {termination_counts}')
                if (step+1) % max(1, args.steps//5) == 0:
                    print(f'[INFO] {action_name} evaluation: {step+1}/{args.steps}', flush=True)
        save_run(completed, '' if completed else 'Simulator window closed')
        print(f'[INFO] Episode resets during evaluation: {reset_count} (includes terminations and timeouts).', flush=True)
        print(f'[INFO] Saved {action_name} diagnostics: {output}', flush=True)
    except Exception as error:
        term = env.unwrapped.action_manager.get_term(action_name)
        if term.records:
            save_run(False, str(error))
            print(f'[INFO] Saved partial {action_name} diagnostics: {output}', flush=True)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
