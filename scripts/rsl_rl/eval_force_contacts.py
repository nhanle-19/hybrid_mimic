"""Evaluate one FORCE clip from frame zero, retaining contact evidence.

Uses the training torso/QP guards, stops on the first reset, and never wraps the
reference. Optional archived manual actions bypass the learned policy without
changing the controller settings or introducing contact masks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True, type=Path)
parser.add_argument('--motion_file', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--manual_labels', type=Path, help='Archived annotations.npz raw caps; checkpoint supplies configuration only.')
parser.add_argument('--posture_kp', type=float, help='Explicit evaluation-only posture position gain.')
parser.add_argument('--posture_kd', type=float, help='Explicit evaluation-only posture velocity gain.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
controller_overrides = {key: value for key, value in (
    ('posture_position_gain', args.posture_kp), ('posture_velocity_gain', args.posture_kd)) if value is not None}
if any(not math.isfinite(value) or value <= 0 for value in controller_overrides.values()):
    parser.error('Posture gains must be finite and positive')
launcher = AppLauncher(args, fast_shutdown=False)
app = launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.motion import WBCACCSinglePassMotion
from whole_body_tracking.utils.wbc_acc_contact_policy import FEET, sole_vertices
from whole_body_tracking.utils.wbc_contact_mapping import decode_force_values


def array(value):
    return value.detach().cpu().numpy().copy()


def normalized(value):
    """Compare YAML snapshots without constructing Python objects from tags."""
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    if isinstance(value, str):
        if value.lower() in ('true', 'false', 'null', 'none'):
            return {'true': True, 'false': False, 'null': None, 'none': None}[value.lower()]
        try:
            return float(value)
        except ValueError:
            return value
    return value


def main():
    task = 'Tracking-WBC-FORCE-T1-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    checkpoint = args.checkpoint.resolve()
    with np.load(args.motion_file, allow_pickle=False) as motion:
        frame_count = len(motion['joint_pos'])
        fps = float(motion['fps'].item())
    if frame_count < 2 or not np.isclose(fps * cfg.sim.dt * cfg.decimation, 1.):
        raise ValueError('A nonempty clip at the policy frequency is required')
    manual_actions, manual_metadata = None, None
    if args.manual_labels:
        from manual_wbc_schedule import load_force_annotations
        manual_actions, manual_metadata = load_force_annotations(
            args.manual_labels, args.motion_file, frame_count, fps, action_dim=8)
    # Refuse silent controller/network drift from the downloaded training run.
    saved = yaml.load((checkpoint.parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    current_action = cfg.actions.wbc_force.to_dict()
    for key in ('controller', 'friction', 'batched_max_iterations', 'batched_tolerance', 'residual_tolerance'):
        if normalized(saved['actions']['wbc_force'][key]) != normalized(current_action[key]):
            raise ValueError(f'Training/evaluation action setting differs: {key}')
    for actual, expected, name in ((cfg.sim.dt, saved['sim']['dt'], 'physics dt'),
                                  (cfg.decimation, saved['decimation'], 'decimation')):
        if normalized(actual) != normalized(expected):
            raise ValueError(f'Training/evaluation setting differs: {name}')
    # Validate the baseline first, then apply only explicitly requested gains.
    training_controller = normalized(saved['actions']['wbc_force']['controller'])
    for key, value in controller_overrides.items():
        setattr(cfg.actions.wbc_force.controller, key, value)
    evaluation_controller = normalized(cfg.actions.wbc_force.controller.to_dict())
    if controller_overrides:
        print('[CONTACT EVAL] Explicit controller overrides: '+json.dumps(controller_overrides), flush=True)
    saved_agent = yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())
    if saved_agent['policy']['class_name'] != agent.policy.class_name:
        raise ValueError('Checkpoint predates the bounded FORCE mean and /0.9 cap mapping; train a fresh policy')
    agent_config = agent.to_dict()
    for key in ('policy', 'empirical_normalization', 'algorithm'):
        agent_config[key] = saved_agent[key]
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    cfg.commands.motion.class_type = WBCACCSinglePassMotion
    cfg.commands.motion.resampling_time_range = (1.e9, 1.e9)
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    cfg.terminations.motion_complete = None
    cfg.episode_length_s = frame_count / fps + 1.
    cfg.actions.wbc_force.record_diagnostics = True
    cfg.actions.wbc_force.allow_cpu_batched_evaluation = args.device == 'cpu'
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output/'diagnostics.npz'
    if output.exists():
        raise FileExistsError(output)
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    base = env.unwrapped
    records, completed, reason = [], False, ''
    terminations = {name: 0 for name in base.termination_manager.active_terms}
    term = base.action_manager.get_term('wbc_force')
    command = base.command_manager.get_term('motion')
    full_reference = {}
    try:
        obs, _ = env.reset()
        if manual_actions is None:
            runner = OnPolicyRunner(env, agent_config, log_dir=None, device=base.device)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=base.device)
        else:
            manual_actions = torch.as_tensor(manual_actions, device=base.device)
        with torch.inference_mode():
            term._reference_states()
            for field in ('position', 'rotation', 'velocity'):
                full_reference['full_reference_foot_'+field] = array(torch.stack(
                    [term.references['frames'][foot][field] for foot in FEET], 1))
            full_reference['full_reference_q'] = array(term.references['q'])
            full_reference['sole_vertices'] = array(sole_vertices(base.device, torch.float64))
            for step in range(frame_count):
                if not app.is_running():
                    reason = 'Simulator closed'
                    break
                frame = int(command.time_steps[0])
                if frame != step:
                    raise RuntimeError(f'Reference skipped/wrapped: expected {step}, got {frame}')
                state = term._states()
                chosen = policy(obs) if manual_actions is None else manual_actions[frame:frame+1]
                if not torch.isfinite(chosen).all() or not torch.isfinite(obs).all():
                    raise RuntimeError('Nonfinite policy input/output')
                rec = dict(frame=frame, raw_action=array(chosen[0]),
                           cap=array(decode_force_values(chosen).reshape(-1, 2, 4)[0]),
                           actual_q=array(state['q'][0]), reference_q=array(term.references['q'][frame]),
                           pre_ground_force=array(term._gpu_ground_forces()[0]),
                           body_contact_force=array(base.scene.sensors['contact_forces'].data.net_forces_w[0]))
                for field in ('position', 'rotation', 'velocity'):
                    rec['foot_'+field] = array(torch.stack([state['frames'][f][field][0] for f in FEET]))
                obs, _, dones, _ = env.step(chosen)
                failed = bool(dones[0])
                # A terminated environment has already reset; never record its
                # new state as the terminal state of the attempted motion.
                rec['post_valid'] = not failed
                rec['post_ground_force'] = (np.full((2, 3), np.nan) if failed
                                             else array(term._gpu_ground_forces()[0]))
                records.append(rec)
                if failed:
                    for name in terminations:
                        terminations[name] = int(base.termination_manager.get_term(name)[0])
                    reason = f'First termination at step {step+1}: {terminations}'
                    break
                if (step+1) % 100 == 0:
                    print(f'[CONTACT EVAL] {args.motion_file.stem}: {step+1}/{frame_count}', flush=True)
            completed = len(records) == frame_count and not reason
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            motion_file=str(args.motion_file.resolve()), motion_sha256=hashlib.sha256(args.motion_file.read_bytes()).hexdigest(),
            expected_frames=frame_count, attempted_frames=len(records), completed=completed,
            failure_reason=reason, terminations=terminations, seed=args.seed,
            deterministic=True, policy_dt_s=1/fps, physics_dt_s=cfg.sim.dt,
            training_episode_length_s=float(saved['episode_length_s']), evaluation_episode_length_s=cfg.episode_length_s,
            training_controller_matches=training_controller == evaluation_controller,
            controller_overrides=controller_overrides, training_controller=training_controller,
            evaluation_controller=evaluation_controller, reference_preparation=command.reference_preparation,
            qp_contact_model=term.solver.contact_model,
            force_cap_scope=term.solver.force_cap_scope, policy_action_dim=term.action_dim,
            point_force_max_N=array(term.normal_force_max).tolist(),
            qp_contact_points_local=array(term.solver.contact_points(base.device, torch.float64)).tolist(),
            policy_source='learned_policy' if args.manual_labels is None else 'manual_labels',
            manual_labels_file=str(args.manual_labels.resolve()) if args.manual_labels else None,
            manual_labels_sha256=hashlib.sha256(args.manual_labels.read_bytes()).hexdigest() if args.manual_labels else None,
            manual_labels_metadata=manual_metadata,
            contact_policy=('Unmodified learned caps, no geometry gate or manual schedule' if args.manual_labels is None
                            else 'Archived manual raw caps with current controller gains and force maxima; no learned policy or geometry gate'),
            measurement='Pre-step kinematics; diagnostic wrench at first physics substep; post forces at last substep. Reset samples excluded.',
            simulation_device=base.device, torso_guards=True, reference_wrap=False)
        if term.records:
            term.save_diagnostics(output, completed=completed, failure_reason=reason)
            with np.load(output, allow_pickle=False) as archive:
                data = dict(archive)
            data.update(full_reference)
            data.update({'audit_'+key: np.asarray([r[key] for r in records]) for key in records[0]})
            data.update(metadata_json=np.asarray(json.dumps(metadata)),
                        body_contact_names=np.asarray(base.scene.sensors['contact_forces'].body_names),
                        robot_physics_materials=array(base.scene['robot'].root_physx_view.get_material_properties()))
            np.savez_compressed(output, **data)
        (args.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('[CONTACT EVAL] '+json.dumps(metadata), flush=True)
        env.close()
        print('[CONTACT EVAL] Environment closed; all artifacts saved.', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
        print('[CONTACT EVAL] Simulator closed.', flush=True)
