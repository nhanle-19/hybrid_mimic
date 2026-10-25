"""Compare constant FORCE/AUTH inputs in serial, first-failure toy sway rollouts.

Candidates are fractions in [0, 1]: two force capacities or six xyz authorities.
Rank by completion/survival, then equally normalized CoM, torso and foot errors.
Fresh-process replay is a separate repeatability check after the search.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from manual_wbc_schedule import encode_contact_values_numpy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--controller', choices=('force', 'authority'), required=True)
parser.add_argument('--motion_file', type=Path, required=True)
parser.add_argument('--candidates', type=Path, required=True, help='JSON list of fraction vectors.')
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--force_max', type=float, nargs=2, metavar=('LEFT_N', 'RIGHT_N'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.force_max is not None and (args.controller != 'force' or
        not all(np.isfinite(x) and x > 0 for x in args.force_max)):
    parser.error('--force_max requires FORCE and two positive finite maxima')
fractions = np.asarray(json.loads(args.candidates.read_text()), dtype=float)
width = 2 if args.controller == 'force' else 6
if (fractions.ndim != 2 or fractions.shape[1] != width or not len(fractions)
        or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions > 1))):
    parser.error(f'Expected a nonempty list of {width}-element fraction vectors in [0, 1]')
with np.load(args.motion_file, allow_pickle=False) as archive:
    frames, fps = len(archive['joint_pos']), float(archive['fps'].item())
if frames < 1 or not np.isclose(fps, 50.):
    parser.error('Expected a nonempty 50 FPS reference')
launcher = AppLauncher(args, fast_shutdown=False)
app = launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.motion import WBCACCSinglePassMotion
from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw
from whole_body_tracking.utils.wbc_acc_gpu_control import rotation_log


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    task = f'Toy-Standing-WBC-{args.controller.upper()}-T1-v0'
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    cfg.commands.motion.class_type = WBCACCSinglePassMotion
    cfg.episode_length_s = frames / fps + 1.
    action_name = 'wbc_' + args.controller
    action_cfg = getattr(cfg.actions, action_name)
    if args.force_max is not None:
        action_cfg.controller.contact_force_max = tuple(args.force_max)
    action_cfg.standing_only = False
    action_cfg.record_diagnostics = True
    action_cfg.allow_cpu_batched_evaluation = args.device == 'cpu'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(task, cfg=cfg)
    rows = []
    try:
        term = env.unwrapped.action_manager.get_term(action_name)
        motion = env.unwrapped.command_manager.get_term('motion')
        robot = env.unwrapped.scene['robot']
        order = [robot.body_names.index(n) for n in term.dynamics.body_names]
        feet = [robot.body_names.index(n) for n in ('left_foot_link', 'right_foot_link')]
        root = robot.body_names.index('Trunk')
        raw = encode_contact_values_numpy(fractions)
        for candidate, values in enumerate(raw):
            env.reset(seed=args.seed)
            term.records.clear()
            if term.references is None:
                term._reference_states()
            actions = torch.as_tensor(values[None], device=env.unwrapped.device, dtype=torch.float32)
            totals = torch.zeros(3, device=env.unwrapped.device, dtype=torch.float64)
            reasons = {}
            for step in range(frames):
                if not app.is_running():
                    raise RuntimeError('Simulator closed during tuning')
                if int(motion.time_steps[0]) != step:
                    raise RuntimeError('Reference frame alignment lost during tuning')
                data = robot.data
                com = (data.body_com_pos_w[:, order].double()*term.dynamics.mass[None, :, None]).sum(1)/term.dynamics.mass.sum()
                com -= env.unwrapped.scene.env_origins
                com_error = (com-term.references['com'][step]).square().sum()
                current_rotation = rotation_xyzw(data.body_quat_w[:, root][:, [1, 2, 3, 0]].double())
                desired_rotation = term.references['frames']['Trunk']['rotation'][step]
                rotation_error = rotation_log(desired_rotation@current_rotation.transpose(-1, -2)).square().sum()
                desired_feet = torch.stack([term.references['frames'][foot]['position'][step]
                                           for foot in ('left_foot_link', 'right_foot_link')])
                actual_feet = data.body_pos_w[:, feet]-env.unwrapped.scene.env_origins[:, None]
                foot_error = (actual_feet-desired_feet).square().sum(-1).mean()
                totals += torch.stack((com_error, rotation_error, foot_error))
                _, _, terminated, timed_out, _ = env.step(actions)
                if bool((terminated | timed_out).any()):
                    manager = env.unwrapped.termination_manager
                    reasons = {name: int(manager.get_term(name)[0]) for name in manager.active_terms}
                    break
            steps = step+1
            errors = (totals/steps).sqrt().cpu().numpy()
            row = dict(candidate=candidate, fractions=fractions[candidate].tolist(), raw_outputs=values.tolist(),
                       completed=not bool(reasons), steps=steps, termination_reasons=reasons,
                       com_rmse_m=float(errors[0]), torso_rmse_rad=float(errors[1]), foot_rmse_m=float(errors[2]),
                       tracking_score=float(errors @ np.array([1/.02, 1/np.deg2rad(2.4), 1/.02])))
            rows.append(row)
            term.save_diagnostics(args.output_dir/f'trial_{candidate:02d}.npz', completed=row['completed'],
                                  failure_reason=json.dumps(reasons) if reasons else '')
            best = min(rows, key=lambda r: (not r['completed'], -r['steps'], r['tracking_score']))
            report = dict(controller=args.controller, motion_file=str(args.motion_file.resolve()),
                          contact_force_max=list(action_cfg.controller.contact_force_max),
                          seed=args.seed, status='requires_fresh_simulator_replay', best_candidate=best['candidate'], trials=rows)
            (args.output_dir/'trials.json').write_text(json.dumps(report, indent=2)+'\n')
            (args.output_dir/'best_inputs.json').write_text(json.dumps({args.controller: best['raw_outputs']}, indent=2)+'\n')
            print(f'[TUNE] {json.dumps(row)}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
