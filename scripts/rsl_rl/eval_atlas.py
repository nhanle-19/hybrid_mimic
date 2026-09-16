"""Evaluate the separate Atlas controller and save physics-rate diagnostics."""
import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Tracking-Atlas-T1-Eval-v0')
parser.add_argument('--motion_file', required=True)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--qp_backend', choices=('batched', 'osqp'), default='batched',
                    help='GPU batched QP by default; osqp is the explicit CPU reference backend.')
parser.add_argument('--steps', type=int, default=141)
parser.add_argument('--video', action='store_true', help='Record the rollout to MP4; works with --headless.')
parser.add_argument('--video_dir', default='eval_data/atlas/videos', help='Directory for recorded MP4 files.')
parser.add_argument('--output', default='eval_data/atlas/diagnostics.npz')
parser.add_argument('--manual_support', type=float, nargs=2, default=(1., 1.), metavar=('LEFT', 'RIGHT'),
                    help='Manual support activations for --qp_only (actual geometry still gates support).')
parser.add_argument('--manual_weight', type=float, default=50., help='All six manual motion weights per foot for --qp_only.')
parser.add_argument('--qp_only', '--zero_policy', dest='zero_policy', action='store_true',
                    help='Simulate reference tracking with manual support activations and motion weights.')
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra = parser.parse_known_args()
if not args.zero_policy and args.load_run is None:
    parser.error('Provide --qp_only for controller-only simulation, or --load_run for an Atlas residual policy')
if args.zero_policy and args.load_run is not None:
    parser.error('--qp_only/--zero_policy cannot be combined with --load_run')
if args.steps < 1:
    parser.error('--steps must be positive')
if not all(math.isfinite(x) and 0 <= x <= 1 for x in args.manual_support):
    parser.error('--manual_support must contain two values in [0,1]')
if args.video:
    args.enable_cameras = True
sys.argv = [sys.argv[0]]+hydra
app = AppLauncher(args).app

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
    env_cfg.actions.atlas.record_diagnostics = True
    env_cfg.actions.atlas.backend = args.qp_backend
    controller = env_cfg.actions.atlas.controller
    if not controller.contact_weight_min < args.manual_weight < controller.contact_weight_max:
        raise ValueError('Manual weight must be strictly between configured bounds')
    print('[INFO] Actual collider geometry gates support; all six contact-motion requests are soft.', flush=True)
    env_cfg.sim.device = args.device
    env = gym.make(args.task, cfg=env_cfg, render_mode='rgb_array' if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, step_trigger=lambda step: step == 0,
            video_length=args.steps, name_prefix='atlas-qp' if args.zero_policy else 'atlas-policy',
            fps=round(1 / env.unwrapped.step_dt), disable_logger=True)
        print(f'[INFO] Recording Atlas video to: {Path(args.video_dir).resolve()}', flush=True)
    env = RslRlVecEnvWrapper(env)
    try:
        obs, _ = env.reset()
        if args.zero_policy:
            actions = torch.zeros((args.num_envs, env.unwrapped.action_manager.total_action_dim),
                                  device=env.unwrapped.device)
            actions[:, 0], actions[:, 7] = args.manual_support
            probability = (args.manual_weight-controller.contact_weight_min)/(controller.contact_weight_max-controller.contact_weight_min)
            logits = math.log(probability/(1-probability))
            actions[:, 1:7] = logits
            actions[:, 8:14] = logits
            policy = lambda _: actions
            print('[INFO] QP-only tracking: reference feedback + Atlas QP; manual activations and six motion weights per foot.', flush=True)
        else:
            from rsl_rl.runners import OnPolicyRunner
            path = get_checkpoint_path(str(Path('logs/rsl_rl')/agent_cfg.experiment_name), agent_cfg.load_run, agent_cfg.load_checkpoint)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
            runner.load(path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f'[INFO] Loaded Atlas policy: {path}', flush=True)
        reset_count = 0
        completed = True
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    completed = False
                    break
                obs, _, dones, _ = env.step(policy(obs))
                reset_count += int(dones.sum().item())
                if (step+1) % max(1, args.steps//5) == 0:
                    print(f'[INFO] Atlas evaluation: {step+1}/{args.steps}', flush=True)
        env.unwrapped.action_manager.get_term('atlas').save_diagnostics(args.output, completed=completed,
            failure_reason='' if completed else 'Simulator window closed')
        print(f'[INFO] Episode resets during evaluation: {reset_count} (includes terminations and timeouts).', flush=True)
        print(f'[INFO] Saved Atlas diagnostics: {args.output}', flush=True)
    except Exception as error:
        term = env.unwrapped.action_manager.get_term('atlas')
        if term.records:
            term.save_diagnostics(args.output, finalize=False, completed=False, failure_reason=str(error))
            print(f'[INFO] Saved partial Atlas diagnostics: {args.output}', flush=True)
        raise
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
