"""Evaluate the separate Atlas controller and save physics-rate diagnostics."""
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='Tracking-Atlas-T1-Eval-v0')
parser.add_argument('--motion_file', required=True)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=141)
parser.add_argument('--output', default='eval_data/atlas/diagnostics.npz')
parser.add_argument('--contact_schedule', default=None, help='Boolean NPY (motion_frames,2) planned stance mask.')
parser.add_argument('--zero_policy', action='store_true', help='Use zero HybridMimic policy outputs for a pipeline check, not reference tracking.')
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra = parser.parse_known_args()
if not args.zero_policy and args.load_run is None:
    parser.error('Provide --load_run for a retrained Atlas policy, or --zero_policy')
if args.steps < 1:
    parser.error('--steps must be positive')
sys.argv = [sys.argv[0]]+hydra
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks


@hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.commands.motion.motion_file = args.motion_file
    env_cfg.hybrid_controller.record_diagnostics = True
    env_cfg.hybrid_controller.contact_schedule_file = args.contact_schedule
    env_cfg.sim.device = args.device
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    try:
        obs, _ = env.reset()
        if args.zero_policy:
            policy = lambda _: torch.zeros((args.num_envs, env.unwrapped.hybrid_controller.action_dim), device=env.unwrapped.device)
        else:
            path = get_checkpoint_path(str((Path('logs/rsl_rl')/agent_cfg.experiment_name).resolve()),
                                       agent_cfg.load_run, agent_cfg.load_checkpoint)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
            runner.load(path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f'[INFO] Loaded Atlas policy: {path}', flush=True)
        with torch.inference_mode():
            for step in range(args.steps):
                obs, _, _, _ = env.step(policy(obs))
                if (step+1) % max(1, args.steps//5) == 0:
                    print(f'[INFO] Atlas evaluation: {step+1}/{args.steps}', flush=True)
        env.unwrapped.hybrid_controller.save_diagnostics(args.output)
        print(f'[INFO] Saved Atlas diagnostics: {args.output}', flush=True)
    except Exception as error:
        term = env.unwrapped.hybrid_controller
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
