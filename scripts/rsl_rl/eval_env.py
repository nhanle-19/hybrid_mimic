"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np

from isaaclab.app import AppLauncher
# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    if args_cli.task in ("Tracking-Hybrid-T1-v0", "Tracking-Momentum-T1-v0"):
        env_cfg.decimation = 10
        env_cfg.sim.dt = 0.02 / 10
    env_cfg.decimation = 10
    env_cfg.sim.dt = 0.02 / 10
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs = env.get_observations()
    #obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    # Simulate for 700 timesteps
    if args_cli.task == "Tracking-Hybrid-T1-v0":
        eval_name = "eval_data/hybrid_eval_data.npz"
    elif args_cli.task == "Tracking-Momentum-T1-v0":
        eval_name = "eval_data/momentum_eval_data.npz"
    else:
        eval_name = "eval_data/pd_eval_data.npz"

    iters = 8

    duration = 400

    terminated = np.zeros((iters, env_cfg.scene.num_envs, duration), dtype=bool)
    root_vel_error = np.zeros((iters, env_cfg.scene.num_envs, duration))
    root_vel = np.zeros((iters, env_cfg.scene.num_envs, duration))
    forces = np.zeros((iters,))

    for i in range(iters):
        print(f"[INFO] Starting eval iteration {i+1}/{iters}")
        with torch.inference_mode():
            obs, _ = env.reset()
        force_mag = (i**1.5) * 100 + 300.0
        forces[i] = force_mag
        for c in range(duration):
            # run everything in inference mode
            set_random_force(env.unwrapped, c, force_mag)
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                if actions.ndim == 1:
                    actions = torch.reshape(actions, (1, -1))
                # env stepping
                obs, _, terminated_, _ = env.step(actions)
                terminated[i, :, c] = terminated_.cpu().numpy()
                vel_error = motion_global_anchor_velocity_error(env)
                vel = root_velocity(env)
                root_vel_error[i, :, c] = vel_error.cpu().numpy()
                root_vel[i, :, c] = vel.cpu().numpy()
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

    # Save the eval data
    np.savez(eval_name, **{"terminated": terminated,
                           "root_vel_error": root_vel_error,
                           "root_vel": root_vel,
                           "forces": forces})
    # close the simulator
    env.close()

def motion_global_anchor_velocity_error(env) -> torch.Tensor:
    command = env.unwrapped.command_manager.get_term("motion")
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, 0] - command.robot_body_lin_vel_w[:, 0]), dim=-1
    )
    return torch.sqrt(error)

def root_velocity(env) -> torch.Tensor:
    command = env.unwrapped.command_manager.get_term("motion")
    vel = torch.sum(
        torch.square(command.robot_body_lin_vel_w[:, 0]), dim=-1
    )
    return torch.sqrt(vel)

def set_random_force(env, step, frc_mag):

    num_envs = env.num_envs
    # apply random impulse forces to the robot at if timestep meets threshold
    # round(sin(env_num / 6) * 100 + 150 ) == step
    frc_mask = torch.arange(num_envs, device=env.device)
    frc_mask = torch.round(torch.sin(frc_mask / 6) * 75 + 100).to(torch.int64)
    
    frc = torch.randn((num_envs, 3), device=env.device)
    frc = frc * frc_mag / torch.linalg.norm(frc, dim=-1, keepdim=True)

    frc = torch.where(frc_mask[:, None] == step, frc, torch.zeros_like(frc))

    robot = env.scene["robot"]
    num_bodies = robot.num_bodies
    forces_b = torch.zeros((num_envs, 1, 3), device=env.device)
    torques_b = torch.zeros((num_envs, 1, 3), device=env.device)
    forces_b[:, 0, :] = frc  # apply to base link

    body_ids = torch.arange(num_bodies, device=env.device)

    robot.set_external_force_and_torque(forces = forces_b, torques = torques_b, body_ids = [0], is_global = True)

    return

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
