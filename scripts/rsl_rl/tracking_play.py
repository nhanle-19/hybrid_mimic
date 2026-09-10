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
parser.add_argument("--checkpoint_no", type=int, default=None, help="Checkpoint number to load.")
parser.add_argument("--debug_observations", action="store_true", help="Print the first environment's observation vector every step.")
parser.add_argument("--record_contacts", action="store_true", help="Save control-rate normal contact forces and available model wrenches in the evaluation NPZ.")
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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

body_names = [
    "Trunk", "left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"
]

def _get_body_indexes(command, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    return [command.cfg.body_names.index(name) for name in body_names]

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
        if args_cli.checkpoint_no is not None:
            file = f"model_{args_cli.checkpoint_no}.pt"

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        elif args_cli.motion_file is None:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # create isaac environment
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
        print(f"[INFO] Recording evaluation video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print("[INFO] Policy loaded.", flush=True)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    print(f"[INFO] Policy exported to: {export_model_dir}/policy.onnx", flush=True)
    # reset environment
    obs, _ = env.get_observations()
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    #motion_cmd.time_steps = torch.ones_like(motion_cmd.time_steps, 
    #                                         device = motion_cmd.time_steps.device) * 0
    all_envs = torch.arange(env_cfg.scene.num_envs, device=motion_cmd.time_steps.device, dtype = motion_cmd.time_steps.dtype)
    motion_cmd.reset_command(all_envs)
    #obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    duration = env.unwrapped.command_manager.get_term("motion").motion.time_step_total - 1
    duration = min(duration, 10 * 50)


    sim_action = None
    sim_obs = None
    sim_pos = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    sim_vel = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    sim_angvel = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    ref_pos = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    ref_vel = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    ref_angvel = np.zeros((duration, env_cfg.scene.num_envs, 5, 3))
    sim_ori = np.zeros((duration, env_cfg.scene.num_envs, 4))

    sim_joint_pos = np.zeros((duration, env_cfg.scene.num_envs, 23))
    sim_joint_vel = np.zeros((duration, env_cfg.scene.num_envs, 23))
    sim_joint_torque = np.zeros((duration, env_cfg.scene.num_envs, 23))
    sim_ff_torque = np.zeros((duration, env_cfg.scene.num_envs, 23))
    sim_joint_torque_nle = np.zeros((duration, env_cfg.scene.num_envs, 23))
    sim_des_joint_pos = np.zeros((duration, env_cfg.scene.num_envs, 23))

    total_steps = min(duration, args_cli.video_length) if args_cli.video else duration
    progress_interval = max(1, total_steps // 5)
    contact_samples = {}
    contact_metadata = {}
    if args_cli.record_contacts:
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        contact_metadata = {
            "contact_body_names": np.asarray(contact_sensor.body_names),
            "control_dt": np.asarray(env.unwrapped.step_dt),
            "contact_force_description": np.asarray("Isaac Lab 2.2 net normal contact forces; excludes friction; sampled after env.step"),
        }
        controller = getattr(env.unwrapped, "hybrid_controller", None)
        if controller is not None:
            contact_metadata["model_contact_body_names"] = np.asarray(controller.end_effector_names)
        print("[INFO] Recording contact diagnostics at the control rate.", flush=True)
    print(f"[INFO] Starting evaluation: {env_cfg.scene.num_envs} environments, {total_steps} steps.", flush=True)
    for c in range(duration):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if actions.ndim == 1:
                actions = torch.reshape(actions, (1, -1))
            if sim_action is None:
                sim_action = np.zeros((duration, env_cfg.scene.num_envs, actions.shape[1]))
            
            # env stepping
            
            pol_obs = obs
            if sim_obs is None:
                sim_obs = np.zeros((duration, env_cfg.scene.num_envs, pol_obs.shape[-1]))
            sim_obs[c, :, :] = pol_obs.cpu().numpy()
            if args_cli.debug_observations:
                print(sim_obs[c, 0, :])

            robot = env.unwrapped.scene["robot"]
            command = env.unwrapped.command_manager.get_term("motion")
            action_ = env.unwrapped.action_manager.get_term("joint_pos")
            obs, _, _, _ = env.step(actions)
            if args_cli.record_contacts:
                sample = {
                    "contact_net_forces_w": contact_sensor.data.net_forces_w,
                    "contact_sample_reset": env.unwrapped.reset_buf,
                }
                model_info = getattr(env.unwrapped, "hybrid_rew_info", None)
                if model_info is not None:
                    # Wrenches are predictions from the final physics substep,
                    # not measured forces or a control-interval average.
                    sample["model_contact_wrench_w"] = model_info["grf"].reshape(env_cfg.scene.num_envs, -1, 6)
                    sample["model_force_logits"] = model_info["w"]
                for key, value in sample.items():
                    contact_samples.setdefault(key, []).append(value.detach().cpu().numpy().copy())
            body_ids = _get_body_indexes(command, body_names)

            
            sim_pos_ = command.robot_body_pos_w[:, body_ids, :] - env.unwrapped.scene.env_origins[:, None, :]
            ref_pos_ = command.body_pos_w[:, body_ids, :] - env.unwrapped.scene.env_origins[:, None, :]

            sim_action[c, :, :] = actions.cpu().numpy()
            sim_des_joint_pos[c, :, :] = action_.processed_actions.cpu().numpy()
            sim_pos[c, :, :, :] = sim_pos_.cpu().numpy()
            sim_vel[c, :, :, :] = command.robot_body_lin_vel_w[:, body_ids, :].cpu().numpy()
            sim_angvel[c, :, :, :] = command.robot_body_ang_vel_w[:, body_ids, :].cpu().numpy()
            ref_pos[c, :, :, :] = ref_pos_.cpu().numpy()
            ref_vel[c, :, :, :] = command.body_lin_vel_w[:, body_ids, :].cpu().numpy()
            ref_angvel[c, :, :, :] = command.body_ang_vel_w[:, body_ids, :].cpu().numpy()
            sim_joint_pos[c, :, :] = robot.data.joint_pos.cpu().numpy()
            sim_joint_torque[c, :, :] = robot.data.applied_torque.cpu().numpy()
            sim_ori[c, :, :] = robot.data.root_link_quat_w.cpu().numpy()
            sim_joint_vel[c, :, :] = robot.data.joint_vel.cpu().numpy()
            try:
                #print(env.unwrapped.hybrid_rew_info["ff_tau"])
                sim_ff_torque[c, :, :] = env.unwrapped.hybrid_rew_info["ff_tau"].cpu().numpy()
                sim_joint_torque_nle[c, :, :] = env.unwrapped.hybrid_rew_info["nle"].cpu().numpy()
            except:
                pass
        #jnt_pos = obs["policy"][0, 61:84]
        if (c + 1) % progress_interval == 0 or c + 1 == total_steps:
            print(f"[INFO] Evaluation progress: {c + 1}/{total_steps} steps.", flush=True)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    eval_name = "eval_data/tracking_play_data.npz"
    os.makedirs(os.path.dirname(eval_name), exist_ok=True)
    contact_output = {key: np.stack(values) for key, values in contact_samples.items()}
    contact_output.update(contact_metadata)
    np.savez(eval_name, **{
                            **contact_output,
                            "body_names": np.asarray(body_names),
                            "sim_action": sim_action,
                            "sim_pos": sim_pos,
                            "sim_vel": sim_vel,
                            "sim_angvel": sim_angvel,
                            "ref_pos": ref_pos,
                            "ref_vel": ref_vel,
                            "ref_angvel": ref_angvel,
                            "joint_pos": sim_joint_pos,
                            "joint_vel": sim_joint_vel,
                            "joint_torque": sim_joint_torque,
                            "sim_obs": sim_obs,
                            "sim_ori": sim_ori,
                            "sim_ff_torque": sim_ff_torque,
                            "sim_nle_torque": sim_joint_torque_nle,
                            "sim_des_joint_pos": sim_des_joint_pos
                           })

    print(f"[INFO] Evaluation data saved to: {os.path.abspath(eval_name)}", flush=True)
    # Closing the video wrapper flushes any unfinished recording.
    env.close()
    if args_cli.video:
        print(f"[INFO] Video recording finalized in: {video_kwargs['video_folder']}", flush=True)
    print("[INFO] Evaluation complete.", flush=True)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
