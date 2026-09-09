"""Replay a converted reference motion directly on T1, without policy or physics stepping.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --motion_file retargeted_motion/example_training.npz \
        --headless --video --output_file eval_data/reference_motion.mp4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--registry_name", type=str, help="W&B motion artifact name.")
source.add_argument("--motion_file", type=Path, help="Local converted training NPZ (not the raw retargeted NPZ).")
parser.add_argument("--video", action="store_true", help="Record one motion cycle and exit.")
parser.add_argument("--output_file", type=Path, default=Path("eval_data/reference_motion.mp4"))
parser.add_argument("--video_length", type=int, help="Maximum number of frames; default is the complete motion.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video_length is not None and args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if args_cli.width <= 0 or args_cli.height <= 0:
    parser.error("--width and --height must be positive")
if args_cli.video:
    args_cli.enable_cameras = True
    if args_cli.output_file.suffix.lower() != ".mp4":
        parser.error("--output_file must end in .mp4")
if args_cli.motion_file is not None and not args_cli.motion_file.is_file():
    parser.error(f"Motion file does not exist: {args_cli.motion_file}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from whole_body_tracking.robots.t1 import BOOSTER_T1_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
        ),
    )

    # articulation
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    writer = None
    annotator = None
    render_product = None
    if args_cli.video:
        import imageio.v2 as imageio
        import omni.replicator.core as rep

        args_cli.output_file.parent.mkdir(parents=True, exist_ok=True)
        render_product = rep.create.render_product("/OmniverseKit_Persp", (args_cli.width, args_cli.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach([render_product])
        writer = imageio.get_writer(str(args_cli.output_file), fps=1.0 / sim_dt, codec="libx264", macro_block_size=2)

    frame_count = 0
    frame_limit = min(motion.time_step_total, args_cli.video_length or motion.time_step_total)
    if args_cli.video:
        print(f"[INFO]: Recording {frame_limit} reference frames at {1.0 / sim_dt:g} FPS", flush=True)
    try:
        while simulation_app.is_running():
            index = frame_count % motion.time_step_total
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[index, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[index, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[index, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[index, 0]
            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(
                motion.joint_pos[index].unsqueeze(0), motion.joint_vel[index].unsqueeze(0)
            )
            scene.write_data_to_sim()
            pos_lookat = root_states[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
            sim.render()
            scene.update(sim_dt)

            if writer is not None:
                # Warm up the renderer at frame zero, without advancing the motion.
                if frame_count == 0:
                    for _ in range(10):
                        sim.render()
                rgb = annotator.get_data()
                if rgb.size == 0:
                    raise RuntimeError("Renderer returned no image; reference recording is incomplete.")
                writer.append_data(np.asarray(rgb)[..., :3].copy())
            frame_count += 1
            if args_cli.video and (frame_count == 1 or frame_count % 50 == 0 or frame_count == frame_limit):
                print(f"[INFO]: Recorded {frame_count}/{frame_limit} frames", flush=True)
            if args_cli.video and frame_count >= frame_limit:
                break
    finally:
        if writer is not None:
            print("[INFO]: Finalizing MP4...", flush=True)
            writer.close()
            print(f"[INFO]: MP4 closed: {args_cli.output_file.resolve()}", flush=True)
        if annotator is not None:
            annotator.detach([render_product])
        if render_product is not None:
            render_product.destroy()
    if args_cli.video:
        status = "Saved" if frame_count == frame_limit else "Saved partial recording:"
        print(f"[INFO]: {status} {frame_count} frames to {args_cli.output_file.resolve()}", flush=True)


def resolve_motion_file():
    if args_cli.motion_file is not None:
        return args_cli.motion_file
    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    return pathlib.Path(artifact.download()) / "motion.npz"


def main():
    motion_file = resolve_motion_file()
    with np.load(motion_file) as data:
        required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        if missing := required.difference(data.files):
            raise ValueError(f"Use a converted training NPZ; missing fields: {sorted(missing)}")
        fps = float(np.asarray(data["fps"]).item())
        if not np.isfinite(fps) or fps <= 0 or len(data["joint_pos"]) == 0:
            raise ValueError("Motion must have a positive FPS and at least one frame.")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    motion = MotionLoader(str(motion_file), [0], [0], sim.device)
    if motion.joint_pos.shape[1] != scene["robot"].num_joints:
        raise ValueError("Motion joint count does not match the T1 robot.")
    # Run the simulator
    run_simulator(sim, scene, motion)


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        simulation_app.close()
