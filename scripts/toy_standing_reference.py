"""Generate a stationary, heel-grounded toes-up T1 reference for WBC experiments."""
import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--side', choices=('left', 'right'), default='left')
parser.add_argument('--tilt_deg', type=float, default=15.)
parser.add_argument('--seconds', type=float, default=10., help='Reference clip duration at 50 FPS.')
parser.add_argument('--output_dir', type=Path, default=Path('eval_data/toy_standing'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.tilt_deg) or not 0 <= args.tilt_deg <= 30:
    parser.error('--tilt_deg must be in [0, 30]')
if not math.isfinite(args.seconds) or args.seconds <= 0:
    parser.error('--seconds must be positive and finite')
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix
from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.t1 import BOOSTER_T1_CFG
from whole_body_tracking.utils.toy_standing import heel_up_pose
from whole_body_tracking.utils.wbc_acc_torch_model import WBCACCTorchModel
from whole_body_tracking.utils.wbc_acc_contact_policy import sole_vertices


@configclass
class ToySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.UsdFileCfg(
        usd_path=f'{ASSET_DIR}/booster/t1/wbc_acc_ground.usda'))
    light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=1500.))
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002, device=args.device))
    cfg = ToySceneCfg(num_envs=1, env_spacing=2.)
    scene = InteractiveScene(cfg)
    sim.reset()
    robot = scene['robot']
    model = WBCACCTorchModel(sim.device)
    points = sole_vertices(sim.device)
    target, _, state = heel_up_pose(model, points, robot.data.default_joint_pos,
                                   robot.joint_names, args.side, args.tilt_deg)
    limits = robot.data.soft_joint_pos_limits
    if not bool(((target >= limits[..., 0]) & (target <= limits[..., 1])).all()):
        raise ValueError('Toy pose exceeds simulator soft joint limits')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def array(tensor):
        return np.repeat(tensor.detach().cpu().numpy(), math.ceil(args.seconds*50), axis=0)

    # Native simulator joint/body ordering works with both standing WBC tasks.
    # Repeat the stationary pose to make a reference clip suitable for video.
    body_pos = torch.stack([state['frames'][n]['position'] for n in robot.body_names], 1).float()
    body_quat = torch.stack([quat_from_matrix(state['frames'][n]['rotation'])
                             for n in robot.body_names], 1).float()
    reference = args.output_dir / 'reference.npz'
    np.savez_compressed(reference, fps=np.array(50.), joint_pos=array(target),
                        joint_vel=array(torch.zeros_like(target)), body_pos_w=array(body_pos),
                        body_quat_w=array(body_quat), body_lin_vel_w=array(torch.zeros_like(body_pos)),
                        body_ang_vel_w=array(torch.zeros_like(body_pos)),
                        joint_names=np.asarray(robot.joint_names), body_names=np.asarray(robot.body_names),
                        tilted_side=args.side, tilt_deg=args.tilt_deg)
    print(f'[INFO] Reference: {reference.resolve()}', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
