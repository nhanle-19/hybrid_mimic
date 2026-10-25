"""Generate T1 standing, sway, sliding, or in-place foot-rotation references."""
import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--side', choices=('left', 'right'), default='left')
parser.add_argument('--tilt_deg', type=float, help='Heel tilt: default 15 degrees, or 0 for --slide/--rotate.')
parser.add_argument('--seconds', type=float, default=10., help='Reference clip duration at 50 FPS.')
motion = parser.add_mutually_exclusive_group()
motion.add_argument('--sway', action='store_true', help='Shift CoM sideways and lean the torso with fixed feet.')
motion.add_argument('--slide', action='store_true', help='Slide --side foot sideways on the floor with a fixed upright torso.')
motion.add_argument('--rotate', action='store_true', help='Yaw --side foot about its fixed sole centre in a supported pose.')
parser.add_argument('--yaw_amplitude_deg', type=float, default=15., help='Rotate: peak yaw either side of neutral.')
parser.add_argument('--toe_pitch_deg', type=float, default=0.,
                    help='Rotate: raise the heel by this pitch angle and pivot about the grounded toe edge.')
parser.add_argument('--support_shift_m', type=float, default=.08, help='Rotate: shift root toward the supporting foot.')
parser.add_argument('--root_lower_m', type=float, default=.02, help='Rotate: lower root from the neutral standing pose.')
parser.add_argument('--support_roll_deg', type=float, default=5., help='Rotate: lean toward the supporting foot.')
parser.add_argument('--slide_amplitude_m', type=float, default=.020, help='Slide: peak displacement from neutral in metres.')
parser.add_argument('--com_sway_m', type=float, default=.020, help='Peak sideways CoM displacement in metres.')
parser.add_argument('--torso_roll_deg', type=float, default=2.4, help='Peak torso lean in degrees, toward the CoM shift.')
parser.add_argument('--period_s', type=float, default=4., help='Motion cycle duration in seconds.')
parser.add_argument('--stance_widen_m', type=float, default=0.,
                    help='Sway: extra total foot separation in metres; each foot moves outward by half.')
parser.add_argument('--output_dir', type=Path,
                    help='Default: eval_data/toy_standing[_sway] or toy_<side>_foot_{slide,rotation,toe_rotation}.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.tilt_deg is None:
    args.tilt_deg = 0. if args.slide or args.rotate else 15.
if not math.isfinite(args.tilt_deg) or not 0 <= args.tilt_deg <= 30:
    parser.error('--tilt_deg must be in [0, 30]')
if not math.isfinite(args.seconds) or args.seconds <= 0:
    parser.error('--seconds must be positive and finite')
if not math.isfinite(args.stance_widen_m) or not 0 <= args.stance_widen_m <= .10:
    parser.error('--stance_widen_m must be in [0, 0.10]')
if args.stance_widen_m and not args.sway:
    parser.error('--stance_widen_m requires --sway')
if not math.isfinite(args.toe_pitch_deg) or not 0 <= args.toe_pitch_deg <= 30.:
    parser.error('--toe_pitch_deg must be in [0, 30]')
if args.toe_pitch_deg and not args.rotate:
    parser.error('--toe_pitch_deg requires --rotate')
if args.slide:
    if args.tilt_deg != 0.:
        parser.error('--slide requires --tilt_deg 0 (both feet flat)')
    if not math.isfinite(args.slide_amplitude_m) or not 0 < args.slide_amplitude_m <= .05:
        parser.error('--slide_amplitude_m must be in (0, 0.05]')
if args.rotate:
    if args.tilt_deg != 0.:
        parser.error('--rotate requires --tilt_deg 0; use --toe_pitch_deg to raise the heel')
    for name, low, high in [('yaw_amplitude_deg', 0., 30.), ('support_shift_m', 0., .12),
                            ('root_lower_m', 0., .06), ('support_roll_deg', 0., 10.)]:
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high or (name == 'yaw_amplitude_deg' and value == 0):
            parser.error(f'Invalid --{name}: expected {low} to {high}')
if args.sway or args.slide or args.rotate:
    if args.seconds < 4.:
        parser.error('--sway/--slide/--rotate requires --seconds >= 4')
    if not math.isfinite(args.period_s) or args.period_s < 2.:
        parser.error('--period_s must be finite and >= 2')
if args.sway:
    if not math.isfinite(args.com_sway_m) or not 0 <= args.com_sway_m <= .06:
        parser.error('--com_sway_m must be in [0, 0.06]')
    if not math.isfinite(args.torso_roll_deg) or not 0 <= args.torso_roll_deg <= 10.:
        parser.error('--torso_roll_deg must be in [0, 10]')
args.output_dir = args.output_dir or Path(f'eval_data/toy_{args.side}_foot_toe_rotation' if args.rotate and args.toe_pitch_deg else
                                        f'eval_data/toy_{args.side}_foot_rotation' if args.rotate else
                                        f'eval_data/toy_{args.side}_foot_slide' if args.slide else
                                        'eval_data/toy_standing_sway' if args.sway else 'eval_data/toy_standing')
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
from whole_body_tracking.utils.toy_standing import heel_up_pose, sway_reference, slide_reference, rotation_reference
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
    if args.sway or args.slide or args.rotate:
        # Reference IK is small and deterministic on CPU, regardless of rollout device.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        reference_fn = rotation_reference if args.rotate else slide_reference if args.slide else sway_reference
        motion_args = dict(yaw_amplitude_deg=args.yaw_amplitude_deg, support_shift_m=args.support_shift_m,
            root_lower_m=args.root_lower_m, support_roll_deg=args.support_roll_deg,
            toe_pitch_deg=args.toe_pitch_deg) if args.rotate else dict(
            slide_amplitude_m=args.slide_amplitude_m) if args.slide else dict(
            tilt_deg=args.tilt_deg, com_sway_m=args.com_sway_m, torso_roll_deg=args.torso_roll_deg,
            stance_widen_m=args.stance_widen_m)
        values = reference_fn(WBCACCTorchModel('cpu'), sole_vertices('cpu'),
            robot.data.default_joint_pos[:1].cpu(), robot.joint_names, robot.body_names,
            robot.data.soft_joint_pos_limits[0].cpu(), side=args.side, seconds=args.seconds,
            period_s=args.period_s, **motion_args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        reference = args.output_dir / 'reference.npz'
        np.savez_compressed(reference, **values)
        print(f'[INFO] {"Rotation" if args.rotate else "Slide" if args.slide else "Sway"} reference: {reference.resolve()} '
              f'({len(values["joint_pos"])} frames)', flush=True)
        return
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
