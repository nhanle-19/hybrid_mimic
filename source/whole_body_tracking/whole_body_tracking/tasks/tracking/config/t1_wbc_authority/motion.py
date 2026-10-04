"""Explicit grounded nominal stance for controller-only pre-training validation."""
import torch
from isaaclab.utils.math import quat_from_matrix
from ..t1_wbc_acc.motion import WBCACCStandingMotion
from whole_body_tracking.utils.wbc_acc_torch_model import WBCACCTorchModel
from whole_body_tracking.utils.wbc_acc_contact_policy import sole_vertices, FEET


class WBCAuthorityNominalStandingMotion(WBCACCStandingMotion):
    """Replace the supplied motion with a symmetric, grounded stationary pose.

    Opt-in validation fixture, never a silent change to tracking references.
    Uses actual simulator joint/body ordering and verifies both sole planes.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        robot = env.scene['robot']
        model = WBCACCTorchModel(env.device)
        joint_pos = robot.data.default_joint_pos[:1].clone()
        for side in ('Left', 'Right'):
            for joint, value in (('Hip_Pitch', -.2), ('Hip_Roll', 0.), ('Hip_Yaw', 0.),
                                 ('Knee_Pitch', .42), ('Ankle_Pitch', -.22), ('Ankle_Roll', 0.)):
                joint_pos[:, robot.joint_names.index(f'{side}_{joint}')] = value
        joint_pos[:, robot.joint_names.index('Waist')] = 0.
        limits = robot.data.soft_joint_pos_limits[:1]
        if not bool(((joint_pos >= limits[..., 0]) & (joint_pos <= limits[..., 1])).all()):
            raise ValueError('Nominal validation stance is outside simulator soft joint limits')
        order = [robot.joint_names.index(name) for name in model.joint_names]
        q = torch.zeros((1, 30), device=env.device, dtype=torch.float64)
        q[:, 6], q[:, 7:] = 1., joint_pos[:, order].double()
        v = torch.zeros((1, 29), device=env.device, dtype=torch.float64)
        state = model.state(q, v)
        points = sole_vertices(env.device)
        heights = torch.cat([(points[i]@state['frames'][foot]['rotation'][0].T+
                              state['frames'][foot]['position'][0])[:, 2] for i, foot in enumerate(FEET)])
        if float(heights.max()-heights.min()) > 1e-6:
            raise ValueError('Nominal validation stance must have two level sole planes')
        q[:, 2] += env.cfg.actions.wbc_authority.controller.ground_height-heights.min()
        state = model.state(q, v)
        self.motion.joint_pos[:] = joint_pos
        for name, frame in state['frames'].items():
            body = robot.body_names.index(name)
            self.motion._body_pos_w[:, body] = frame['position'].float()
            self.motion._body_quat_w[:, body] = quat_from_matrix(frame['rotation']).float()
        for name in ('joint_vel', '_body_lin_vel_w', '_body_ang_vel_w'):
            getattr(self.motion, name).zero_()
