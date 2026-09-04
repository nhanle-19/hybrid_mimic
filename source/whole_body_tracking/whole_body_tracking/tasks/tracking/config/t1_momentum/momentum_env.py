import torch
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.config.t1_hybrid.hybrid_env import (
    HybridEnv,
    get_pd_gains_in_dof_order,
    make_dof_ordered_jointpos_and_action_scale,
)
from whole_body_tracking.utils import momentum_wbc


class MomentumWBCEnv(HybridEnv):
    """HybridMimic environment using a momentum-based whole-body controller."""

    def _initialize_hybrid_runtime(self):
        robot = self.scene["robot"]
        self.hybrid_controller = momentum_wbc.MomentumBasedWholeBodyController(robot, self.cfg.hybrid_controller)
        self.hybrid_rew_info = None
        self.sensor_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=self.hybrid_controller.end_effector_names,
        )

        self.lcc_bias = {
            "com_vel": torch.zeros((self.num_envs, 3), device=self.device),
            "com_angvel": torch.zeros((self.num_envs, 3), device=self.device),
            "mass_fac": torch.ones((self.num_envs,), device=self.device),
            "i_fac": torch.ones((self.num_envs, 3, 3), device=self.device),
            "jac_fac": torch.ones(
                (
                    self.num_envs,
                    self.hybrid_controller.end_effector_count * 6,
                    self.hybrid_controller.joint_count + 6,
                ),
                device=self.device,
            ),
            "pos": torch.zeros(
                (self.num_envs, self.hybrid_controller.end_effector_count + 1, 3),
                device=self.device,
            ),
            "grav_vec": torch.zeros((self.num_envs, 3), device=self.device),
        }
        self.prev_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.sensor_cfg.resolve(self.scene)
        action_scale_cfg = self.cfg.actions.joint_pos.scale
        self.kp, self.kd = get_pd_gains_in_dof_order(robot, self.num_envs, device=self.device)
        self.offset, self.action_scale = make_dof_ordered_jointpos_and_action_scale(
            robot,
            self.num_envs,
            robot.cfg.init_state.joint_pos,
            action_scale_cfg,
        )
        self.hybrid_controller.default_joint_pos = self.offset
        self.hybrid_controller.action_scale = self.action_scale
