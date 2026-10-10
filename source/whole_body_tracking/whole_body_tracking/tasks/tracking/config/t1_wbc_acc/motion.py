"""Deterministic evaluation start without changing the baseline motion command."""
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
import torch


def foot_collision(env, sensor_cfg, threshold):
    """Current filtered force; compatible with Isaac versions lacking filtered history."""
    forces = env.scene.sensors[sensor_cfg.name].data.force_matrix_w
    if forces is None:
        raise RuntimeError('WBCACC foot-collision reward requires filtered contact forces')
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).flatten(1).any(dim=1).float()


class WBCACCTrainingMotion(MotionCommand):
    """Use one corrected reference for reset, observations, rewards, and the QP."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from whole_body_tracking.utils.wbc_reference import prepare_wbc_reference
        root = self.robot.body_names.index('Trunk')
        action_name = next(name for name in ('wbc_authority', 'wbc_force', 'wbc_acc') if hasattr(env.cfg.actions, name))
        controller = getattr(env.cfg.actions, action_name).controller
        values, self.reference_preparation = prepare_wbc_reference(
            self.motion.joint_pos.cpu().numpy(), self.motion._body_pos_w[:, root].cpu().numpy(),
            self.motion._body_quat_w[:, root].cpu().numpy(), self.robot.joint_names, self.robot.body_names,
            self.robot.data.soft_joint_pos_limits[0].cpu().numpy(), float(self.motion.fps[0]), controller.ground_height)
        for name, value in values.items():
            target = getattr(self.motion, '_'+name if name.startswith('body_') else name)
            target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))
        print(f'[INFO] WBC reference preparation: {self.reference_preparation}', flush=True)


class WBCACCEvaluationMotion(WBCACCTrainingMotion):
    def _resample_command(self, env_ids):
        if len(env_ids):
            self.reset_command(env_ids)


class WBCACCStandingMotion(WBCACCEvaluationMotion):
    """Hold the first pose indefinitely, with zero reference velocity."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name in ('joint_pos', '_body_pos_w', '_body_quat_w'):
            values = getattr(self.motion, name)
            values[:] = values[:1].clone()
        for name in ('joint_vel', '_body_lin_vel_w', '_body_ang_vel_w'):
            getattr(self.motion, name).zero_()

    def _update_command(self):
        # The parent advances by one before updating relative reference poses.
        # Keep frame zero without triggering an end-of-clip robot reset.
        self.time_steps.fill_(-1)
        super()._update_command()
