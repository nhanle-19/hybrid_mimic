"""Deterministic evaluation start without changing the baseline motion command."""
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
import torch


def foot_collision(env, sensor_cfg, threshold):
    """Current filtered force; compatible with Isaac versions lacking filtered history."""
    forces = env.scene.sensors[sensor_cfg.name].data.force_matrix_w
    if forces is None:
        raise RuntimeError('WBCACC foot-collision reward requires filtered contact forces')
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).flatten(1).any(dim=1).float()


class WBCACCEvaluationMotion(MotionCommand):
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
