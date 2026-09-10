"""Deterministic evaluation start without changing the baseline motion command."""
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
import torch


def foot_collision(env, sensor_cfg, threshold):
    """Current filtered force; compatible with Isaac versions lacking filtered history."""
    forces = env.scene.sensors[sensor_cfg.name].data.force_matrix_w
    if forces is None:
        raise RuntimeError('Atlas foot-collision reward requires filtered contact forces')
    return (torch.linalg.vector_norm(forces, dim=-1) > threshold).flatten(1).any(dim=1).float()


class AtlasEvaluationMotion(MotionCommand):
    def _resample_command(self, env_ids):
        if len(env_ids):
            self.reset_command(env_ids)
