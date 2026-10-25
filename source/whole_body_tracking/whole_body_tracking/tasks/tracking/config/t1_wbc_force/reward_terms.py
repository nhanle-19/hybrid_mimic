"""FORCE rewards in percentage points of one reference clip.

RewardManager multiplies terms by step_dt. Return rates here so each executed
reference frame contributes 100 / frame_count before the configured weight.
"""
import torch
from isaaclab.managers import ManagerTermBase

from whole_body_tracking.utils.wbc_contact_mapping import decode_contact_values
from whole_body_tracking.utils.wbc_acc_contact_policy import FEET, sole_vertices
from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw


def reference_motion_complete(env, command_name='motion'):
    """Successful clip boundary, evaluated after executing the final frame."""
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total-1


class _MotionScaledTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params.get('command_name', 'motion'))
        self.frame_count = self.command.motion.time_step_total
        if self.frame_count < 1 or env.step_dt <= 0:
            raise ValueError('FORCE rewards require a nonempty motion and positive control timestep')
        self.scale = 100. / (self.frame_count * env.step_dt)


class MotionCompletionReward(_MotionScaledTerm):
    """Credit only newly executed frames, with no credit for skipped prefixes."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.last_frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.completed_frames = torch.zeros_like(self.last_frame)

    def __call__(self, env, command_name='motion'):
        frame = self.command.time_steps
        # A held endpoint or an internal clip wrap must not earn extra progress.
        new_frame = (frame > self.last_frame) & (frame >= 0) & (frame < self.frame_count)
        credited = new_frame & ~env.termination_manager.terminated
        self.completed_frames += credited.long()
        self.last_frame.copy_(torch.maximum(self.last_frame, frame))
        return credited.float()*self.scale

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        completed = self.completed_frames[ids]
        if completed.numel():
            self._env.extras.setdefault('log', {})['WBCForce/completed_motion_percent'] = (
                completed.float().mean()*100./self.frame_count)
        self.last_frame[ids] = -1
        self.completed_frames[ids] = 0


class SwingForcePenalty(_MotionScaledTerm):
    """Penalize decoded capacity while the REFERENCE sole is airborne.

    The reference supplies only a reward weight. Actual contact gating and the NN
    outputs sent to the QP are untouched. Toe/heel contact is determined using the
    lowest collision vertex, not the height of the foot-link origin.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low = cfg.params.get('clearance_start', .001)
        high = cfg.params.get('clearance_full', .005)
        if not 0 <= low < high:
            raise ValueError('Swing clearance thresholds must satisfy 0 <= start < full')
        controller = env.action_manager.get_term('wbc_force').cfg.controller
        motion = self.command.motion
        body_ids = [self.command.robot.body_names.index(name) for name in FEET]
        position = motion._body_pos_w[:, body_ids]
        rotation = rotation_xyzw(motion._body_quat_w[:, body_ids][..., [1, 2, 3, 0]])
        vertices = sole_vertices(position.device, position.dtype) @ rotation.transpose(-1, -2)
        clearance = (vertices[..., 2] + position[..., None, 2] - controller.ground_height).amin(-1)
        phase = ((clearance-low)/(high-low)).clamp(0., 1.)
        self.swing_weight = phase.square()*(3.-2.*phase)

    def __call__(self, env, command_name='motion', clearance_start=.001, clearance_full=.005):
        caps = decode_contact_values(env.action_manager.action)
        return (self.swing_weight[self.command.time_steps]*caps.square()).mean(-1)*self.scale


class ForceActionChangePenalty(_MotionScaledTerm):
    """Squared decoded-cap changes, excluding the first action after reset."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(env.action_manager.action)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, command_name='motion'):
        caps = decode_contact_values(env.action_manager.action)
        penalty = (caps-self.previous).square().mean(-1)
        penalty = torch.where(self.valid, penalty, 0.)
        self.previous.copy_(caps)
        self.valid.fill_(True)
        return penalty*self.scale

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0.
        self.valid[ids] = False
