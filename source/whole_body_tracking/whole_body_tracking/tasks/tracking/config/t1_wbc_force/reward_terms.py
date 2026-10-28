"""FORCE rewards in percentage points of one reference clip.

RewardManager multiplies terms by step_dt. Return rates here so each executed
reference frame contributes 100 / frame_count before the configured weight.
"""
import torch
from isaaclab.managers import ManagerTermBase

from whole_body_tracking.utils.wbc_contact_mapping import decode_contact_values
from whole_body_tracking.utils.wbc_acc_contact_policy import FEET, sole_vertices
from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw
from whole_body_tracking.utils.wbc_force_learning import (
    ForceMetrics, swing_weights, unloading_weights,
)


def _metrics(env):
    if not hasattr(env, '_wbc_force_metrics'):
        env._wbc_force_metrics = ForceMetrics()
    return env._wbc_force_metrics


def _reference_clearance(env):
    """Cache collider-derived heights in the robot's full body ordering."""
    if not hasattr(env, '_wbc_force_reference_clearance'):
        command = env.command_manager.get_term('motion')
        controller = env.action_manager.get_term('wbc_force').cfg.controller
        body_ids = [command.robot.body_names.index(name) for name in FEET]
        motion = command.motion
        position = motion._body_pos_w[:, body_ids]
        rotation = rotation_xyzw(motion._body_quat_w[:, body_ids][..., [1, 2, 3, 0]])
        vertices = sole_vertices(position.device, position.dtype) @ rotation.transpose(-1, -2)
        heights = (vertices[..., 2]+position[..., None, 2]-controller.ground_height).amin(-1)
        env._wbc_force_reference_clearance = (heights, body_ids, controller.ground_height)
    return env._wbc_force_reference_clearance


def reference_motion_complete(env, command_name='motion'):
    """Successful clip boundary, evaluated after executing the final frame."""
    command = env.command_manager.get_term(command_name)
    starts, lengths = _clip_bounds(command)
    return command.time_steps >= starts+lengths-1


def _clip_bounds(command):
    if hasattr(command.motion, 'clip_bounds'):
        return command.motion.clip_bounds(command.time_steps)
    return torch.zeros_like(command.time_steps), torch.full_like(command.time_steps, command.motion.time_step_total)


class _MotionScaledTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.params.get('command_name', 'motion'))
        self.frame_count = self.command.motion.time_step_total
        if self.frame_count < 1 or env.step_dt <= 0:
            raise ValueError('FORCE rewards require a nonempty motion and positive control timestep')
        self.step_dt = env.step_dt

    @property
    def scale(self):
        return 100. / (_clip_bounds(self.command)[1] * self.step_dt)


class MotionCompletionReward(_MotionScaledTerm):
    """Credit only newly executed frames, with no credit for skipped prefixes."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.last_frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.completed_frames = torch.zeros_like(self.last_frame)
        self.start_frame = torch.full_like(self.last_frame, -1)
        self.episode_clip_start = torch.zeros_like(self.last_frame)
        self.episode_clip_length = torch.full_like(self.last_frame, self.frame_count)
        self.episode_clip_id = torch.zeros_like(self.last_frame)
        self.previous_caps = torch.zeros((env.num_envs, 2), device=env.device)
        self.previous_reference = torch.zeros_like(self.previous_caps)
        self.zero_swing_steps = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)
        self.metrics = _metrics(env)

    def __call__(self, env, command_name='motion'):
        frame = self.command.time_steps
        starts, lengths = _clip_bounds(self.command)
        local_frame = frame-starts
        first = self.start_frame < 0
        self.start_frame[first] = local_frame[first]
        self.episode_clip_start[first] = starts[first]
        self.episode_clip_length[first] = lengths[first]
        if hasattr(self.command.motion, 'frame_clip_ids'):
            self.episode_clip_id[first] = self.command.motion.frame_clip_ids[frame[first]]
        self.metrics.count('Sampling/start_count', first)
        self.metrics.mean('Sampling/start_phase_percent', local_frame.float()*100./lengths, first)
        self.metrics.mean('Sampling/start_frame_zero_percent', (local_frame == 0).float()*100., first)
        self.metrics.mean('Sampling/start_final_sixth_percent', (local_frame >= lengths*5/6).float()*100., first)
        for clip, name in self._clip_names():
            self.metrics.count(f'Motion/{name}/start_count', first & (self.episode_clip_id == clip))
        # A held endpoint or an internal clip wrap must not earn extra progress.
        new_frame = (frame > self.last_frame) & (frame >= 0) & (frame < self.frame_count)
        credited = new_frame & ~env.termination_manager.terminated
        self.completed_frames += credited.long()
        self.last_frame.copy_(torch.maximum(self.last_frame, frame))
        self._record_step_metrics(env, frame, first, credited)
        return credited.float()*self.scale

    def _record_step_metrics(self, env, frame, first, credited):
        reference = _reference_clearance(env)[0][frame]
        caps = decode_contact_values(env.action_manager.action)
        self.metrics.mean('Caps/both_zero_percent', (caps == 0).all(-1).float()*100.)
        for i, side in enumerate(('left', 'right')):
            swing, stance = reference[:, i] >= .005, reference[:, i] <= .001
            zero = (caps[:, i] == 0).float()*100.
            self.metrics.count(f'Contact/{side}_swing_samples', swing)
            self.metrics.mean(f'Contact/{side}_swing_zero_cap_percent', zero, swing)
            self.metrics.mean(f'Contact/{side}_swing_cap_mean', caps[:, i], swing)
            liftoff = (reference[:, i] > .001) & (self.previous_reference[:, i] <= .001) & ~first
            self.metrics.count(f'Contact/{side}_liftoff_count', liftoff)
            self.metrics.mean(f'Contact/{side}_liftoff_zero_cap_percent', zero, liftoff)
            holding = swing & (caps[:, i] == 0)
            ended = (self.zero_swing_steps[:, i] > 0) & ~holding
            self.metrics.mean(f'Contact/{side}_zero_hold_ms', self.zero_swing_steps[:, i].float()*env.step_dt*1000., ended)
            self.zero_swing_steps[:, i] = torch.where(holding, self.zero_swing_steps[:, i]+1, 0)
            if hasattr(env.scene, 'sensors'):
                sensor = env.scene.sensors[f'{side}_foot_ground_contact']
                fz = sensor.data.force_matrix_w[:, 0, 0, 2]
                self.metrics.mean(f'Contact/{side}_swing_measured_fz_N', fz, swing)
            self.metrics.mean(f'Caps/{side}_stance_zero_percent', zero, stance)
            self.metrics.mean(f'Caps/{side}_mean', caps[:, i])
            self.metrics.mean(f'Caps/{side}_absolute_change', (caps[:, i]-self.previous_caps[:, i]).abs(), ~first)
        self.metrics.mean('Reward/completion_points_per_step', credited.float()*self.scale*env.step_dt*self.cfg.weight)
        for clip, name in self._clip_names():
            selected = self.episode_clip_id == clip
            self.metrics.count(f'Motion/{name}/policy_steps', selected)
            for i, side in enumerate(('left', 'right')):
                self.metrics.mean(f'Motion/{name}/{side}_cap_mean', caps[:, i], selected)
                self.metrics.mean(f'Motion/{name}/{side}_zero_cap_percent', (caps[:, i] == 0).float()*100., selected)
                self.metrics.mean(f'Motion/{name}/{side}_cap_change', (caps[:, i]-self.previous_caps[:, i]).abs(), selected & ~first)
        self.previous_caps.copy_(caps)
        self.previous_reference.copy_(reference)

    def _clip_names(self):
        motion = self.command.motion
        return enumerate(motion.clip_names) if getattr(motion, 'num_clips', 1) > 1 else ()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        completed, start = self.completed_frames[ids], self.start_frame[ids]
        lengths = self.episode_clip_length[ids]
        valid = start >= 0
        failed = self._env.termination_manager.terminated[ids]
        success = (self.last_frame[ids] >= self.episode_clip_start[ids]+lengths-1) & ~failed
        full = valid & (start == 0)
        available = (lengths-start).clamp_min(1)
        self.metrics.count('Episode/count', valid)
        self.metrics.mean('Episode/start_phase_percent', start.float()*100./lengths, valid)
        self.metrics.mean('Episode/available_suffix_steps', available.float(), valid)
        self.metrics.mean('Episode/executed_clip_percent', completed.float()*100./lengths, valid)
        self.metrics.mean('Episode/suffix_completed_percent', completed.float()*100./available, valid)
        self.metrics.mean('Episode/suffix_success_percent', success.float()*100., valid)
        self.metrics.mean('Episode/failure_percent', failed.float()*100., valid)
        self.metrics.count('FullMotion/episode_count', full)
        self.metrics.mean('FullMotion/completed_percent', completed.float()*100./lengths, full)
        self.metrics.mean('FullMotion/success_percent', success.float()*100., full)
        for clip, name in self._clip_names():
            selected = valid & (self.episode_clip_id[ids] == clip)
            prefix = f'Motion/{name}'
            self.metrics.count(f'{prefix}/episode_count', selected)
            self.metrics.mean(f'{prefix}/success_percent', success.float()*100., selected)
            self.metrics.mean(f'{prefix}/executed_clip_percent', completed.float()*100./lengths, selected)
            self.metrics.count(f'{prefix}/full_episode_count', selected & full)
            self.metrics.mean(f'{prefix}/full_success_percent', success.float()*100., selected & full)
        manager = self._env.termination_manager
        for i, side in enumerate(('left', 'right')):
            hold = self.zero_swing_steps[ids, i]
            self.metrics.mean(f'Contact/{side}_zero_hold_ms', hold.float()*self._env.step_dt*1000., hold > 0)
        for name in ('anchor_pos', 'anchor_ori', 'qp_infeasible'):
            if name in manager.active_terms:
                self.metrics.mean(f'Failure/{name}_percent', manager.get_term(name)[ids].float()*100., valid)
        self.last_frame[ids] = -1
        self.completed_frames[ids] = 0
        self.start_frame[ids] = -1
        self.zero_swing_steps[ids] = 0


class SwingForcePenalty(_MotionScaledTerm):
    """Penalize capacity during reference swing and a short unloading lead-in.

    The reference supplies only a reward weight. Actual contact gating and the NN
    outputs sent to the QP are untouched. Toe/heel contact is determined using the
    lowest collision vertex, not the height of the foot-link origin.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low = cfg.params.get('clearance_start', .001)
        high = cfg.params.get('clearance_full', .005)
        clearance = _reference_clearance(env)[0]
        self.swing_weight = swing_weights(clearance, low, high)
        slices = getattr(self.command.motion, 'clip_slices', [(0, self.frame_count)])
        self.unloading_weight = torch.cat([unloading_weights(self.swing_weight[start:end],
            cfg.params.get('unload_lead_time', 0.), env.step_dt) for start, end in slices])

    def __call__(self, env, command_name='motion', clearance_start=.001, clearance_full=.005, unload_lead_time=0.):
        caps = decode_contact_values(env.action_manager.action)
        value = (self.unloading_weight[self.command.time_steps]*caps.square()).mean(-1)*self.scale
        _metrics(env).mean('Reward/swing_cap_points_per_step', value*env.step_dt*self.cfg.weight)
        return value


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
        value = penalty*self.scale
        _metrics(env).mean('Reward/cap_change_points_per_step', value*env.step_dt*self.cfg.weight)
        return value

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0.
        self.valid[ids] = False
