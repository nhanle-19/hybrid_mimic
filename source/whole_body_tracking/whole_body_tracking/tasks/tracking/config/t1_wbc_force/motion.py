"""Sample clips equally, adapt phases within each clip, and preserve boundaries."""
import math
import torch
from ..t1_wbc_acc.motion import WBCACCTrainingMotion


class WBCForceTrainingMotion(WBCACCTrainingMotion):
    supports_motion_library = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.motion.num_clips > 1:
            counts = [int((end-start)*env.step_dt)+1 for start, end in self.motion.clip_slices]
            self.clip_bin_slices = []
            frame_bins, bin_clips, local_bins, counts_per_bin = [], [], [], []
            offset = 0
            for clip, ((start, end), count) in enumerate(zip(self.motion.clip_slices, counts)):
                self.clip_bin_slices.append((offset, offset+count))
                frame_bins.append(torch.arange(end-start, device=self.device)*count//(end-start)+offset)
                bin_clips.extend([clip]*count)
                local_bins.extend(range(count))
                counts_per_bin.extend([count]*count)
                offset += count
            self.frame_bins = torch.cat(frame_bins)
            self.bin_clips = torch.tensor(bin_clips, device=self.device)
            self.local_bins = torch.tensor(local_bins, device=self.device)
            self.counts_per_bin = torch.tensor(counts_per_bin, device=self.device)
            self.bin_count = offset
            self.bin_failed_count = torch.zeros(offset, device=self.device)
            self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
            print('[INFO] Equal-probability motion library: '+', '.join(
                f'{name} ({end-start} frames)' for name, (start, end) in zip(self.motion.clip_names, self.motion.clip_slices)), flush=True)

    def _adaptive_sampling(self, env_ids):
        if self.motion.num_clips == 1:
            return super()._adaptive_sampling(env_ids)
        probability = self.cfg.start_from_beginning_probability
        if not 0. <= probability <= 1.:
            raise ValueError('start_from_beginning_probability must lie in [0, 1]')
        failed = self._env.termination_manager.terminated[env_ids]
        bins = self.frame_bins[self.time_steps[env_ids]][failed]
        self._current_bin_failed += torch.bincount(bins, minlength=self.bin_count)
        pieces = []
        for start, end in self.clip_bin_slices:
            weights = self.bin_failed_count[start:end]+self.cfg.adaptive_uniform_ratio/(end-start)
            weights = torch.nn.functional.pad(weights[None, None],
                (0, self.cfg.adaptive_kernel_size-1), mode='replicate')
            weights = torch.nn.functional.conv1d(weights, self.kernel[None, None]).flatten()
            pieces.append(weights/weights.sum()/self.motion.num_clips)
        probabilities = torch.cat(pieces)
        sampled = torch.multinomial(probabilities, len(env_ids), replacement=True)
        clips = self.bin_clips[sampled]
        lengths, starts = self.motion.clip_lengths[clips], self.motion.clip_starts[clips]
        frames = ((self.local_bins[sampled]+torch.rand(len(env_ids), device=self.device))
                  /self.counts_per_bin[sampled]*(lengths-1)).long()
        frames = torch.where(torch.rand(len(env_ids), device=self.device) < probability, 0, frames)
        self.time_steps[env_ids] = starts+frames
        probabilities *= 1.-probability
        for start, _ in self.clip_bin_slices:
            probabilities[start] += probability/self.motion.num_clips
        entropy = -(probabilities*(probabilities+1e-12).log()).sum()/math.log(self.bin_count)
        maximum, index = probabilities.max(0)
        self.metrics['sampling_entropy'][:] = entropy
        self.metrics['sampling_top1_prob'][:] = maximum
        self.metrics['sampling_top1_bin'][:] = index.float()/self.bin_count

    def _update_command(self):
        # ManagerBasedRLEnv resets done environments before command.compute().
        # They have not executed an action at their newly sampled phase yet.
        # The parent advances one frame, so cancel that advance only on reset.
        starts, lengths = self.motion.clip_bounds(self.time_steps)
        self.time_steps -= (self._env.episode_length_buf == 0).long()
        # End-of-clip termination normally resets first; clamp defensively so
        # neither a held endpoint nor a missed reset enters the next clip.
        self.time_steps.copy_(torch.minimum(self.time_steps, starts+lengths-2))
        super()._update_command()
