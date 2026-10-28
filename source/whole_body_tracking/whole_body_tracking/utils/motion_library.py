"""NPZ motion storage with explicit clip boundaries for training libraries."""
from pathlib import Path

import numpy as np
import torch


class MotionLibrary:
    def __init__(self, source, device='cpu'):
        source = Path(source).expanduser()
        paths = sorted(source.glob('*.npz')) if source.is_dir() else [source]
        if not paths or any(not p.is_file() for p in paths):
            raise FileNotFoundError(f'No motion NPZ files found: {source}')
        fields = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
                  'body_lin_vel_w', 'body_ang_vel_w')
        chunks, lengths, names = [], [], None
        fps = None
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                rate = float(archive['fps'].item())
                if not np.isfinite(rate) or rate <= 0 or (fps is not None and not np.isclose(rate, fps)):
                    raise ValueError(f'Motion frame rates must be positive and match: {path}')
                fps = rate
                values = {key: archive[key] for key in fields}
                length = len(values['joint_pos'])
                if length < 1 or any(len(a) != length or not np.isfinite(a).all() for a in values.values()):
                    raise ValueError(f'Motion arrays must be nonempty, finite and frame-aligned: {path}')
                if chunks and any(a.shape[1:] != chunks[0][key].shape[1:] for key, a in values.items()):
                    raise ValueError(f'Motion array shapes differ: {path}')
                if len(paths) > 1:
                    if any(key not in archive for key in ('joint_names', 'body_names')):
                        raise ValueError(f'Motion libraries require joint_names and body_names: {path}')
                    ordering = tuple(tuple(archive[key].tolist()) for key in ('joint_names', 'body_names'))
                    if names is not None and ordering != names:
                        raise ValueError(f'Motion joint/body ordering differs: {path}')
                    names = ordering
                chunks.append(values)
                lengths.append(length)
        self.fps = np.asarray(fps)
        for key in fields:
            setattr(self, '_'+key if key.startswith('body_') else key,
                    torch.as_tensor(np.concatenate([c[key] for c in chunks]), dtype=torch.float32, device=device))
        self.clip_paths = [str(p.resolve()) for p in paths]
        self.clip_names = [p.stem for p in paths]
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_ends = self.clip_lengths.cumsum(0)
        self.clip_starts = self.clip_ends-self.clip_lengths
        self.frame_clip_ids = torch.repeat_interleave(torch.arange(len(paths), device=device), self.clip_lengths)
        self.clip_slices = [(int(start), int(start+length)) for start, length in zip(np.cumsum([0]+lengths[:-1]), lengths)]
        self.time_step_total = sum(lengths)
        self.num_clips = len(paths)

    def clip_bounds(self, frames):
        ids = self.frame_clip_ids[frames]
        return self.clip_starts[ids], self.clip_lengths[ids]

    def window_indices(self, frames, offsets):
        starts, lengths = self.clip_bounds(frames)
        ids = frames[:, None]+torch.as_tensor(offsets, device=frames.device)[None]
        return torch.maximum(starts[:, None], torch.minimum(ids, (starts+lengths-1)[:, None]))
