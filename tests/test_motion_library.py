"""Mixed motion boundaries, independent normalization, and balanced sampling."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/whole_body_tracking/whole_body_tracking'
spec = importlib.util.spec_from_file_location('motion_library_test', SOURCE/'utils/motion_library.py')
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)


def write_clip(path, frames, value=0., fps=50., names=('a', 'b')):
    np.savez(path, fps=fps, joint_pos=np.full((frames, 2), value), joint_vel=np.zeros((frames, 2)),
             body_pos_w=np.zeros((frames, 2, 3)), body_quat_w=np.tile([1., 0., 0., 0.], (frames, 2, 1)),
             body_lin_vel_w=np.zeros((frames, 2, 3)), body_ang_vel_w=np.zeros((frames, 2, 3)),
             joint_names=np.asarray(names), body_names=np.asarray(['left_foot_link', 'right_foot_link']))


def test_library_preserves_clip_boundaries_and_clamps_lookahead(tmp_path):
    write_clip(tmp_path/'a.npz', 3, value=1.)
    write_clip(tmp_path/'b.npz', 5, value=100.)
    motion = library.MotionLibrary(tmp_path)
    assert motion.clip_slices == [(0, 3), (3, 8)]
    ids = motion.window_indices(torch.tensor([1, 2, 3, 7]), [-3, 0, 1, 10])
    assert ids.tolist() == [[0, 1, 2, 2], [0, 2, 2, 2], [3, 3, 4, 7], [4, 7, 7, 7]]
    assert motion.joint_pos[ids[:2]].max() == 1.
    assert motion.joint_pos[ids[2:]].min() == 100.


@pytest.mark.parametrize('kwargs,message', [({'fps': 60.}, 'frame rates'),
    ({'names': ('b', 'a')}, 'ordering'), ({'frames': 0}, 'nonempty'), ({'value': np.nan}, 'finite')])
def test_library_rejects_incompatible_clips(tmp_path, kwargs, message):
    write_clip(tmp_path/'a.npz', 3)
    options = dict(frames=4)
    options.update(kwargs)
    write_clip(tmp_path/'b.npz', **options)
    with pytest.raises(ValueError, match=message):
        library.MotionLibrary(tmp_path)


def test_empty_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        library.MotionLibrary(tmp_path)


def motion_class(motion):
    tree = ast.parse((SOURCE/'tasks/tracking/config/t1_wbc_force/motion.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    class Parent:
        def __init__(self, cfg, env):
            self.cfg, self._env, self.motion, self.device = cfg, env, motion, 'cpu'
            self.kernel = torch.ones(1)
            self.time_steps = torch.zeros(env.num_envs, dtype=torch.long)
            self.metrics = {k: torch.zeros(env.num_envs) for k in
                            ('sampling_entropy', 'sampling_top1_prob', 'sampling_top1_bin')}

        def _update_command(self):
            self.time_steps += 1

    ns = dict(WBCACCTrainingMotion=Parent, torch=torch, math=math)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<multi_motion>', 'exec'), ns)
    return ns['WBCForceTrainingMotion']


def test_equal_clip_sampling_survives_length_and_failure_imbalance(tmp_path):
    write_clip(tmp_path/'a.npz', 100)
    write_clip(tmp_path/'b.npz', 1000)
    motion = library.MotionLibrary(tmp_path)
    cfg = NS(start_from_beginning_probability=.2, adaptive_uniform_ratio=.1, adaptive_kernel_size=1)
    env = NS(num_envs=20000, step_dt=.02, termination_manager=NS(terminated=torch.zeros(20000, dtype=torch.bool)))
    cmd = motion_class(motion)(cfg, env)
    cmd.bin_failed_count[-1] = 1.e6
    torch.manual_seed(42)
    cmd._adaptive_sampling(torch.arange(env.num_envs))
    clips = motion.frame_clip_ids[cmd.time_steps]
    assert abs(float(clips.float().mean())-.5) < .02
    for clip in (0, 1):
        local = cmd.time_steps[clips == clip]-motion.clip_starts[clip]
        assert abs(float((local == 0).float().mean())-.2) < .03
        assert (local < motion.clip_lengths[clip]).all()
    # Environments at endpoints hold there; newly reset ones execute frame zero.
    cmd.time_steps = torch.tensor([99, 1099, 100, 0])
    env.episode_length_buf = torch.tensor([10, 10, 0, 0])
    cmd._update_command()
    assert cmd.time_steps.tolist() == [99, 1099, 100, 0]


def test_bundled_motions_have_expected_identity_and_order():
    motion = library.MotionLibrary(ROOT/'retargeted_motion/training_list')
    assert motion.clip_names == [
        '01_walk', '02_tilted_sway', '03_standing', '04_narrow_sway',
        '05_slide', '06_sole_rotation', '07_toe_rotation', '08_turn_right', '09_front_kick',
    ]
    assert motion.clip_lengths.tolist() == [261, 500, 500, 500, 500, 500, 500, 661, 115]
    assert motion.time_step_total == 4037
    with np.load(motion.clip_paths[1]) as sway:
        assert sway['tilted_side'].item() == 'left'
        assert sway['tilt_deg'].item() == 15.
