"""Standing command regression checks without launching Isaac Sim."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('frames', [1, 4])
def test_standing_freezes_pose_and_never_resets_at_clip_boundary(monkeypatch, frames):
    class MotionCommand:
        def __init__(self, cfg, env):
            self.motion = env.motion
            self.time_steps = torch.zeros(2, dtype=torch.long)
            self.resets = 0

        def reset_command(self, env_ids):
            self.resets += 1
            self.time_steps[env_ids] = 0

        def _update_command(self):
            self.time_steps += 1
            ids = torch.where(self.time_steps >= frames)[0]
            self._resample_command(ids)

    dependency = ModuleType('whole_body_tracking.tasks.tracking.mdp.commands')
    dependency.MotionCommand = MotionCommand
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    path = Path(__file__).resolve().parents[1] / (
        'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/motion.py')
    spec = importlib.util.spec_from_file_location('standing_motion_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    poses = ('joint_pos', '_body_pos_w', '_body_quat_w')
    velocities = ('joint_vel', '_body_lin_vel_w', '_body_ang_vel_w')
    motion = SimpleNamespace(**{name: torch.arange(frames * 3).reshape(frames, 3).float() + 1
                               for name in poses + velocities})
    first = motion.joint_pos[0].clone()
    command = module.WBCACCStandingMotion(None, SimpleNamespace(motion=motion))
    for _ in range(20):
        command._update_command()
        assert command.time_steps.eq(0).all()
    assert command.resets == 0
    for name in poses:
        torch.testing.assert_close(getattr(motion, name), first.expand(frames, -1))
    for name in velocities:
        assert getattr(motion, name).eq(0).all()
    command._resample_command(torch.tensor([1]))
    assert command.resets == 1
