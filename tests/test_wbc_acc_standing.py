"""Standing command regression checks without launching Isaac Sim."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import numpy as np


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
    # Reference preparation is tested independently; this fixture isolates
    # standing's hold/clip-boundary behavior from simulator initialization.
    monkeypatch.setattr(module.WBCACCTrainingMotion, '__init__', MotionCommand.__init__)
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


@pytest.mark.parametrize('fps', [np.array(50.), np.array([50.])])
def test_standing_initialization_accepts_both_npz_fps_formats(monkeypatch, tmp_path, fps):
    """Exercise reference preparation through the real standing constructor."""
    archive = tmp_path / 'reference.npz'
    np.savez(archive, fps=fps)
    with np.load(archive) as data:
        loaded_fps = data['fps']

    class MotionCommand:
        def __init__(self, cfg, env):
            self.motion, self.robot = env.motion, env.robot

    dependency = ModuleType('whole_body_tracking.tasks.tracking.mdp.commands')
    dependency.MotionCommand = MotionCommand
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    preparation = ModuleType('whole_body_tracking.utils.wbc_reference')
    observed_fps = []

    def prepare(joints, root, quat, joint_names, body_names, limits, rate, ground):
        observed_fps.append(rate)
        return {}, {'fps': rate}

    preparation.prepare_wbc_reference = prepare
    monkeypatch.setitem(sys.modules, preparation.__name__, preparation)
    path = Path(__file__).resolve().parents[1] / (
        'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/motion.py')
    spec = importlib.util.spec_from_file_location('standing_fps_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    motion = SimpleNamespace(fps=loaded_fps,
        joint_pos=torch.ones(2, 1), joint_vel=torch.ones(2, 1),
        _body_pos_w=torch.zeros(2, 1, 3), _body_quat_w=torch.zeros(2, 1, 4),
        _body_lin_vel_w=torch.ones(2, 1, 3), _body_ang_vel_w=torch.ones(2, 1, 3))
    robot = SimpleNamespace(body_names=['Trunk'], joint_names=['joint'],
                            data=SimpleNamespace(soft_joint_pos_limits=torch.tensor([[[-3., 3.]]])))
    action = SimpleNamespace(controller=SimpleNamespace(ground_height=0.))
    env = SimpleNamespace(motion=motion, robot=robot,
                          cfg=SimpleNamespace(actions=SimpleNamespace(wbc_authority=action)))
    command = module.WBCACCStandingMotion(None, env)
    assert observed_fps == [50.]
    assert command.motion.joint_vel.count_nonzero() == 0


@pytest.mark.parametrize('frames', [1, 4])
def test_single_pass_never_resets_robot_at_clip_boundary(monkeypatch, frames):
    class MotionCommand:
        def __init__(self, cfg, env):
            self.motion = SimpleNamespace(time_step_total=frames)
            self.time_steps = torch.zeros(1, dtype=torch.long)

        def _update_command(self):
            self.time_steps += 1
            ids = torch.where(self.time_steps >= frames)[0]
            self._resample_command(ids)

        def reset_command(self, env_ids):
            pytest.fail('Single-pass playback reset robot state at the clip boundary')

    dependency = ModuleType('whole_body_tracking.tasks.tracking.mdp.commands')
    dependency.MotionCommand = MotionCommand
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    path = Path(__file__).resolve().parents[1] / (
        'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/motion.py')
    spec = importlib.util.spec_from_file_location('single_pass_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.WBCACCTrainingMotion, '__init__', MotionCommand.__init__)
    command = module.WBCACCSinglePassMotion(None, None)
    observed = []
    for _ in range(frames+3):
        observed.append(int(command.time_steps[0]))
        command._update_command()
    assert observed == list(range(frames))+[frames-1]*3
