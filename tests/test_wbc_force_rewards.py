"""FORCE percentage rewards, reference swing geometry, and reset boundaries."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/whole_body_tracking/whole_body_tracking'


@pytest.fixture
def rewards(monkeypatch):
    managers = ModuleType('isaaclab.managers')

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    managers.ManagerTermBase = ManagerTermBase
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', managers)
    for name in ('wbc_contact_mapping', 'wbc_acc_contact_policy', 'wbc_acc_torch_model'):
        module_name = 'whole_body_tracking.utils.'+name
        spec = importlib.util.spec_from_file_location(module_name, SOURCE/'utils'/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location('force_reward_terms_test',
        SOURCE/'tasks/tracking/config/t1_wbc_force/reward_terms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(frames=10, count=2, dt=.02):
    command = NS(time_steps=torch.zeros(count, dtype=torch.long),
                 motion=NS(time_step_total=frames))
    return NS(num_envs=count, device='cpu', step_dt=dt, extras={},
        command_manager=NS(get_term=lambda _: command),
        action_manager=NS(action=torch.zeros(count, 2)),
        termination_manager=NS(terminated=torch.zeros(count, dtype=torch.bool)))


@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_full_clip_earns_exactly_100_and_holding_end_adds_nothing(rewards, frames, dt):
    env = make_env(frames, dt=dt)
    term = rewards.MotionCompletionReward(NS(params={}), env)
    total = torch.zeros(env.num_envs)
    for frame in range(frames):
        term.command.time_steps.fill_(frame)
        total += term(env)*dt
    torch.testing.assert_close(total, torch.full_like(total, 100.), atol=3e-4, rtol=0)
    assert term(env).count_nonzero() == 0
    term.command.time_steps.fill_(0)  # Hidden reference wrap earns no extra progress.
    assert term(env).count_nonzero() == 0
    term.reset()
    assert env.extras['log']['WBCForce/completed_motion_percent'] == 100.


def test_failure_and_random_start_receive_only_executed_successful_frames(rewards):
    env = make_env(10)
    term = rewards.MotionCompletionReward(NS(params={}), env)
    total = torch.zeros(2)
    for frame in (7, 8, 9):
        term.command.time_steps.fill_(frame)
        env.termination_manager.terminated[1] = frame == 9
        total += term(env)*env.step_dt
    torch.testing.assert_close(total, torch.tensor([30., 20.]))
    # Final-frame completion can coincide with failure; failure gets no alive credit.
    assert rewards.reference_motion_complete(env).tolist() == [True, True]
    term.reset(torch.tensor([0]))
    assert env.extras['log']['WBCForce/completed_motion_percent'] == 30.
    assert term.completed_frames.tolist() == [0, 2]
    term.command.time_steps[:] = torch.tensor([0, 9])
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([10., 0.]))


def test_action_change_uses_decoded_caps_and_exempts_each_reset(rewards):
    env = make_env(100)
    term = rewards.ForceActionChangePenalty(NS(params={}), env)
    env.action_manager.action[:] = torch.tensor([[-3., 3.], [0., 0.]])
    assert term(env).count_nonzero() == 0  # First actions are not changes from a fictitious 0.5 cap.
    env.action_manager.action[:] = torch.tensor([[-30., 30.], [3., 0.]])
    # Saturated raw changes have no effect. One 0.5 cap change gives mean square .125.
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([0., .125]))
    term.reset(torch.tensor([1]))
    env.action_manager.action[:] = torch.tensor([[3., -3.], [-3., -3.]])
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([1., 0.]))
    assert term(env).count_nonzero() == 0


def geometry_env(rewards):
    env = make_env(3)
    command = env.command_manager.get_term('motion')
    # Reverse the body order to catch index assumptions.
    command.robot = NS(body_names=['right_foot_link', 'left_foot_link'])
    motion = command.motion
    motion._body_pos_w = torch.zeros(3, 2, 3, dtype=torch.float64)
    motion._body_quat_w = torch.zeros(3, 2, 4, dtype=torch.float64)
    motion._body_quat_w[..., 0] = 1.
    points = rewards.sole_vertices(dtype=torch.float64)
    floor = .2
    for side, body in enumerate((1, 0)):
        motion._body_pos_w[:, body, 2] = floor-points[side, :, 2].min()
    # Left-foot clearance is 1, 3 and 5 mm. Right foot stays planted.
    motion._body_pos_w[:, 1, 2] += torch.tensor([.001, .003, .005])
    env.action_manager.get_term = lambda _: NS(cfg=NS(controller=NS(ground_height=floor)))
    return env


def test_swing_penalty_ramps_with_reference_clearance_and_ignores_actual_support(rewards):
    env = geometry_env(rewards)
    term = rewards.SwingForcePenalty(NS(params={}), env)
    torch.testing.assert_close(term.swing_weight[:, 0], torch.tensor([0., .5, 1.], dtype=torch.float64),
                               atol=1e-6, rtol=0)
    assert term.swing_weight[:, 1].count_nonzero() == 0
    term.command.time_steps[:] = torch.tensor([1, 2])
    # Raw zero means a 0.5 capacity; only the left foot is penalized.
    expected = torch.tensor([.0625, .125], dtype=torch.float64)*100/3
    torch.testing.assert_close(term(env)*env.step_dt, expected, atol=1e-6, rtol=0)
    env.action_manager.action[:, 0] = -3.
    env.action_manager.action[:, 1] = 3.
    assert term(env).count_nonzero() == 0
    # No actual-state/contact input is supplied: actual gating cannot erase the penalty.


def test_grounded_toe_pivot_is_not_reference_swing(rewards):
    env = geometry_env(rewards)
    motion = env.command_manager.get_term('motion').motion
    angle = torch.tensor(.35, dtype=torch.float64)
    motion._body_quat_w[:, 1] = torch.tensor([torch.cos(angle/2), 0., torch.sin(angle/2), 0.])
    rotation = rewards.rotation_xyzw(motion._body_quat_w[:, 1][:, [1, 2, 3, 0]])
    vertices = rewards.sole_vertices(dtype=torch.float64)[0] @ rotation.transpose(-1, -2)
    motion._body_pos_w[:, 1, 2] = .2-vertices[..., 2].amin(-1)
    term = rewards.SwingForcePenalty(NS(params={}), env)
    assert term.swing_weight.count_nonzero() == 0


@pytest.mark.parametrize('params', [{'clearance_start': .005, 'clearance_full': .001},
                                  {'clearance_start': -.001}])
def test_swing_rejects_invalid_thresholds(rewards, params):
    with pytest.raises(ValueError, match='thresholds'):
        rewards.SwingForcePenalty(NS(params=params), geometry_env(rewards))
