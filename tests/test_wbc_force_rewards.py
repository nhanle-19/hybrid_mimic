"""FORCE percentage rewards, reference contact geometry, and reset boundaries."""
import ast
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
    for name in ('wbc_contact_mapping', 'wbc_acc_contact_policy', 'wbc_acc_torch_model', 'wbc_force_learning'):
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
    reference_quat = torch.zeros(frames, 2, 4)
    reference_quat[..., 0] = 1.
    command = NS(time_steps=torch.zeros(count, dtype=torch.long),
                 motion=NS(time_step_total=frames, _body_pos_w=torch.zeros(frames, 2, 3),
                           _body_quat_w=reference_quat),
                 robot=NS(body_names=['right_foot_link', 'left_foot_link']))
    action = NS(cfg=NS(controller=NS(ground_height=0., contact_force_max=(600., 600.))),
                previous_wrench=None, qp_failed=torch.zeros(count, dtype=torch.bool))
    return NS(num_envs=count, device='cpu', step_dt=dt, extras={},
        scene=NS(),
        command_manager=NS(get_term=lambda _: command),
        action_manager=NS(action=torch.zeros(count, 2),
                          get_term=lambda _: action),
        termination_manager=NS(terminated=torch.zeros(count, dtype=torch.bool), active_terms=[]))


def point_env(frames=10, count=2, dt=.02):
    env = make_env(frames, count, dt)
    env.action_manager.action = torch.zeros(count, 8)
    action = env.action_manager.get_term('wbc_force')
    action.normal_force_max = torch.full((8,), 100.)
    action.previous_point_forces = None
    return env


def test_corner_rewards_use_each_reference_point_without_gating_actions(rewards):
    env = point_env(frames=3)
    motion = env.command_manager.get_term('motion').motion
    points = rewards.sole_vertices(dtype=torch.float32)
    motion._body_pos_w[..., 2] = -points[0, 0, 2]
    # Body index 1 is the left foot. Pitch it and place its low edge at ground.
    angle = torch.tensor(.3)
    motion._body_quat_w[:, 1] = torch.tensor([torch.cos(angle/2), 0., torch.sin(angle/2), 0.])
    rotation = rewards.rotation_xyzw(motion._body_quat_w[:, 1, [1, 2, 3, 0]])
    rotated = points[0]@rotation.transpose(-1, -2)
    motion._body_pos_w[:, 1, 2] = -rotated[..., 2].amin(-1)
    heights = rewards._reference_clearance(env)[0]
    assert heights.shape == (3, 8)
    assert (heights[:, :2] > .005).all() and (heights[:, 2:] < .001).all()
    env.action_manager.action.fill_(40.)
    swing = rewards.SwingForcePenalty(NS(params={}, weight=-.5), env)
    torch.testing.assert_close(swing(env)/swing.scale, torch.full((2,), .25))
    assert (env.action_manager.action == 40.).all()  # Geometry is a reward weight only.
    env.action_manager.action.fill_(-3.)
    zero = rewards.FalseZeroPenalty(NS(params={}, weight=-.4), env)
    torch.testing.assert_close(zero(env)/zero.scale, torch.full((2,), .75))


def test_corner_gap_uses_individual_point_forces_and_logs_foot_totals(rewards):
    env = point_env(count=1)
    env.action_manager.action.fill_(40.)
    action = env.action_manager.get_term('wbc_force')
    action.previous_point_forces = torch.zeros(1, 2, 4, 3)
    action.previous_point_forces[..., 2] = torch.tensor([[[100., 100., 0., 0.], [100., 100., 100., 100.]]])
    term = rewards.ForceCapGapPenalty(NS(params={}, weight=-.08), env)
    torch.testing.assert_close(term(env)/term.scale, torch.tensor([.25]))
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Caps/left_unused_force_N'] == 200.
    assert logs['WBCForce/Caps/right_unused_force_N'] == 0.
    action.qp_failed[:] = True
    torch.testing.assert_close(term(env)/term.scale, torch.ones(1))


def test_corner_completion_logs_all_eight_caps_and_whole_foot_release(rewards):
    env = point_env()
    env.action_manager.action[:] = torch.tensor([-3., -3., -3., 40., -3., -3., -3., -3.])
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    term(env)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Caps/left_mean'] == .25
    assert logs['WBCForce/Caps/right_mean'] == 0.
    assert logs['WBCForce/Caps/left_stance_zero_percent'] == 0.
    assert logs['WBCForce/Caps/right_stance_zero_percent'] == 100.
    assert logs['WBCForce/Caps/left_corner_3_mean'] == 1.
    assert logs['WBCForce/Caps/left_corner_0_mean'] == 0.


def test_corner_rewards_and_logs_use_point_nine_cap_mapping(rewards):
    env = point_env(count=1)
    term = rewards.ForceCapGapPenalty(NS(params={}, weight=-.08), env)
    # Raw zero is 4/9 capacity; no QP force has used any of it yet.
    torch.testing.assert_close(term(env)/term.scale, torch.tensor([(4/9)**2]))
    completion = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    completion(env)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Caps/left_mean'] == pytest.approx(4/9)
    assert logs['WBCForce/Caps/right_corner_3_mean'] == pytest.approx(4/9)


def test_active_reward_configuration_includes_stronger_tracking_and_cap_change_costs(rewards):
    tree = ast.parse((SOURCE/'tasks/tracking/config/t1_wbc_force/flat_env_cfg.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WBCForceRewardsCfg')
    namespace = dict(vars(rewards), configclass=lambda cls: cls, RewardTermCfg=NS)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<force_rewards_cfg>', 'exec'), namespace)
    config = namespace['WBCForceRewardsCfg']()
    assert not hasattr(config, 'contact_orientation')
    assert config.false_zero.func is rewards.FalseZeroPenalty and config.false_zero.weight == -.4
    assert config.force_cap_gap.func is rewards.ForceCapGapPenalty and config.force_cap_gap.weight == -.08
    assert config.joint_tracking.func is rewards.JointTrackingPenalty
    assert config.joint_tracking.weight == -.5 and config.joint_tracking.params == {'error_scale': .2}
    assert config.motion_completion.weight == 1.
    assert config.swing_force.weight == -.5 and config.action_change.weight == -.5


@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_joint_tracking_clip_cost_and_logging_are_independent_of_length_and_timestep(rewards, frames, dt):
    env = make_env(frames, dt=dt)
    command = env.command_manager.get_term('motion')
    command.joint_pos = torch.full((2, 23), .5)
    command.robot_joint_pos = command.joint_pos.clone()
    term = rewards.JointTrackingPenalty(NS(params={'error_scale': .2}, weight=-.25), env)
    assert term(env).count_nonzero() == 0
    rewards._metrics(env).drain()
    command.robot_joint_pos[1] += .1
    points = term(env)*dt*term.cfg.weight
    # Constant 0.1-radian RMS error costs 25 * (1 - exp(-0.25)) per full clip.
    torch.testing.assert_close(points*frames, torch.tensor([0., -5.5299804]))
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Tracking/joint_position_rmse_rad'] == pytest.approx(.05)
    assert logs['WBCForce/Reward/joint_tracking_points_per_step'] == pytest.approx(float(points.mean()))


def test_joint_tracking_uses_current_reference_mean_error_and_remains_bounded_on_failure(rewards):
    env = make_env()
    command = env.command_manager.get_term('motion')
    command.joint_pos = torch.tensor([[.2, .6], [.2, .6]])
    command.robot_joint_pos = torch.tensor([[0., .6], [.4, .6]])
    term = rewards.JointTrackingPenalty(NS(params={}, weight=-.25), env)
    expected = torch.full((2,), 1.-torch.exp(torch.tensor(-.5)))
    torch.testing.assert_close(term(env)/term.scale, expected)
    command.joint_pos = command.joint_pos.repeat(1, 3)
    command.robot_joint_pos = command.robot_joint_pos.repeat(1, 3)
    torch.testing.assert_close(term(env)/term.scale, expected)  # Mean, not sum over joints.
    command.joint_pos.copy_(command.robot_joint_pos)
    assert term(env).count_nonzero() == 0  # Reference updates are read on every call.
    command.robot_joint_pos += 100.
    env.termination_manager.terminated[:] = True
    torch.testing.assert_close(term(env)/term.scale, torch.ones(2))


def test_joint_tracking_scales_by_each_current_clip(rewards):
    env = make_env(frames=15)
    command = env.command_manager.get_term('motion')
    command.motion.clip_bounds = lambda frame: (torch.where(frame < 5, 0, 5), torch.where(frame < 5, 5, 10))
    command.time_steps[:] = torch.tensor([0, 5])
    command.joint_pos, command.robot_joint_pos = torch.zeros(2, 23), torch.full((2, 23), .2)
    term = rewards.JointTrackingPenalty(NS(params={}, weight=-.25), env)
    expected = (1.-torch.exp(torch.tensor(-1.)))*torch.tensor([20., 10.])
    torch.testing.assert_close(term(env)*env.step_dt, expected)


@pytest.mark.parametrize('scale', [0., -.1, float('nan'), float('inf')])
def test_joint_tracking_rejects_invalid_error_scale(rewards, scale):
    with pytest.raises(ValueError, match='error_scale'):
        rewards.JointTrackingPenalty(NS(params={'error_scale': scale}, weight=-.25), make_env())


@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_full_clip_earns_exactly_100_and_holding_end_adds_nothing(rewards, frames, dt):
    env = make_env(frames, dt=dt)
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    total = torch.zeros(env.num_envs)
    for frame in range(frames):
        term.command.time_steps.fill_(frame)
        total += term(env)*dt
    torch.testing.assert_close(total, torch.full_like(total, 100.), atol=3e-4, rtol=0)
    assert term(env).count_nonzero() == 0
    term.command.time_steps.fill_(0)  # Hidden reference wrap earns no extra progress.
    assert term(env).count_nonzero() == 0
    term.reset()
    logs = term.metrics.drain()
    assert logs['WBCForce/FullMotion/completed_percent'] == 100.
    assert logs['WBCForce/FullMotion/success_percent'] == 100.
    assert logs['WBCForce/Sampling/start_count'] == 2
    assert logs['WBCForce/Sampling/start_frame_zero_percent'] == 100.


def test_failure_and_random_start_receive_only_executed_successful_frames(rewards):
    env = make_env(10)
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    total = torch.zeros(2)
    for frame in (7, 8, 9):
        term.command.time_steps.fill_(frame)
        env.termination_manager.terminated[1] = frame == 9
        total += term(env)*env.step_dt
    torch.testing.assert_close(total, torch.tensor([30., 20.]))
    # Final-frame completion can coincide with failure; failure gets no alive credit.
    assert rewards.reference_motion_complete(env).tolist() == [True, True]
    term.reset(torch.tensor([0]))
    logs = term.metrics.drain()
    assert logs['WBCForce/Episode/executed_clip_percent'] == 30.
    assert logs['WBCForce/Episode/suffix_completed_percent'] == 100.
    assert logs['WBCForce/FullMotion/episode_count'] == 0.
    assert 'WBCForce/FullMotion/success_percent' not in logs
    assert term.completed_frames.tolist() == [0, 2]
    term.command.time_steps[:] = torch.tensor([0, 9])
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([10., 0.]))


def test_action_change_uses_decoded_caps_and_exempts_each_reset(rewards):
    env = make_env(100)
    term = rewards.ForceActionChangePenalty(NS(params={}, weight=-.05), env)
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
    env.action_manager.get_term('wbc_force').cfg.controller.ground_height = floor
    return env


@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_false_zero_is_clip_scaled_and_uses_decoded_caps(rewards, frames, dt):
    env = make_env(frames, count=3, dt=dt)
    command = env.command_manager.get_term('motion')
    # Both caps zero, left cap zero, then raw zeros (decoded caps are 0.5).
    env.action_manager.action[:] = torch.tensor([[-3., -30.], [-3., 3.], [0., 0.]])
    env.termination_manager.terminated[0] = True  # A failing step still incurs cost.
    term = rewards.FalseZeroPenalty(NS(params={}, weight=-.5), env)
    assert term.contact_weight.min() == 1.
    total = torch.zeros(3, dtype=torch.float64)
    for frame in range(frames):
        command.time_steps.fill_(frame)
        total += term(env)*dt*term.cfg.weight
    torch.testing.assert_close(total, torch.tensor([-50., -25., 0.], dtype=torch.float64), atol=1.e-5, rtol=0)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Reward/false_zero_points_per_step'] == pytest.approx(-25./frames)


def test_false_zero_uses_reference_contact_and_excludes_swing(rewards):
    env = geometry_env(rewards)
    command = env.command_manager.get_term('motion')
    command.motion._body_pos_w[2, 1, 2] += .001  # Clearly above the 5 mm threshold.
    # Actual positions, orientations, and measured loads cannot erase this cost.
    env.action_manager.action[:] = torch.tensor([-3., 3.])
    term = rewards.FalseZeroPenalty(NS(params={}, weight=-.5), env)
    torch.testing.assert_close(term.contact_weight[:, 0], torch.tensor([1., .5, 0.], dtype=torch.float64),
                               atol=1.e-6, rtol=0)
    command.time_steps[:] = torch.tensor([0, 1])
    weighted = term(env)*env.step_dt*term.cfg.weight
    torch.testing.assert_close(weighted, torch.tensor([-25./3, -12.5/3], dtype=torch.float64), atol=1.e-6, rtol=0)
    env.action_manager.action[:, 0] = -2.19  # Small positive decoded caps are not false zeros.
    assert (rewards.decode_force_values(env.action_manager.action)[:, 0] > 0).all()
    assert term(env).count_nonzero() == 0
    env.action_manager.action[:, 0] = -3.
    command.time_steps.fill_(2)
    assert term(env).count_nonzero() == 0  # Reference swing is excluded.


@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_force_cap_gap_is_clip_scaled_and_normalized_per_foot(rewards, frames, dt):
    env = make_env(frames, count=3, dt=dt)
    env.action_manager.action.fill_(3.)
    action = env.action_manager.get_term('wbc_force')
    action.cfg.controller.contact_force_max = (200., 800.)
    action.previous_wrench = torch.full((3, 2, 6), 999., dtype=torch.float64)
    action.previous_wrench[..., 2] = torch.tensor([[100., 400.], [200., 400.], [200., 800.]])
    term = rewards.ForceCapGapPenalty(NS(params={}, weight=-.1), env)
    total = torch.zeros(3, dtype=torch.float64)
    for frame in range(frames):
        term.command.time_steps.fill_(frame)
        total += term(env)*dt*term.cfg.weight
    torch.testing.assert_close(total, torch.tensor([-2.5, -1.25, 0.], dtype=torch.float64), atol=1.e-6, rtol=0)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Reward/force_cap_gap_points_per_step'] == pytest.approx(-1.25/frames)
    assert logs['WBCForce/Caps/left_unused_force_N'] == pytest.approx(100./3)
    assert logs['WBCForce/Caps/right_unused_force_N'] == pytest.approx(800./3)
    # Rewards read the latest solve, not a wrench captured during initialization.
    action.previous_wrench = torch.zeros_like(action.previous_wrench)
    torch.testing.assert_close(term(env)*dt, torch.full((3,), 100./frames))


def test_force_cap_gap_handles_zero_caps_saturation_and_failed_solves(rewards):
    env = make_env(frames=100, count=6)
    env.action_manager.action.fill_(3.)
    env.action_manager.action[0] = 0.  # 300 N caps, not zero caps.
    env.action_manager.action[1] = -3.
    action = env.action_manager.get_term('wbc_force')
    term = rewards.ForceCapGapPenalty(NS(params={}, weight=-.1), env)
    # Before any solve, no capacity has been used.
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([.25, 0., 1., 1., 1., 1.]))
    action.previous_wrench = torch.zeros(6, 2, 6)
    action.previous_wrench[..., 2] = torch.tensor([
        [150., 300.], [0., 0.], [600.001, 600.001], [-.001, -.001],
        [float('nan'), float('inf')], [600., 600.]])
    action.qp_failed[5] = True  # Damping is applied instead of this candidate wrench.
    value = term(env)*env.step_dt
    torch.testing.assert_close(value, torch.tensor([.03125, 0., 0., 1., 1., 1.]))
    assert torch.isfinite(value).all()
    assert all(torch.isfinite(torch.tensor(v)) for v in rewards._metrics(env).drain().values())


@pytest.mark.parametrize('maximum', [(0., 600.), (-1., 600.), (float('nan'), 600.),
                                    (600., float('inf')), (600.,), 600.])
def test_force_cap_gap_rejects_invalid_maxima(rewards, maximum):
    env = make_env()
    env.action_manager.get_term('wbc_force').cfg.controller.contact_force_max = maximum
    with pytest.raises(ValueError, match='finite positive'):
        rewards.ForceCapGapPenalty(NS(params={}, weight=-.1), env)


def test_swing_penalty_ramps_with_reference_clearance_and_ignores_actual_support(rewards):
    env = geometry_env(rewards)
    term = rewards.SwingForcePenalty(NS(params={'unload_lead_time': 0.}, weight=-.5), env)
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
    term = rewards.SwingForcePenalty(NS(params={}, weight=-.5), env)
    assert term.swing_weight.count_nonzero() == 0


@pytest.mark.parametrize('side', [0, 1], ids=['left', 'right'])
@pytest.mark.parametrize('roll,pitch,grounded_corners', [
    (0., .35, 2), (0., -.35, 2), (.30, 0., 2), (-.30, 0., 2), (.30, .35, 1),
], ids=['toe', 'heel', 'edge_positive_roll', 'edge_negative_roll', 'single_corner'])
def test_partial_contact_uses_minimum_height_and_only_penalizes_full_liftoff(
        rewards, side, roll, pitch, grounded_corners):
    env = geometry_env(rewards)
    motion = env.command_manager.get_term('motion').motion
    points = rewards.sole_vertices(dtype=torch.float64)
    # Both feet start planted; geometry_env deliberately reverses body ordering.
    for foot, body in enumerate((1, 0)):
        motion._body_pos_w[:, body, 2] = .2-points[foot, :, 2].min()
    roll, pitch = torch.tensor([roll, pitch], dtype=torch.float64)/2
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    body = (1, 0)[side]
    motion._body_quat_w[:, body] = torch.stack((cr*cp, sr*cp, cr*sp, -sr*sp))
    rotation = rewards.rotation_xyzw(motion._body_quat_w[:, body][:, [1, 2, 3, 0]])
    vertices = points[side] @ rotation.transpose(-1, -2)
    clearance = torch.tensor([0., .0005, .006], dtype=torch.float64)
    motion._body_pos_w[:, body, 2] = .2-vertices[..., 2].amin(-1)+clearance
    world_heights = vertices[..., 2]+motion._body_pos_w[:, body, None, 2]-.2
    # An average would wrongly apply the full swing penalty to these contacts.
    assert (world_heights.mean(-1) > .005).all()
    assert int((world_heights[0].abs() < 1e-10).sum()) == grounded_corners
    actual_clearance = rewards._reference_clearance(env)[0]
    torch.testing.assert_close(actual_clearance[:, side], clearance, atol=1e-10, rtol=0)
    term = rewards.SwingForcePenalty(NS(params={'unload_lead_time': 0.}, weight=-.5), env)
    torch.testing.assert_close(term.swing_weight[:, side], torch.tensor([0., 0., 1.], dtype=torch.float64))
    assert term.swing_weight[:, 1-side].count_nonzero() == 0
    env.action_manager.action.fill_(3.)  # Positive capacity on both feet.
    term.command.time_steps[:] = torch.tensor([0, 1])
    assert term(env).count_nonzero() == 0
    term.command.time_steps.fill_(2)
    assert (term(env) > 0).all()

    # A grounded toe, heel, edge, or corner still makes a zero cap a false zero.
    command = env.command_manager.get_term('motion')
    false_zero = rewards.FalseZeroPenalty(NS(params={}, weight=-.5), env)
    env.action_manager.action[:, side] = -3.
    command.time_steps[:] = torch.tensor([0, 1])
    torch.testing.assert_close(false_zero(env)*env.step_dt,
                               torch.full((2,), 50./3, dtype=torch.float64))
    command.time_steps.fill_(2)
    assert false_zero(env).count_nonzero() == 0


@pytest.mark.parametrize('params', [{'clearance_start': .005, 'clearance_full': .001},
                                  {'clearance_start': -.001}])
@pytest.mark.parametrize('name', ['SwingForcePenalty', 'FalseZeroPenalty'])
def test_contact_penalties_reject_invalid_thresholds(rewards, params, name):
    with pytest.raises(ValueError, match='thresholds'):
        getattr(rewards, name)(NS(params=params, weight=-.5), geometry_env(rewards))


def test_episode_metrics_weight_episodes_not_reset_batches_and_ignore_initial_reset(rewards):
    env = make_env(frames=10, count=4)
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    term.reset()
    initial = term.metrics.drain()
    assert initial['WBCForce/Episode/count'] == 0
    assert 'WBCForce/Episode/suffix_success_percent' not in initial
    term.command.time_steps[:] = torch.tensor([9, 9, 9, 0])
    term(env)
    term.reset(torch.tensor([0, 1, 2]))
    # Draining rollout statistics must not reset the still-running episode.
    first = term.metrics.drain()
    assert first['WBCForce/Episode/count'] == 3
    for frame in range(1, 10):
        term.command.time_steps.fill_(frame)
        term(env)
    env.termination_manager.terminated[3] = True
    term.reset(torch.tensor([3]))
    last = term.metrics.drain()
    assert last['WBCForce/FullMotion/episode_count'] == 1
    assert last['WBCForce/FullMotion/success_percent'] == 0  # Failure wins over last frame.
    assert last['WBCForce/FullMotion/completed_percent'] == 100
    assert last['WBCForce/Episode/failure_percent'] == 100


def test_conditional_contact_metrics_and_reset_action_changes(rewards):
    env = geometry_env(rewards)
    command = env.command_manager.get_term('motion')
    command.time_steps.fill_(2)
    command.motion._body_pos_w[2, 1, 2] += .01
    env.action_manager.action[:] = torch.tensor([[-3., 3.], [3., 3.]])
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    term(env)
    logs = term.metrics.drain()
    assert logs['WBCForce/Contact/left_swing_zero_cap_percent'] == 50
    assert logs['WBCForce/Contact/left_swing_cap_mean'] == .5
    assert 'WBCForce/Contact/right_swing_zero_cap_percent' not in logs
    assert 'WBCForce/Caps/left_absolute_change' not in logs


def test_zero_hold_duration_survives_rollout_drains_and_flushes_on_reset(rewards):
    env = geometry_env(rewards)
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    term.command.motion._body_pos_w[2, 1, 2] += .01
    term.command.time_steps.fill_(2)
    env.action_manager.action[:] = torch.tensor([[-3., 3.], [-3., 3.]])
    term(env)
    term.metrics.drain()
    term(env)
    term(env)
    env.action_manager.action[0, 0] = 3.
    term(env)  # First hold ends at 60 ms; second continues until reset at 80 ms.
    term.reset(torch.tensor([1]))
    logs = term.metrics.drain()
    assert logs['WBCForce/Contact/left_zero_hold_ms'] == 70.
    assert term.zero_swing_steps.count_nonzero() == 0


def test_mixed_clips_normalize_independently_and_report_their_own_endpoints(rewards):
    env = make_env(frames=15)
    motion = env.command_manager.get_term('motion').motion
    motion.frame_clip_ids = torch.tensor([0]*5+[1]*10)
    motion.clip_names, motion.num_clips = ['walk', 'sway'], 2
    motion.clip_bounds = lambda frame: (torch.where(frame < 5, 0, 5), torch.where(frame < 5, 5, 10))
    term = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    total = torch.zeros(2)
    for step in range(10):
        term.command.time_steps[:] = torch.tensor([min(step, 4), 5+step])
        total += term(env)*env.step_dt
        if step == 4:
            assert rewards.reference_motion_complete(env).tolist() == [True, False]
    torch.testing.assert_close(total, torch.tensor([100., 100.]))
    term.reset()
    logs = term.metrics.drain()
    assert logs['WBCForce/Motion/walk/full_success_percent'] == 100.
    assert logs['WBCForce/Motion/sway/full_success_percent'] == 100.
    assert logs['WBCForce/Sampling/start_frame_zero_percent'] == 100.


@pytest.mark.parametrize('name,raw', [('FalseZeroPenalty', -3.), ('ForceCapGapPenalty', 3.)])
def test_new_penalties_scale_by_current_clip_in_mixed_batch(rewards, name, raw):
    env = make_env(frames=15)
    command = env.command_manager.get_term('motion')
    command.motion.clip_bounds = lambda frame: (torch.where(frame < 5, 0, 5), torch.where(frame < 5, 5, 10))
    command.time_steps[:] = torch.tensor([0, 5])
    env.action_manager.action.fill_(raw)
    term = getattr(rewards, name)(NS(params={}, weight=-.5), env)
    torch.testing.assert_close(term(env)*env.step_dt, torch.tensor([20., 10.], dtype=term(env).dtype))


def test_unloading_lead_does_not_cross_between_clips(rewards):
    env = geometry_env(rewards)
    motion = env.command_manager.get_term('motion').motion
    motion.clip_slices = [(0, 2), (2, 3)]
    motion._body_pos_w[:2, 1, 2] -= torch.tensor([.001, .003], dtype=torch.float64)
    motion._body_pos_w[2, 1, 2] += .01
    term = rewards.SwingForcePenalty(NS(params={'unload_lead_time': .04}, weight=-.5), env)
    assert term.unloading_weight[:2].count_nonzero() == 0
    assert term.unloading_weight[2, 0] == 1.
