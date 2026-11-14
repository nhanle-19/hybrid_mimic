"""Sole-corner and whole-body COM tracking for FORCE2, without Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_wbc_force_rewards import rewards as force_rewards


SOURCE = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking'


@pytest.fixture
def rewards(force_rewards):
    # Reuse only the simulator stubs; load FORCE2's independent reward module.
    spec = importlib.util.spec_from_file_location('force2_rewards_test',
        SOURCE/'tasks/tracking/config/t1_wbc_force2/reward_terms.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(frames=10, dt=.02, steps=(0, 0)):
    dtype = torch.float64
    count = len(steps)
    # Simulator ordering deliberately differs from the dynamics model ordering.
    names = ['right_foot_link', 'Trunk', 'left_foot_link']
    position = torch.tensor([[.3, -.1, .03], [.1, .2, .6], [.2, .1, .03]], dtype=dtype).repeat(frames, 1, 1)
    position[..., 0] += torch.arange(frames, dtype=dtype)[:, None]*.02
    quat = torch.zeros(frames, 3, 4, dtype=dtype)
    quat[..., 0] = 1.
    quat[:, 1, [0, 3]] = 2.**-.5  # Trunk yaw is 90 degrees.
    motion = NS(time_step_total=frames, _body_pos_w=position, _body_quat_w=quat,
                _body_lin_vel_w=torch.zeros_like(position), _body_ang_vel_w=torch.zeros_like(position))
    origins = torch.arange(count, dtype=dtype)[:, None]*torch.tensor([[10., -20., 3.]], dtype=dtype)
    current = torch.tensor(steps)
    link_pos = position[current]+origins[:, None]
    # Known rotated local COM offsets in simulator ordering.
    offsets = torch.tensor([[0., .03, 0.], [-.2, .1, 0.], [.05, 0., 0.]], dtype=dtype)
    data = NS(body_link_pos_w=link_pos.clone(), body_link_quat_w=quat[current].clone(),
              body_com_pos_w=link_pos+offsets, body_com_lin_vel_w=torch.zeros_like(link_pos),
              body_link_ang_vel_w=torch.zeros_like(link_pos))
    robot = NS(body_names=names, data=data)
    command = NS(time_steps=current, motion=motion, robot=robot)
    model = NS(body_names=['Trunk', 'left_foot_link', 'right_foot_link'],
               mass=torch.tensor([4., 1., 2.], dtype=dtype),
               com_local=torch.tensor([[.1, .2, 0.], [.05, 0., 0.], [0., .03, 0.]], dtype=dtype),
               inertia=torch.diag_embed(torch.tensor([[.1, .2, .3], [.02, .03, .04], [.03, .04, .05]], dtype=dtype)))
    action = NS(dynamics=model)
    return NS(num_envs=count, device='cpu', step_dt=dt,
              command_manager=NS(get_term=lambda _: command),
              action_manager=NS(get_term=lambda _: action, action=torch.zeros(count, 8)),
              scene=NS(env_origins=origins), termination_manager=NS(terminated=torch.zeros(count, dtype=torch.bool)))


def term_cfg(scale=.05):
    return NS(params={'error_scale': scale}, weight=-.25)


@pytest.mark.parametrize('name', ['SolePointTrackingPenalty', 'COMTrackingPenalty'])
def test_exact_tracking_at_current_frame_and_environment_origin_invariance(rewards, name):
    env = make_env(steps=(2, 7))
    term = getattr(rewards, name)(term_cfg(), env)
    torch.testing.assert_close(term(env), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    shift = torch.tensor([[3., 7., -2.], [-4., 1., .3]], dtype=torch.float64)
    data = term.command.robot.data
    env.scene.env_origins += shift
    data.body_link_pos_w += shift[:, None]
    data.body_com_pos_w += shift[:, None]
    data.body_link_quat_w *= -1  # Quaternion sign cannot alter geometry.
    torch.testing.assert_close(term(env), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    term.command.time_steps.fill_(0)
    assert (term(env) > 0).all()  # Target follows the executed reference frame.


def test_sole_tracking_sums_all_eight_points_without_cap_or_contact_gating(rewards):
    env = make_env()
    term = rewards.SolePointTrackingPenalty(term_cfg(), env)
    data = term.command.robot.data
    data.body_link_pos_w[0, 2, 0] += .01  # Left only: four corner errors.
    data.body_link_pos_w[1, [0, 2], 0] += .01  # Both feet: eight corner errors.
    summed = torch.tensor([4., 8.], dtype=torch.float64)*.01**2
    expected = -torch.expm1(-summed/.05**2)
    torch.testing.assert_close(term(env)/term.scale, expected)
    env.action_manager.action.fill_(-100.)
    torch.testing.assert_close(term(env)/term.scale, expected)
    env.action_manager.action.fill_(100.)
    torch.testing.assert_close(term(env)/term.scale, expected)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Tracking/sole_point_position_sum_squared_m2'] == pytest.approx(float(summed.mean()))
    assert logs['WBCForce/Tracking/sole_point_position_rmse_m'] == pytest.approx(float((summed/8).sqrt().mean()))


def test_sole_rotation_is_penalized_even_with_unchanged_link_origin(rewards):
    env = make_env(steps=(0,))
    term = rewards.SolePointTrackingPenalty(term_cfg(scale=.5), env)
    term.command.robot.data.body_link_quat_w[:, 2] = torch.tensor([0., 0., 0., 1.])  # Left yaw pi.
    points = rewards.sole_vertices(dtype=torch.float64)[0]
    summed = 4*points[:, :2].square().sum()
    torch.testing.assert_close(term(env)/term.scale, -torch.expm1(-summed/.5**2)[None])


def test_com_uses_all_link_masses_local_offsets_and_current_simulator_state(rewards):
    env = make_env()
    term = rewards.COMTrackingPenalty(term_cfg(), env)
    # Initial COM has rotated Trunk offset (-.2, .1, 0), not the link origin.
    reference = (2*torch.tensor([.3, -.07, .03])+4*torch.tensor([-.1, .3, .6])+
                 torch.tensor([.25, .1, .03]))/7
    torch.testing.assert_close(term.reference_com[0], reference.double(), atol=2e-8, rtol=0)
    data = term.command.robot.data
    data.body_com_pos_w[0, 1, 0] += .07  # 4/7 mass: whole-body COM moves .04 m.
    data.body_com_pos_w[1, 2, 0] += .07  # 1/7 mass: whole-body COM moves .01 m.
    error = torch.tensor([.04, .01], dtype=torch.float64)
    torch.testing.assert_close(term(env)/term.scale, -torch.expm1(-error.square()/.05**2))
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Tracking/com_position_error_m'] == pytest.approx(.025)


def test_com_penalizes_linear_and_angular_velocity_with_zero_position_error(rewards):
    env = make_env()
    term = rewards.COMTrackingPenalty(term_cfg(), env)
    data = term.command.robot.data
    data.body_com_lin_vel_w[0, :, 0] = .2  # Uniform translation, no rotation.
    center = (term.mass_weights[None, :, None]*data.body_com_pos_w).sum(1)
    offset = data.body_com_pos_w-center[:, None]
    omega = torch.tensor([0., 0., .5], dtype=torch.float64)
    data.body_link_ang_vel_w[1] = omega
    data.body_com_lin_vel_w[1] = torch.cross(omega.expand_as(offset[1]), offset[1], dim=-1)
    expected = -torch.expm1(torch.full((2,), -1., dtype=torch.float64))
    torch.testing.assert_close(term(env)/term.scale, expected)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Tracking/com_position_error_m'] == pytest.approx(0., abs=1e-14)
    assert logs['WBCForce/Tracking/com_linear_velocity_error_m_s'] == pytest.approx(.1)
    assert logs['WBCForce/Tracking/com_angular_velocity_error_rad_s'] == pytest.approx(.25)
    # Position and both velocity costs add inside the bounded penalty.
    data.body_link_ang_vel_w[0] = omega
    data.body_com_lin_vel_w[0] += torch.cross(omega.expand_as(offset[0]), offset[0], dim=-1)
    data.body_com_pos_w[0, :, 0] += .05
    torch.testing.assert_close(term(env)[0]/term.scale[0], -torch.expm1(torch.tensor(-3., dtype=torch.float64)))


def test_reference_centroidal_velocity_recovers_rigid_motion_with_rotated_com_offsets(rewards):
    env = make_env(steps=(0,))
    command = env.command_manager.get_term('motion')
    motion, data = command.motion, command.robot.data
    weights = torch.tensor([2., 4., 1.], dtype=torch.float64)/7
    center = (weights[None, :, None]*data.body_com_pos_w).sum(1)[0]
    omega = torch.tensor([.1, -.2, .3], dtype=torch.float64)
    linear = torch.tensor([.2, -.1, .05], dtype=torch.float64)
    # Set an instantaneous rigid twist using NPZ link-origin velocities.
    motion._body_ang_vel_w[:] = omega
    motion._body_lin_vel_w[0] = linear+torch.cross(
        omega.expand_as(motion._body_pos_w[0]), motion._body_pos_w[0]-center, dim=-1)
    data.body_link_ang_vel_w[:] = omega
    data.body_com_lin_vel_w[:] = linear+torch.cross(
        omega.expand_as(data.body_com_pos_w), data.body_com_pos_w-center, dim=-1)
    term = rewards.COMTrackingPenalty(term_cfg(), env)
    torch.testing.assert_close(term.reference_velocity[0], linear)
    torch.testing.assert_close(term.reference_angular_velocity[0], omega)
    torch.testing.assert_close(term(env), torch.zeros(1, dtype=torch.float64), atol=1e-15, rtol=0)
    # Other frames can be unrelated; no numerical time differentiation is used.
    motion._body_lin_vel_w[1:] = 1000.
    motion._body_ang_vel_w[1:] = -1000.
    other = rewards.COMTrackingPenalty(term_cfg(), env)
    torch.testing.assert_close(other(env), term(env))
    assert not hasattr(other, 'reference_acceleration')


@pytest.mark.parametrize('field', ['velocity_error_scale', 'angular_velocity_error_scale'])
@pytest.mark.parametrize('scale', [0., -.1, float('nan'), float('inf')])
def test_com_rejects_invalid_velocity_scales(rewards, field, scale):
    cfg = term_cfg()
    cfg.params[field] = scale
    with pytest.raises(ValueError, match=field):
        rewards.COMTrackingPenalty(cfg, make_env())


@pytest.mark.parametrize('name', ['SolePointTrackingPenalty', 'COMTrackingPenalty'])
@pytest.mark.parametrize('frames,dt', [(1, .02), (10, .02), (261, .02), (100, .01)])
def test_tracking_clip_cost_preserves_normalization_and_bounds(rewards, name, frames, dt):
    env = make_env(frames=frames, dt=dt)
    term = getattr(rewards, name)(term_cfg(), env)
    data = term.command.robot.data
    data.body_link_pos_w += torch.tensor([.03, .04, 0.])
    data.body_com_pos_w += torch.tensor([.03, .04, 0.])
    exponent = 8. if name == 'SolePointTrackingPenalty' else 1.
    expected = -25*(1.-torch.exp(torch.tensor(-exponent, dtype=torch.float64)))
    # Shared per-clip scaling is float32, including when geometry is float64.
    torch.testing.assert_close(term(env)*dt*term.cfg.weight*frames, expected.expand(2), atol=4e-6, rtol=1e-7)
    data.body_link_pos_w += 100.
    data.body_com_pos_w += 100.
    env.termination_manager.terminated[:] = True
    torch.testing.assert_close(term(env)*dt*term.cfg.weight*frames,
                               torch.full((2,), -25., dtype=torch.float64), atol=4e-6, rtol=1e-7)


@pytest.mark.parametrize('name', ['SolePointTrackingPenalty', 'COMTrackingPenalty'])
def test_tracking_normalizes_per_clip_in_a_motion_library(rewards, name):
    env = make_env(frames=15, steps=(0, 5))
    term = getattr(rewards, name)(term_cfg(), env)
    term.command.motion.clip_bounds = lambda frame: (torch.where(frame < 5, 0, 5), torch.where(frame < 5, 5, 10))
    data = term.command.robot.data
    data.body_link_pos_w += .01
    data.body_com_pos_w += .01
    values = term(env)
    torch.testing.assert_close(values[0], 2*values[1])


@pytest.mark.parametrize('name', ['SolePointTrackingPenalty', 'COMTrackingPenalty'])
@pytest.mark.parametrize('scale', [0., -.1, float('nan'), float('inf')])
def test_tracking_rejects_invalid_error_scale(rewards, name, scale):
    with pytest.raises(ValueError, match='error_scale'):
        getattr(rewards, name)(term_cfg(scale), make_env())


def test_force2_reward_configuration_replaces_joint_tracking(rewards):
    tree = ast.parse((SOURCE/'tasks/tracking/config/t1_wbc_force2/flat_env_cfg.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'WBCForceRewardsCfg')
    namespace = dict(vars(rewards), configclass=lambda cls: cls, RewardTermCfg=NS)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<force2_rewards_cfg>', 'exec'), namespace)
    config = namespace['WBCForceRewardsCfg']()
    assert not hasattr(config, 'joint_tracking') and not hasattr(rewards, 'JointTrackingPenalty')
    assert config.sole_point_tracking.func is rewards.SolePointTrackingPenalty
    assert config.com_tracking.func is rewards.COMTrackingPenalty
    assert config.sole_point_tracking.weight == -.25
    assert config.sole_point_tracking.params == {'error_scale': .05}
    assert config.com_tracking.weight == -.25
    assert config.com_tracking.params == {
        'error_scale': .05, 'velocity_error_scale': .2, 'angular_velocity_error_scale': .5}
    assert config.motion_completion.weight == 1.
    assert config.swing_force.weight == -.125
    assert config.action_change.weight == -.5
    assert config.false_zero.weight == -.1
    assert config.force_cap_gap.weight == -.08
