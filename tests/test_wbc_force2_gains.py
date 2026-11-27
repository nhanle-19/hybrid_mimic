"""FORCE2 adaptive PD: physical routing, cap isolation, and reward scaling."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, action_modules, DEVICES
from test_wbc_force_rewards import point_env, rewards as force_rewards
from test_wbc_force2_rewards import rewards, reward_config
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_gpu_control import select_state
from wbc_force_gpu_control import WBCCornerForceGPUQP, WBCCornerForcePDGPUQP, encode_force_capacities
from wbc_contact_mapping import decode_force_values, decode_force_values_numpy


@pytest.fixture
def gain_cfg(cfg):
    cfg.posture_position_gain, cfg.posture_velocity_gain = 225., 30.
    cfg.learned_posture_gains, cfg.posture_gain_relative_range = True, .5
    cfg.point_force_max_body_weights = 1.
    return cfg


def test_gain_mapping_and_cap_isolation(gain_cfg):
    raw = torch.tensor([[0., 0.], [-100., 100.], [.5, -.5]], dtype=torch.float64)
    solver = WBCCornerForcePDGPUQP(gain_cfg)
    actions = torch.cat((torch.zeros(3, 8), raw), -1)
    gains = solver.posture_gains(actions)
    torch.testing.assert_close(gains[:2], raw.new_tensor([[225., 30.], [112.5, 45.]]))
    assert 225 < gains[2, 0] < 337.5 and 15 < gains[2, 1] < 30
    torch.testing.assert_close(decode_force_values(actions), decode_force_values(actions[:, :8]))
    np.testing.assert_allclose(decode_force_values_numpy(actions.numpy()), decode_force_values(actions).numpy())
    assert solver.action_names[-2:] == ('posture_kp', 'posture_kd')
    assert solver.action_transform['posture_gains']['defaults'] == [225., 30.]
    for invalid in (torch.zeros(1, 8), torch.full((1, 10), float('nan'))):
        with pytest.raises(ValueError, match='10 finite'):
            solver.decode_policy(invalid)


@pytest.mark.parametrize('field,value', [('posture_gain_relative_range', 0.),
    ('posture_gain_relative_range', 1.), ('posture_gain_relative_range', float('nan')),
    ('posture_position_gain', -1.), ('posture_velocity_gain', float('inf'))])
def test_invalid_gain_configuration_rejected(gain_cfg, field, value):
    setattr(gain_cfg, field, value)
    with pytest.raises(ValueError):
        WBCCornerForcePDGPUQP(gain_cfg)


@pytest.mark.parametrize('device', DEVICES)
def test_gain_actions_reach_physical_qp_and_match_independent_fixed_gain_solves(gain_cfg, device):
    model = WBCACCTorchModel(device)
    state = standing_state(model, 4)
    ref = copy.deepcopy(state)
    ref['q'][:, 7:] += torch.linspace(-.003, .003, 23, device=device)
    ref['v'][:, 6:] += torch.linspace(.04, -.02, 23, device=device)
    caps = encode_force_capacities(state['q'].new_full((4, 8), .8), 8)
    raw_gains = caps.new_tensor([[0., 0.], [2., 0.], [0., 2.], [-2., -2.]])
    actions = torch.cat((caps, raw_gains), -1)
    masks = torch.ones((4, 2, 4), dtype=torch.bool, device=device)
    limits = caps.new_full((4, 23), 60.)
    result = WBCCornerForcePDGPUQP(gain_cfg).solve(state, ref, actions, masks, limits)
    assert not result['failed'].any()
    assert (result['torque'].abs() <= limits+2e-5).all()
    assert result['residual'].max() < 2e-5
    for i in range(4):
        fixed_cfg = copy.copy(gain_cfg)
        fixed_cfg.posture_position_gain, fixed_cfg.posture_velocity_gain = result['posture_gains'][i].tolist()
        # Independent OSQP solve with scalar gains, rather than policy decoding.
        expected = WBCCornerForceGPUQP(fixed_cfg, backend='osqp').solve(
            select_state(state, slice(i, i+1)), select_state(ref, slice(i, i+1)),
            caps[i:i+1], masks[i:i+1], limits[i:i+1])
        assert not expected['failed'].any()
        torch.testing.assert_close(result['torque'][i:i+1], expected['torque'], atol=.01, rtol=1e-3)
        torch.testing.assert_close(result['acceleration'][i:i+1], expected['acceleration'], atol=2e-3, rtol=1e-3)
    assert (result['torque'][1:]-result['torque'][0]).norm(dim=-1).min() > .001
    # Disabled mode is exactly the archived eight-output controller.
    gain_cfg.learned_posture_gains = False
    legacy = WBCCornerForcePDGPUQP(gain_cfg).solve(state, ref, caps, masks, limits)
    fixed = WBCCornerForceGPUQP(gain_cfg).solve(state, ref, caps, masks, limits)
    torch.testing.assert_close(legacy['torque'], fixed['torque'], atol=0, rtol=0)
    torch.testing.assert_close(result['torque'][0], fixed['torque'][0], atol=1e-8, rtol=0)


@pytest.mark.parametrize('frames,dt', [(10, .02), (200, .01)])
def test_gain_penalty_is_relative_quadratic_and_clip_normalized(rewards, reward_config, gain_cfg, frames, dt):
    env = point_env(frames=frames, count=4, dt=dt)
    env.action_manager.get_term('wbc_force').cfg.controller = gain_cfg
    env.action_manager.action = torch.zeros(4, 10, dtype=torch.float64)
    term = rewards.TrackingGainDeviationPenalty(
        NS(params={}, weight=reward_config.tracking_gain_deviation.weight), env)
    assert term(env).count_nonzero() == 0
    # Half the maximum deviation has one quarter the cost, for either sign.
    env.action_manager.action[:, 8:] = torch.tensor([[100., 100.], [.5, .5], [-.5, -.5], [100., 0.]])
    env.action_manager.action[1:3, 8:] = torch.atanh(torch.tensor([[.5, .5], [-.5, -.5]]))
    total = term(env)*dt*frames*term.cfg.weight
    torch.testing.assert_close(total, total.new_tensor([-.5, -.125, -.125, -.25]), atol=1e-6, rtol=1e-7)
    # Even maximum gain deviation leaves at least 99.5% of a surviving frame's
    # completion credit before the separate contact/tracking penalties.
    completion = rewards.MotionCompletionReward(NS(params={}, weight=reward_config.motion_completion.weight), env)
    alive = (completion(env)*dt*completion.cfg.weight).to(total)
    assert (alive > 0).all()
    assert ((alive+total/frames) >= alive*.995-1e-8).all()
    # Relative penalty is unchanged by different configured nominal gains.
    gain_cfg.posture_position_gain *= 2
    gain_cfg.posture_velocity_gain *= 3
    torch.testing.assert_close(term(env)*dt*frames*term.cfg.weight, total)
    gain_cfg.learned_posture_gains = False
    env.action_manager.action = env.action_manager.action[:, :8]
    assert term(env).count_nonzero() == 0


def test_all_contact_rewards_ignore_appended_gain_outputs(rewards):
    env = point_env(count=2)
    env.action_manager.action = torch.zeros(2, 10)
    terms = [cls(NS(params={}, weight=-.1), env) for cls in
             (rewards.SwingForcePenalty, rewards.FalseZeroPenalty, rewards.ForceCapGapPenalty,
              rewards.ForceActionChangePenalty)]
    before = [term(env).clone() for term in terms]
    env.action_manager.action[:, 8:] = torch.tensor([[-100., 100.], [100., -100.]])
    for term, expected in zip(terms, before):
        torch.testing.assert_close(term(env), expected)
    completion = rewards.MotionCompletionReward(NS(params={}, weight=1.), env)
    completion(env)
    logs = rewards._metrics(env).drain()
    assert logs['WBCForce/Caps/left_mean'] == pytest.approx(4/9)
    assert logs['WBCForce/Caps/right_corner_3_mean'] == pytest.approx(4/9)
    terms[-1].reset(torch.tensor([1]))
    env.action_manager.action[:, :8] = 40.
    change = terms[-1](env)
    assert change[0] > 0 and change[1] == 0


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('randomized', [False, True])
def test_force2_action_initializes_and_resets_gain_outputs(action_modules, monkeypatch, enabled, randomized):
    action, gpu = action_modules
    aliases = {
        'gain_tasks.t1_wbc_acc.wbc_acc_action': action,
        'gain_tasks.t1_wbc_acc.wbc_acc_gpu_action': gpu,
        'gain_tasks.t1_wbc_acc.controller_cfg': sys.modules['wbc_acc_gpu_test.controller_cfg'],
        'whole_body_tracking.utils.wbc_force_gpu_control': sys.modules['wbc_force_gpu_control'],
    }
    for name, module in aliases.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking'
    for name, path in [('whole_body_tracking.utils.wbc_force_randomization', source/'utils/wbc_force_randomization.py'),
                       *[(f'gain_tasks.t1_wbc_force2.{name}', source/f'tasks/tracking/config/t1_wbc_force2/{name}.py')
                         for name in ('controller_cfg', 'wbc_force_action')]]:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    config = module.WBCForceActionCfg()
    config.backend, config.domain_randomization = 'osqp', randomized
    config.controller.learned_posture_gains = enabled
    model = WBCACCTorchModel()
    asset = NS(joint_names=model.joint_names, data=NS(joint_effort_limits=torch.full((2, 23), 60.),
               joint_stiffness=torch.zeros(2, 23), joint_damping=torch.zeros(2, 23)))
    env = NS(device='cpu', num_envs=2, scene={'robot': asset})
    term = config.class_type(config, env)
    assert term.action_dim == (10 if enabled else 8)
    raw = torch.randn(2, term.action_dim)
    term.process_actions(raw)
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.raw_actions[0], raw[0])
    assert term.raw_actions[1].count_nonzero() == 0
    if enabled:
        torch.testing.assert_close(term.solver.posture_gains(term.raw_actions)[1], torch.tensor([225., 30.]))
    with pytest.raises(ValueError, match='finite policy actions'):
        term.process_actions(torch.zeros(2, 8 if enabled else 10))
