"""Cost-only translational authority and physical contact invariants."""
import copy
import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, DEVICES, action_modules
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_contact_policy import available_vertices
from wbc_authority_gpu_control import WBCAuthorityGPUQP, decode_authority_actions, encode_authorities


@pytest.fixture
def authority_cfg(cfg):
    cfg.contact_authority_weights = (100.,)*3
    return cfg


def test_interior_authority_and_manual_roundtrip():
    logits = torch.linspace(-2., 2., 6, dtype=torch.float64).reshape(1, 6).requires_grad_()
    c = decode_authority_actions(logits)
    assert c.shape == (1, 2, 3) and ((c > 0) & (c < 1)).all()
    assert (c.flatten().diff() > 0).all()
    c.sum().backward()
    assert (logits.grad > 0).all()
    torch.testing.assert_close(encode_authorities(c), logits)
    ends = torch.tensor([[[0., 1., .5]]*2], dtype=torch.float64)
    torch.testing.assert_close(decode_authority_actions(encode_authorities(ends)), ends)
    assert decode_authority_actions(encode_authorities(ends))[0, 0, 0] == 0
    assert encode_authorities(ends).tolist() == [[-3., 3., 0., -3., 3., 0.]]


def test_authority_reaches_both_endpoints_with_ordinary_raw_outputs():
    actions = torch.tensor([[-4., -3., 0., 0., 3., 4.]])
    torch.testing.assert_close(decode_authority_actions(actions),
                               torch.tensor([[[0., 0., .5], [.5, 1., 1.]]]))


@pytest.mark.parametrize('actions', [torch.zeros(1, 12), torch.zeros(6), torch.full((1, 6), float('nan'))])
def test_invalid_actions(actions):
    with pytest.raises(ValueError, match='six finite'):
        decode_authority_actions(actions)


@pytest.mark.parametrize('device', DEVICES)
def test_cost_only_authority_preserves_support_and_dynamics(authority_cfg, device):
    state = standing_state(WBCACCTorchModel(device), 4)
    c = torch.full((4, 2, 3), .8, dtype=torch.float64, device=device)
    c[1, 0] = torch.tensor([.001, .002, .01], device=device)
    c[2] = 0.  # Remove only the contact cost; support must remain available.
    masks = available_vertices(state, authority_cfg)
    masks[3, 0] = False
    limits = torch.full((4, 23), 60., dtype=torch.float64, device=device)
    actions = encode_authorities(c)
    solver = WBCAuthorityGPUQP(authority_cfg)
    result = solver.solve(state, state, actions, masks, limits)
    assert not result['failed'].any(), result['residual']
    wrench = result['wrenches']
    assert 'wrench_limits' not in result
    torch.testing.assert_close(result['normal_force_limits'], torch.full_like(c[..., 0], 600.))
    assert (wrench[..., 2] <= result['normal_force_limits']+2e-5).all()
    assert result['contact_weights'][..., 3:].count_nonzero() == 0
    assert (wrench[..., 2] >= -2e-5).all()
    assert (wrench[..., :2].norm(dim=-1) <= .6*wrench[..., 2]+2e-5).all()
    assert wrench[2, :, 2].sum() > 100.  # Zero cost still permits supporting force.
    assert wrench[3, 0].count_nonzero() == 0
    assert result['contact_weights'][2].count_nonzero() == 0
    assert result['contact_weights'][3, 0].count_nonzero() == 0
    torch.testing.assert_close(result['contact_weights'][1, 0, :3], 100*c[1, 0].square())
    generalized = sum(mv(state['frames'][foot]['J'].transpose(-1, -2), wrench[:, i])
                      for i, foot in enumerate(('left_foot_link', 'right_foot_link')))
    inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)
    assert (result['torque'].abs() <= limits+2e-5).all()
    expected = WBCAuthorityGPUQP(authority_cfg, backend='osqp').solve(state, state, actions, masks, limits)
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)


def test_contact_objective_uses_zero_acceleration_and_squared_authority(authority_cfg):
    state = standing_state(WBCACCTorchModel())
    solver = WBCAuthorityGPUQP(authority_cfg)
    c = torch.tensor([[[.2, .3, .4]]*2], dtype=torch.float64)
    _, weights = solver.decode_policy(encode_authorities(c))
    frame = state['frames']['left_foot_link']
    target, effective = solver.contact_objective(frame, None, weights[:, 0], torch.tensor([True]), torch.as_tensor)
    assert target.count_nonzero() == 0
    torch.testing.assert_close(effective[:, :3], 100*c[:, 0].square())
    assert effective[:, 3:].count_nonzero() == 0
    # Foot reference accelerations/velocities must not pull the contact task
    # toward the moving reference: this task resists actual acceleration.
    reference = copy.deepcopy(state)
    for foot in ('left_foot_link', 'right_foot_link'):
        reference['frames'][foot]['acceleration'].fill_(123.)
        reference['frames'][foot]['velocity'].fill_(10.)
    masks = available_vertices(state, authority_cfg)
    limits = torch.full((1, 23), 60., dtype=torch.float64)
    expected = solver.solve(state, state, encode_authorities(c), masks, limits)
    actual = solver.solve(state, reference, encode_authorities(c), masks, limits)
    assert not actual['failed'].any()
    torch.testing.assert_close(actual['torque'], expected['torque'])


@pytest.mark.parametrize('device', DEVICES)
def test_gpu_rollout_precision_regressions_match_osqp(authority_cfg, device):
    """Captured feasible states previously terminated at a 1.4e-7 KKT floor."""
    from pathlib import Path
    import numpy as np
    with np.load(Path(__file__).parent/'fixtures/wbc_authority_precision.npz') as archive:
        data = {k: torch.as_tensor(archive[k], device=device) for k in archive.files}
    model = WBCACCTorchModel(device)
    state = model.state(data['state_q'], data['state_v'])
    reference = model.state(data['reference_q'], data['reference_v'])
    # Keep captured states; explicitly project historical xyz/rxyz actions to xyz.
    actions = data['actions'].reshape(-1, 2, 6)[..., :3].reshape(-1, 6)
    args = (state, reference, actions, data['masks'], data['limits'])
    actual = WBCAuthorityGPUQP(authority_cfg, max_iterations=100, solver_tolerance=1e-6).solve(*args)
    expected = WBCAuthorityGPUQP(authority_cfg, backend='osqp', solver_tolerance=1e-9).solve(*args)
    assert not actual['failed'].any() and not expected['failed'].any()
    assert actual['residual'].max() < 2e-5
    torch.testing.assert_close(actual['torque'], expected['torque'], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(actual['acceleration'], expected['acceleration'], atol=1e-3, rtol=1e-4)


def test_action_interface_and_reward(action_modules, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    action, gpu = action_modules
    aliases = {
        'authority_tasks.t1_wbc_acc.wbc_acc_action': action,
        'authority_tasks.t1_wbc_acc.wbc_acc_gpu_action': gpu,
        'authority_tasks.t1_wbc_acc.controller_cfg': sys.modules['wbc_acc_gpu_test.controller_cfg'],
        'whole_body_tracking.utils.wbc_authority_gpu_control': sys.modules['wbc_authority_gpu_control'],
    }
    for name, module in aliases.items():
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_authority'
    modules = {}
    for name in ('controller_cfg', 'wbc_authority_action', 'rewards'):
        spec = importlib.util.spec_from_file_location('authority_tasks.t1_wbc_authority.'+name, root/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    config = modules['wbc_authority_action'].WBCAuthorityActionCfg()
    config.backend = 'osqp'
    model = WBCACCTorchModel()
    asset = SimpleNamespace(joint_names=model.joint_names,
                            data=SimpleNamespace(joint_effort_limits=torch.full((2, 23), 60.),
                                                 joint_stiffness=torch.zeros((2, 23)),
                                                 joint_damping=torch.zeros((2, 23))))
    env = SimpleNamespace(device='cpu', num_envs=2, scene={'robot': asset})
    term = config.class_type(config, env)
    assert term.action_dim == 6 and isinstance(term.solver, WBCAuthorityGPUQP)
    raw = torch.linspace(-3., 3., 12).reshape(2, 6)
    term.process_actions(raw)
    torch.testing.assert_close(term.raw_actions, raw)
    term.reset(torch.tensor([1]))
    assert term.raw_actions[1].count_nonzero() == 0
    with pytest.raises(ValueError, match='6 finite'):
        term.process_actions(torch.zeros(2, 2))
    caps = torch.full((2, 2, 3), .5)
    caps[0, 0, 0] = .7
    env.action_manager = SimpleNamespace(action=encode_authorities(caps), prev_action=torch.zeros(2, 6))
    torch.testing.assert_close(modules['rewards'].mapped_authority_action_rate(env), torch.tensor([.04/6, 0.]))



@pytest.mark.parametrize('corner', range(4))
def test_single_corner_support_has_no_artificial_moment_cap(authority_cfg, corner):
    state = standing_state(WBCACCTorchModel())
    masks = torch.zeros((1, 2, 4), dtype=torch.bool)
    masks[0, 0, corner] = True
    actions = encode_authorities(torch.ones((1, 2, 3), dtype=torch.float64))
    args = (state, state, actions, masks, torch.full((1, 23), 60., dtype=torch.float64))
    actual = WBCAuthorityGPUQP(authority_cfg).solve(*args)
    expected = WBCAuthorityGPUQP(authority_cfg, backend='osqp').solve(*args)
    assert not actual['failed'].any() and not expected['failed'].any()
    assert actual['wrenches'][0, 0, 2] > 1.
    assert actual['wrenches'][0, 0, 3:].abs().max() > .01
    torch.testing.assert_close(actual['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)


def test_authority_changes_cost_but_not_qp_constraints(authority_cfg, monkeypatch):
    import wbc_acc_gpu_control as base
    calls = []
    original = base.solve_batched_qp

    def record(h, g, inequality, bound, **kwargs):
        calls.append(tuple(t.clone() for t in (h, g, inequality, bound)))
        return original(h, g, inequality, bound, **kwargs)

    monkeypatch.setattr(base, 'solve_batched_qp', record)
    state = standing_state(WBCACCTorchModel())
    masks = available_vertices(state, authority_cfg)
    limits = torch.full((1, 23), 60., dtype=torch.float64)
    solver = WBCAuthorityGPUQP(authority_cfg)
    for value in (0., 1.):
        actions = encode_authorities(torch.full((1, 2, 3), value, dtype=torch.float64))
        result = solver.solve(state, state, actions, masks, limits)
        assert not result['failed'].any()
    low, high = calls
    assert not torch.allclose(low[0], high[0])
    torch.testing.assert_close(low[2], high[2], atol=0, rtol=0)
    torch.testing.assert_close(low[3], high[3], atol=0, rtol=0)
