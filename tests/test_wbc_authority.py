"""Directional authority mapping, coupled QP constraints, and physical invariants."""
import copy
import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, DEVICES, action_modules
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_contact_policy import available_vertices
from wbc_authority_gpu_control import WBCAuthorityGPUQP, decode_authority_actions, encode_authorities


@pytest.fixture
def authority_cfg(cfg):
    cfg.contact_wrench_max = (360., 360., 600., 60., 60., 30.)
    cfg.contact_authority_weights = (100.,)*6
    return cfg


def test_smooth_bounded_authority_and_manual_roundtrip():
    logits = torch.linspace(-4., 4., 12, dtype=torch.float64).reshape(1, 12).requires_grad_()
    c = decode_authority_actions(logits)
    assert c.shape == (1, 2, 6) and ((c > 0) & (c < 1)).all()
    assert (c.flatten().diff() > 0).all()
    c.sum().backward()
    assert (logits.grad > 0).all()
    torch.testing.assert_close(encode_authorities(c), logits)
    ends = torch.tensor([[[0., 1., .5, .2, .8, .9]]*2], dtype=torch.float64)
    torch.testing.assert_close(decode_authority_actions(encode_authorities(ends)), ends)
    assert decode_authority_actions(encode_authorities(ends))[0, 0, 0] == 0


@pytest.mark.parametrize('actions', [torch.zeros(1, 2), torch.zeros(12), torch.full((1, 12), float('nan'))])
def test_invalid_actions(actions):
    with pytest.raises(ValueError, match='twelve finite'):
        decode_authority_actions(actions)


@pytest.mark.parametrize('device', DEVICES)
def test_coupled_constraints_and_dynamics(authority_cfg, device):
    state = standing_state(WBCACCTorchModel(device), 4)
    c = torch.full((4, 2, 6), .8, dtype=torch.float64, device=device)
    c[1, 0] = torch.tensor([.001, .002, .01, .001, .002, .001], device=device)
    c[2] = 0.  # Detached: no force and no soft contact objective.
    masks = available_vertices(state, authority_cfg)
    masks[3, 0] = False
    limits = torch.full((4, 23), 60., dtype=torch.float64, device=device)
    actions = encode_authorities(c)
    solver = WBCAuthorityGPUQP(authority_cfg)
    result = solver.solve(state, state, actions, masks, limits)
    assert not result['failed'].any(), result['residual']
    wrench = result['wrenches']
    assert (wrench.abs() <= result['wrench_limits']+2e-5).all()
    assert (wrench[..., 2] >= -2e-5).all()
    assert (wrench[..., :2].norm(dim=-1) <= .6*wrench[..., 2]+2e-5).all()
    assert wrench[2].count_nonzero() == 0
    assert wrench[3, 0].count_nonzero() == 0
    assert result['contact_weights'][2].count_nonzero() == 0
    assert result['contact_weights'][3, 0].count_nonzero() == 0
    torch.testing.assert_close(result['contact_weights'][1, 0], 100*c[1, 0].square())
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
    c = torch.tensor([[[.2, .3, .4, .5, .6, .7]]*2], dtype=torch.float64)
    _, weights = solver.decode_policy(encode_authorities(c))
    frame = state['frames']['left_foot_link']
    target, effective = solver.contact_objective(frame, None, weights[:, 0], torch.tensor([True]), torch.as_tensor)
    assert target.count_nonzero() == 0
    torch.testing.assert_close(effective, 100*c[:, 0].square())
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


@pytest.mark.parametrize('corner', range(4))
def test_zero_moment_authority_at_single_corner_matches_osqp(authority_cfg, corner):
    """A corner cannot exert pure vertical force with all moments fixed to zero.

    This feasible boundary used to make the batched solver diverge, terminating
    episodes even though the independent OSQP solve succeeded.
    """
    state = standing_state(WBCACCTorchModel())
    masks = torch.zeros((1,2,4), dtype=torch.bool)
    masks[0,0,corner] = True
    values = torch.tensor([[[0.,0.,1.,0.,0.,0.]]*2], dtype=torch.float64)
    limits = torch.full((1,23),60.,dtype=torch.float64)
    args = (state,state,encode_authorities(values),masks,limits)
    actual = WBCAuthorityGPUQP(authority_cfg, solver_tolerance=1e-9).solve(*args)
    expected = WBCAuthorityGPUQP(authority_cfg, backend='osqp').solve(*args)
    assert not actual['failed'].any() and not expected['failed'].any()
    assert actual['wrenches'].abs().max() < 2e-5
    torch.testing.assert_close(actual['acceleration'],expected['acceleration'],atol=2e-3,rtol=1e-3)


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
    args = (state, reference, data['actions'], data['masks'], data['limits'])
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
    assert term.action_dim == 12 and isinstance(term.solver, WBCAuthorityGPUQP)
    raw = torch.linspace(-3., 3., 24).reshape(2, 12)
    term.process_actions(raw)
    torch.testing.assert_close(term.raw_actions, raw)
    term.reset(torch.tensor([1]))
    assert term.raw_actions[1].count_nonzero() == 0
    with pytest.raises(ValueError, match='12 finite'):
        term.process_actions(torch.zeros(2, 2))
    caps = torch.full((2, 2, 6), .5)
    caps[0, 0, 0] = .7
    env.action_manager = SimpleNamespace(action=encode_authorities(caps), prev_action=torch.zeros(2, 12))
    torch.testing.assert_close(modules['rewards'].mapped_authority_action_rate(env), torch.tensor([.04/12, 0.]))


def test_each_axis_scales_its_wrench_bound(authority_cfg):
    from wbc_acc_contact_policy import sole_vertices
    state = standing_state(WBCACCTorchModel())
    solver = WBCAuthorityGPUQP(authority_cfg)
    points = sole_vertices('cpu', torch.float64)
    rays = torch.tensor([[0., 0., -.6, .6], [.6, -.6, 0., 0.], [1., 1., 1., 1.]], dtype=torch.float64)
    enabled = torch.ones(1, 32, dtype=torch.bool)
    c = torch.tensor([[[.2, .3, .4, .5, .6, .7], [.7, .6, .5, .4, .3, .2]]], dtype=torch.float64)
    activation, weights = solver.decode_policy(encode_authorities(c))
    rows, bound = solver.extra_wrench_constraints(state, points, rays, enabled, 29, encode_authorities(c))
    torch.testing.assert_close(activation, c[:, :, 2])
    expected = (c*torch.tensor(authority_cfg.contact_wrench_max))[:, :, [0, 1, 3, 4, 5]].flatten(1)
    torch.testing.assert_close(bound, torch.cat((expected, expected), 1))
    # One point-force ray: moments must be r x f at the foot origin,
    # not moments about the world origin or center of mass.
    x = torch.zeros(1, 61, dtype=torch.float64)
    x[0, 29+3] = 1.
    f = rays[:, 3]
    offset = state['frames']['left_foot_link']['rotation'][0]@points[0, 0]
    wrench = torch.cat((f, torch.cross(offset, f, dim=0)))[[0, 1, 3, 4, 5]]
    actual = mv(rows, x)[0]
    torch.testing.assert_close(actual[:5], wrench)
    torch.testing.assert_close(actual[10:15], -wrench)


@pytest.mark.parametrize('axis', [0, 1, 3, 4, 5])
def test_exact_zero_direction_is_an_equality(authority_cfg, axis):
    state = standing_state(WBCACCTorchModel(), 3)
    c = torch.ones((3, 2, 6), dtype=torch.float64)
    c[0, 0, axis] = 0.
    c[1, 0, axis] = 1e-10
    # A mixed batch must retain other environments' independent constraints.
    limits = torch.full((3, 23), 60., dtype=torch.float64)
    masks = available_vertices(state, authority_cfg)
    actions = encode_authorities(c)
    result = WBCAuthorityGPUQP(authority_cfg).solve(state, state, actions, masks, limits)
    expected = WBCAuthorityGPUQP(authority_cfg, backend='osqp').solve(state, state, actions, masks, limits)
    assert not result['failed'].any() and not expected['failed'].any()
    assert abs(result['wrenches'][0, 0, axis]) < 1e-10
    torch.testing.assert_close(result['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)


def test_low_support_is_feasible_free_acceleration(authority_cfg):
    state = standing_state(WBCACCTorchModel())
    actions = encode_authorities(torch.full((1, 2, 6), .1, dtype=torch.float64))
    result = WBCAuthorityGPUQP(authority_cfg).solve(state, state, actions,
        available_vertices(state, authority_cfg), torch.full((1, 23), 60., dtype=torch.float64))
    assert not result['failed'].any()
    # Insufficient upward support is a valid falling solution, not infeasibility.
    assert result['acceleration'][0, 2] < -1.
    assert (result['wrenches'][..., 2] <= 60.+2e-5).all()


def test_tiny_authority_not_lost_by_squaring(authority_cfg):
    from wbc_acc_contact_policy import sole_vertices
    state = standing_state(WBCACCTorchModel())
    actions = torch.zeros((1, 12), dtype=torch.float64)
    actions[0, 0] = -400.
    solver = WBCAuthorityGPUQP(authority_cfg)
    rays = torch.tensor([[0., 0., -.6, .6], [.6, -.6, 0., 0.], [1., 1., 1., 1.]], dtype=torch.float64)
    _, cap = solver.extra_wrench_constraints(state, sole_vertices(), rays, torch.ones(1, 32, dtype=torch.bool), 29, actions)
    assert cap[0, 0] > 0.
    torch.testing.assert_close(cap[0, 0], 360*actions[0, 0].sigmoid(), atol=0, rtol=1e-12)
