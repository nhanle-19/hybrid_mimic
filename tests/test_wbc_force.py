"""Learned force caps, absent foot objectives, and the two-output action interface."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, action_modules, DEVICES
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_force_gpu_control import WBCForceGPUQP, decode_force_actions, encode_force_capacities


@pytest.mark.parametrize('device', DEVICES)
def test_force_limits_and_inverse_dynamics(cfg, device):
    model = WBCACCTorchModel(device)
    state = standing_state(model, 5)
    q = state['q'].clone()
    q[4, 2] += .2  # Floor clearance cannot silently override the learned caps.
    state = model.state(q, state['v'])
    actions = torch.tensor([[1., 1.], [.01, 1.], [0., 1.], [0., 0.], [1., 1.]],
                           dtype=torch.float64, device=device)
    limits = torch.full((5, 23), 60., dtype=torch.float64, device=device)
    masks = torch.ones((5, 2, 4), dtype=torch.bool, device=device)
    result = WBCForceGPUQP(cfg).solve(state, state, encode_force_capacities(actions), masks, limits)
    assert not result['failed'].any()
    force = result['wrenches'][:, :, :3]
    assert (force[:, :, 2] >= -2e-5).all()
    assert (force[:, :, 2] <= actions*600.+2e-5).all()
    assert (force[:, :, :2].norm(dim=-1) <= .6*force[:, :, 2]+2e-5).all()
    assert result['wrenches'][2, 0].count_nonzero() == 0
    assert result['wrenches'][3].count_nonzero() == 0
    assert (force[4, :, 2] > 1.).all()
    torch.testing.assert_close(result['wrenches'][4], result['wrenches'][0], atol=2e-5, rtol=0)
    torch.testing.assert_close(result['torque'][4], result['torque'][0], atol=2e-5, rtol=0)
    # The cap changes the optimum instead of being an unused action.
    assert force[0, 0, 2] > force[1, 0, 2]+1.
    assert abs(force[1, 0, 2].item()-6.) < .01
    assert (result['torque'].abs() <= limits+2e-5).all()
    generalized = sum(mv(state['frames'][foot]['J'].transpose(-1, -2), result['wrenches'][:, i])
                      for i, foot in enumerate(('left_foot_link', 'right_foot_link')))
    inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)
    expected = WBCForceGPUQP(cfg, backend='osqp').solve(state, state, encode_force_capacities(actions), masks, limits)
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)
    assert 'contact_weights' not in result


def test_no_reference_foot_acceleration_objective(cfg):
    state = standing_state(WBCACCTorchModel())
    actions = torch.ones((1, 2), dtype=torch.float64)
    limits = torch.full((1, 23), 60., dtype=torch.float64)
    masks = torch.ones((1, 2, 4), dtype=torch.bool)
    solver = WBCForceGPUQP(cfg)
    expected = solver.solve(state, state, actions, masks, limits)
    reference = copy.deepcopy(state)
    # Foot reference poses/velocities/accelerations are unnecessary in FORCE.
    del reference['frames']['left_foot_link']
    del reference['frames']['right_foot_link']
    result = solver.solve(state, reference, actions, masks, limits)
    assert not result['failed'].any()
    torch.testing.assert_close(result['torque'], expected['torque'], atol=1e-10, rtol=0)


def test_thousand_newton_caps_are_independent_not_normalized(cfg):
    cfg.contact_force_max = (1000., 1000.)
    state = standing_state(WBCACCTorchModel())
    fractions = torch.tensor([[.12, 1.]], dtype=torch.float64)
    result = WBCForceGPUQP(cfg).solve(state, state, encode_force_capacities(fractions),
        torch.ones((1, 2, 4), dtype=torch.bool), torch.full((1, 23), 60., dtype=torch.float64))
    assert not result['failed'].any()
    torch.testing.assert_close(result['normal_force_limits'], torch.tensor([[120., 1000.]], dtype=torch.float64))
    assert (result['wrenches'][:, :, 2] <= result['normal_force_limits']+2e-5).all()


@pytest.mark.parametrize('actions', [torch.zeros(2, 14), torch.zeros(2), torch.tensor([[float('nan'), 0.]])])
def test_reject_invalid_actions(actions):
    with pytest.raises(ValueError, match='two finite'):
        decode_force_actions(actions)


def test_force_mapping_and_disabled_swing(cfg):
    torch.testing.assert_close(decode_force_actions(torch.tensor([[-3., 3.], [0., 0.]])),
                               torch.tensor([[0., 1.], [.5, .5]]))
    cfg.swing_weight = 1.
    with pytest.raises(ValueError, match='foot-acceleration objectives are disabled'):
        WBCForceGPUQP(cfg)


@pytest.fixture
def force_modules(action_modules, monkeypatch):
    action, gpu = action_modules
    aliases = {
        'force_tasks.t1_wbc_acc.wbc_acc_action': action,
        'force_tasks.t1_wbc_acc.wbc_acc_gpu_action': gpu,
        'force_tasks.t1_wbc_acc.controller_cfg': sys.modules['wbc_acc_gpu_test.controller_cfg'],
        'whole_body_tracking.utils.wbc_force_gpu_control': sys.modules['wbc_force_gpu_control'],
    }
    for name, module in aliases.items():
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_force'
    modules = {}
    for name in ('controller_cfg', 'wbc_force_action', 'rewards'):
        spec = importlib.util.spec_from_file_location('force_tasks.t1_wbc_force.'+name, root/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_force_action_initialization_and_reset(force_modules):
    modules = force_modules
    config = modules['wbc_force_action'].WBCForceActionCfg()
    config.backend = 'osqp'
    model = WBCACCTorchModel()
    asset = SimpleNamespace(joint_names=model.joint_names,
                            data=SimpleNamespace(joint_effort_limits=torch.full((2, 23), 60.),
                                                 joint_stiffness=torch.zeros((2, 23)),
                                                 joint_damping=torch.zeros((2, 23))))
    env = SimpleNamespace(device='cpu', num_envs=2, scene={'robot': asset})
    term = config.class_type(config, env)
    assert term.action_dim == 2 and term.raw_actions.shape == (2, 2)
    assert isinstance(term.solver, WBCForceGPUQP)
    term.process_actions(torch.tensor([[.2, .7], [1., 0.]]))
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.raw_actions, torch.tensor([[.2, .7], [0., 0.]]))
    with pytest.raises(ValueError, match='2 finite'):
        term.process_actions(torch.zeros((2, 14)))
    env.action_manager = SimpleNamespace(action=torch.tensor([[2., -.2], [.5, .25]]),
                                        prev_action=torch.tensor([[1., 0.], [0., .25]]))
    torch.testing.assert_close(modules['rewards'].mapped_force_action_rate(env),
                               torch.tensor([0.012452522737547168, 0.00749814])/.8**2, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('record_diagnostics', [False, True])
@pytest.mark.parametrize('backend', ['batched', 'osqp'])
def test_force_action_uses_caps_without_geometry_gate(force_modules, action_modules, monkeypatch,
                                                     record_diagnostics, backend):
    """Training/evaluation action routing needs neither floor tests nor a contact override."""
    _, gpu = action_modules
    monkeypatch.setattr(gpu, 'available_vertices', lambda *args, **kwargs:
                        pytest.fail('FORCE called the privileged geometry detector'))
    config = force_modules['wbc_force_action'].WBCForceActionCfg()
    config.backend = backend
    config.allow_cpu_batched_evaluation = True
    config.record_diagnostics = record_diagnostics
    model = WBCACCTorchModel()
    standing = standing_state(model, 2)
    q = standing['q'].clone()
    q[:, 2] += .2
    state = model.state(q, standing['v'])
    applied = []
    asset = SimpleNamespace(joint_names=model.joint_names,
        data=SimpleNamespace(joint_effort_limits=torch.full((2, 23), 60.),
                             joint_stiffness=torch.zeros((2, 23)), joint_damping=torch.zeros((2, 23))),
        set_joint_effort_target=lambda torque: applied.append(torque.clone()))
    # No terrain/floor origin is supplied to the action's contact-selection path.
    env = SimpleNamespace(device='cpu', num_envs=2, scene={'robot': asset}, physics_dt=.002,
        cfg=SimpleNamespace(decimation=10), extras={},
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(time_steps=torch.arange(2))))
    term = config.class_type(config, env)
    term.references = state
    term._states = lambda: state
    term.validated = True
    if record_diagnostics:
        term._gpu_ground_forces = lambda: torch.zeros((2, 2, 3))
        term._actual_wrenches = lambda: torch.zeros((2, 2, 6))
    else:
        term._gpu_ground_forces = lambda: pytest.fail('FORCE required measured ground forces')
        term._actual_wrenches = lambda: pytest.fail('FORCE required measured contact wrenches')
    term.set_active_contacts(None)
    with pytest.raises(ValueError):
        term.set_active_contacts(torch.tensor([[False, True], [True, True]]))
    caps = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    term.process_actions(encode_force_capacities(caps))
    term.apply_actions()
    assert len(applied) == 1 and not term.qp_failed.any()
    assert (term.previous_wrench[0, :, 2] > 1.).all()
    assert term.previous_wrench[1, 0].count_nonzero() == 0
    expected = term.solver.solve(state, state, encode_force_capacities(caps),
        torch.ones((2, 2, 4), dtype=torch.bool), term.limits)
    torch.testing.assert_close(applied[0].double(), expected['torque'], atol=2e-5, rtol=0)
    if record_diagnostics:
        assert term.last_prediction['available_vertices'].all()
    else:
        # A previous prediction must not make contact measurements mandatory
        # on the next policy step through the shared wrench-error logger.
        for _ in range(env.cfg.decimation-1):
            term.apply_actions()
        term.process_actions(encode_force_capacities(caps))
        term.apply_actions()
        assert len(applied) == env.cfg.decimation+1 and not term.qp_failed.any()
        torch.testing.assert_close(applied[-1], applied[0], atol=0, rtol=0)


def test_manual_capacities_round_trip():
    caps = torch.tensor([[0., 1.], [.01, .75], [.5, .5]], dtype=torch.float64)
    logits = encode_force_capacities(caps)
    assert torch.isfinite(logits).all()
    assert logits[0].tolist() == [-3., 3.]
    torch.testing.assert_close(decode_force_actions(logits), caps)
    assert decode_force_actions(logits)[0].tolist() == [0., 1.]


def test_force_mapping_preserves_interior_variation():
    logits = torch.tensor([[-2., 1.], [-1., 2.]], requires_grad=True)
    fractions = decode_force_actions(logits)
    assert ((fractions > 0) & (fractions < 1)).all()
    assert (fractions[1] > fractions[0]).all()
    fractions.sum().backward()
    assert (logits.grad > 0).all()
