"""Learned force caps, absent foot objectives, and the two-output action interface."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, action_modules, DEVICES
from wbc_acc_contact_policy import available_vertices
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_force_gpu_control import WBCForceGPUQP, decode_force_actions


@pytest.mark.parametrize('device', DEVICES)
def test_force_limits_and_inverse_dynamics(cfg, device):
    state = standing_state(WBCACCTorchModel(device), 5)
    actions = torch.tensor([[1., 1.], [.01, 1.], [0., 1.], [0., 0.], [1., 1.]],
                           dtype=torch.float64, device=device)
    limits = torch.full((5, 23), 60., dtype=torch.float64, device=device)
    masks = available_vertices(state, cfg)
    masks[4, 0] = False  # A large learned cap cannot create unavailable support.
    result = WBCForceGPUQP(cfg).solve(state, state, actions, masks, limits)
    assert not result['failed'].any()
    force = result['wrenches'][:, :, :3]
    assert (force[:, :, 2] >= -2e-5).all()
    assert (force[:, :, 2] <= actions*600.+2e-5).all()
    assert (force[:, :, :2].norm(dim=-1) <= .6*force[:, :, 2]+2e-5).all()
    assert result['wrenches'][2, 0].count_nonzero() == 0
    assert result['wrenches'][3].count_nonzero() == 0
    assert result['wrenches'][4, 0].count_nonzero() == 0
    # The cap changes the optimum instead of being an unused action.
    assert force[0, 0, 2] > force[1, 0, 2]+1.
    assert abs(force[1, 0, 2].item()-6.) < .01
    assert (result['torque'].abs() <= limits+2e-5).all()
    generalized = sum(mv(state['frames'][foot]['J'].transpose(-1, -2), result['wrenches'][:, i])
                      for i, foot in enumerate(('left_foot_link', 'right_foot_link')))
    inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)
    expected = WBCForceGPUQP(cfg, backend='osqp').solve(state, state, actions, masks, limits)
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)
    assert 'contact_weights' not in result


def test_no_reference_foot_acceleration_objective(cfg):
    state = standing_state(WBCACCTorchModel())
    actions = torch.ones((1, 2), dtype=torch.float64)
    limits = torch.full((1, 23), 60., dtype=torch.float64)
    masks = available_vertices(state, cfg)
    solver = WBCForceGPUQP(cfg)
    expected = solver.solve(state, state, actions, masks, limits)
    reference = copy.deepcopy(state)
    # Foot reference poses/velocities/accelerations are unnecessary in FORCE.
    del reference['frames']['left_foot_link']
    del reference['frames']['right_foot_link']
    result = solver.solve(state, reference, actions, masks, limits)
    assert not result['failed'].any()
    torch.testing.assert_close(result['torque'], expected['torque'], atol=1e-10, rtol=0)


@pytest.mark.parametrize('actions', [torch.zeros(2, 14), torch.zeros(2), torch.tensor([[float('nan'), 0.]])])
def test_reject_invalid_actions(actions):
    with pytest.raises(ValueError, match='two finite'):
        decode_force_actions(actions)


def test_force_mapping_and_disabled_swing(cfg):
    torch.testing.assert_close(decode_force_actions(torch.tensor([[-2., 2.], [.2, .7]])),
                               torch.tensor([[0., 1.], [.2, .7]]))
    cfg.swing_weight = 1.
    with pytest.raises(ValueError, match='foot-acceleration objectives are disabled'):
        WBCForceGPUQP(cfg)


def test_force_action_initialization_and_reset(action_modules, monkeypatch):
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
    torch.testing.assert_close(modules['rewards'].mapped_force_action_rate(env), torch.tensor([0., .125]))
