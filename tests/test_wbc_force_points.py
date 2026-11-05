"""Eight independent corner caps: exact release, physical bounds, and solver parity."""
import pytest
import torch

from test_wbc_acc_gpu import cfg, threads, standing_state, DEVICES
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_gpu_control import select_state, FEET
from wbc_force_gpu_control import WBCCornerForceGPUQP, encode_force_capacities


@pytest.mark.parametrize('device', DEVICES)
def test_all_corner_cap_patterns_and_independent_force_bounds(cfg, device):
    cfg.point_force_max_body_weights = 1.
    # These legacy foot maxima must not become additional total-foot bounds.
    cfg.contact_force_max = (1., 1.)
    state = standing_state(WBCACCTorchModel(device), 256)
    bits = ((torch.arange(256, device=device)[:, None] >> torch.arange(8, device=device)) & 1).bool()
    fractions = bits.double()
    actions = encode_force_capacities(fractions, 8)
    masks = torch.ones((256, 2, 4), dtype=torch.bool, device=device)
    limits = torch.full((256, 23), 60., dtype=torch.float64, device=device)
    solver = WBCCornerForceGPUQP(cfg)
    result = solver.solve(state, state, actions, masks, limits)
    assert not result['failed'].any()
    body_weight = state['mass']*(-state['gravity'][2])
    torch.testing.assert_close(result['normal_force_limits'], fractions.reshape(-1, 2, 4)*body_weight)
    torch.testing.assert_close(result['available'], bits.reshape(-1, 2, 4))
    point_forces = result['point_forces'].reshape(256, 8, 3)
    assert point_forces[~bits].count_nonzero() == 0
    assert (point_forces[..., 2] >= -2e-5).all()
    assert (point_forces[..., 2] <= fractions*body_weight+2e-5).all()
    assert (point_forces[..., :2].norm(dim=-1) <= .6*point_forces[..., 2]+2e-5).all()
    assert (result['wrenches'][-1, :, 2] > 1.).all()
    assert (result['torque'].abs() <= limits+2e-5).all()
    generalized = sum(mv(state['frames'][foot]['J'].transpose(-1, -2), result['wrenches'][:, i])
                      for i, foot in enumerate(FEET))
    inverse = mv(state['M'], result['acceleration'])+state['bias']-generalized
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)
    indices = torch.tensor([0, 1, 2, 16, 85, 170, 255], device=device)
    subset = select_state(state, indices)
    expected = WBCCornerForceGPUQP(cfg, backend='osqp').solve(subset, subset,
        actions[indices], masks[indices], limits[indices])
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'][indices], expected['acceleration'], atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize('device', DEVICES)
def test_unequal_corner_caps_cannot_borrow_capacity_and_do_not_gate_by_height(cfg, device):
    cfg.point_force_max_body_weights = 1.
    model = WBCACCTorchModel(device)
    state = standing_state(model, 2)
    q = state['q'].clone()
    q[1, 2] += .3
    state = model.state(q, state['v'])
    fractions = q.new_tensor([[0., .001, .01, .8, .9, .02, .002, 0.]]).expand(2, -1)
    result = WBCCornerForceGPUQP(cfg).solve(state, state, encode_force_capacities(fractions, 8),
        torch.ones((2, 2, 4), dtype=torch.bool, device=device), q.new_full((2, 23), 60.))
    assert not result['failed'].any()
    fz = result['point_forces'][..., 2].reshape(2, 8)
    maximum = state['mass']*(-state['gravity'][2])
    assert (fz <= fractions*maximum+2e-5).all()
    assert fz[:, [0, 7]].count_nonzero() == 0
    assert fz[:, [1, 6]].max() > .01
    torch.testing.assert_close(result['torque'][1], result['torque'][0], atol=2e-5, rtol=0)
    torch.testing.assert_close(result['point_forces'][1], result['point_forces'][0], atol=2e-5, rtol=0)


def test_corner_policy_rejects_old_two_output_actions(cfg):
    cfg.point_force_max_body_weights = 1.
    with pytest.raises(ValueError, match='8 finite'):
        WBCCornerForceGPUQP(cfg).decode_policy(torch.zeros(1, 2))
