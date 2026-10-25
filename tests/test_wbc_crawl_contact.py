"""Crawl force gating, per-point Jacobians, and manually reviewed transitions."""
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from crawl_force_schedule import load_manual_schedule
from check_crawl_force_qp import controller_config, motion_state, torque_limits
from wbc_crawl_contact import CRAWL_REGIONS, CrawlContactGeometry, WBCCrawlForceQP
from wbc_acc_torch_model import WBCACCTorchModel, mv
from wbc_acc_model import WBCACCModel
from wbc_acc_gpu_control import select_state


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def crawl():
    specification, motion, annotations, schedule = load_manual_schedule()
    model = WBCACCTorchModel()
    cfg = controller_config(specification)
    geometry = CrawlContactGeometry(ROOT/specification['geometry'])
    state = motion_state(model, motion)
    limits = model.tensor(torque_limits(model.joint_names))
    return model, cfg, geometry, state, limits, schedule, annotations


def test_eight_caps_and_zero_gates_with_batched_osqp_parity(crawl):
    model, cfg, geometry, all_states, limit, schedule, _ = crawl
    frames = torch.tensor([0, 70, 104, 496, 542])
    state = select_state(all_states, frames)
    labels = torch.as_tensor(schedule['contact_labels'])[frames]
    # The manual policy encodes OFF in the raw output itself.
    actions = model.tensor(schedule['actions'])[frames].clone()
    actions[0, 0] = -3.  # Exact zero despite a true label and available geometry.
    actions[1, 0] = -2.  # A low positive cap must bind independently of the other seven.
    limits = limit.expand(len(frames), -1)
    result = WBCCrawlForceQP(cfg, geometry).solve(state, state, actions, limits)
    expected = WBCCrawlForceQP(cfg, geometry, backend='osqp').solve(state, state, actions, limits)
    assert not result['failed'].any() and not expected['failed'].any()
    assert result['wrenches'].shape == (5, 8, 6)
    assert result['rho'].shape == (5, 128)
    assert result['point_forces'][~labels].count_nonzero() == 0
    assert result['wrenches'][0, 0].count_nonzero() == 0
    assert result['normal_force_limits'][~labels].count_nonzero() == 0
    forces = result['point_forces']
    assert (forces[..., 2] >= -2e-5).all()
    assert (forces[..., :2].norm(dim=-1) <= .6*forces[..., 2]+2e-5).all()
    assert (forces[..., 2].sum(-1) <= result['normal_force_limits']+2e-5).all()
    assert (result['torque'].abs() <= limits+2e-5).all()
    torch.testing.assert_close(result['torque'], expected['torque'], atol=2e-3, rtol=1e-3)
    inverse = mv(state['M'], result['acceleration'])+state['bias']-result['generalized_contact_force']
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)


@pytest.mark.parametrize('height', [.02, -.04])
def test_positive_outputs_cannot_create_floating_or_deeply_penetrating_support(crawl, height):
    model, cfg, geometry, all_states, limit, _, _ = crawl
    q = all_states['q'][:1].clone()
    q[:, 2] += height
    state = model.state(q, all_states['v'][:1])
    # Only four baseline channels request force; they move outside the valid band.
    actions = model.tensor([[3., 3., 3., 3., -3., -3., -3., -3.]])
    result = WBCCrawlForceQP(cfg, geometry).solve(state, state, actions, limit[None])
    assert not result['failed'].any()
    assert result['point_forces'].count_nonzero() == 0
    assert result['normal_force_limits'].count_nonzero() == 0


def test_material_point_jacobians_match_independent_pinocchio_integration(crawl):
    model, cfg, geometry, all_states, _, _, annotations = crawl
    state = select_state(all_states, slice(70, 71))
    points, available = geometry.at(state, cfg)
    pin_model = WBCACCModel()
    assert pin_model.joint_names == model.joint_names
    step = 1e-6
    velocity = np.linspace(-.2, .3, model.nv)
    plus_q = pin.integrate(pin_model.model, state['q'][0].numpy(), step*velocity)
    minus_q = pin.integrate(pin_model.model, state['q'][0].numpy(), -step*velocity)
    plus, minus = (pin_model.state(q, np.zeros(model.nv)) for q in (plus_q, minus_q))
    for region in (0, 2, 4):  # hand, knee and forearm, with their own body attachments
        assert available[0, region, 0]
        point = int(points['point_ids'][0, region, 0])
        body = str(annotations['body_names'][annotations['point_body_index'][point]])
        local = annotations['point_position_body'][point]
        positions = [s['frames'][body]['position']+s['frames'][body]['rotation']@local for s in (plus, minus)]
        finite_difference = (positions[0]-positions[1])/(2*step)
        predicted = points['J'][0, region, 0].numpy()@velocity
        np.testing.assert_allclose(predicted, finite_difference, atol=1e-7, rtol=1e-6)


def test_smoothed_reference_contact_intervals(crawl):
    _, _, _, _, _, schedule, _ = crawl
    labels = schedule['contact_labels']
    assert labels.shape == (599, 8)
    np.testing.assert_array_equal(labels.sum(0), [599, 599, 599, 599, 11, 0, 16, 0])
    assert not labels[67, 4] and labels[68:73, 4].all() and not labels[73, 4]
    assert labels[496:501, 6].all() and not labels[501, 6]
    assert np.all(schedule['actions'][labels] == 3.)
    assert np.all(schedule['actions'][~labels] == -3.)
    assert np.all(schedule['normal_force_limits'][~labels] == 0.)


@pytest.mark.parametrize('change,match', [
    ('hash', 'stale motion'), ('bridge_gap', 'disagree with geometric'),
    ('overlap', 'overlapping'), ('order', 'region order'), ('normal', 'floor normal')])
def test_reject_stale_or_wrong_manual_labels(tmp_path, change, match):
    specification = json.loads((ROOT/'configs/crawl_force_schedule.json').read_text())
    if change == 'hash':
        specification['motion_sha256'] = 'wrong'
    elif change == 'bridge_gap':
        specification['contact_intervals']['right_forearm'] = [[103, 106], [115, 116]]
    elif change == 'overlap':
        specification['contact_intervals']['left_hand'] = [[0, 400], [400, 598]]
    elif change == 'order':
        specification['region_order'].reverse()
    elif change == 'normal':
        specification['normal_world'] = [1., 0., 0.]
    path = tmp_path/'bad_schedule.json'
    path.write_text(json.dumps(specification))
    with pytest.raises(ValueError, match=match):
        load_manual_schedule(path)


def test_wrong_nn_output_count_is_rejected(crawl):
    _, cfg, geometry, states, limits, _, _ = crawl
    state = select_state(states, slice(0, 1))
    with pytest.raises(ValueError, match='8 finite'):
        WBCCrawlForceQP(cfg, geometry).solve(state, state, torch.zeros(1, 7), limits[None])
