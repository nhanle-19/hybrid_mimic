"""Live-state crawl control and matching simulator collision geometry."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_wbc_crawl_contact import crawl, threads
from wbc_acc_torch_model import rotation_xyzw, mv
from wbc_crawl_control import CrawlForceController, classify_contact_points
from wbc_crawl_contact import CrawlContactGeometry
from wbc_acc_gpu_control import select_state

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_crawl_physics_asset import collider_vertices


def controller(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    return CrawlForceController(model, cfg, geometry, states['q'], states['v'],
        model.tensor(schedule['actions']), float(schedule['fps']), limits[None],
        reference_contact_labels=torch.tensor(schedule['contact_labels']))


def test_reference_frame_labels_do_not_interpolate_across_short_gap(crawl):
    control = controller(crawl)
    # Synthetic one-frame release: the smoothed reference no longer has the
    # old right-forearm chatter, but schedule lookup must preserve short gaps.
    control.labels[104:107, 5] = torch.tensor([True, False, True])
    control.actions = control.actions.clone()
    control.actions[104:107, 5] = control.actions.new_tensor([3., -3., 3.])
    for frame in (104, 105, 106):
        ref, index = control.reference((frame+.75)/control.fps)
        assert index == frame
        assert bool(control.labels[index, 5]) == (frame != 105)
        torch.testing.assert_close(ref['q'][:, 3:7].norm(dim=-1), torch.ones(1, dtype=torch.float64))
    assert control.reference(105/control.fps)[1] == 105


def test_manual_outputs_can_release_within_one_reference_frame(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    actions = model.tensor(schedule['actions']).repeat_interleave(4, dim=0)
    actions[1, 0] = -3.
    control = CrawlForceController(model, cfg, geometry, states['q'], states['v'],
        actions, float(schedule['fps']), limits[None], action_fps=4*float(schedule['fps']))
    state = select_state(states, slice(0, 1))
    released = control.compute(state, .25/control.fps)
    assert released['reference_frame'] == 0 and released['policy_sample'] == 1
    assert released['actual_available'][0, 0].any()
    assert released['point_forces'][0, 0].count_nonzero() == 0
    restored = control.compute(state, .5/control.fps)
    assert restored['reference_frame'] == 0 and restored['policy_sample'] == 2
    torch.testing.assert_close(restored['raw_actions'], actions[2:3])


def test_actual_state_controls_availability_and_feedback(crawl):
    model, _, _, states, _, _, _ = crawl
    control = controller(crawl)
    live_q = states['q'][:1].clone()
    live_q[:, 2] += .02
    actual = model.state(live_q, states['v'][:1])
    result = control.compute(actual, 0.)
    assert not result['failed'].any()
    assert result['contact_labels'][0, :4].all()
    assert result['point_forces'].count_nonzero() == 0
    assert result['normal_force_limits'].count_nonzero() == 0
    assert abs(float(result['root_error'][0])-.02) < 1e-12
    torch.testing.assert_close(result['reference_q'], states['q'][:1])


def test_joint_feedback_uses_live_error(crawl):
    model, _, _, states, _, _, _ = crawl
    control = controller(crawl)
    actual_q = states['q'][:1].clone()
    actual_q[:, 7] += .03  # Head joint, without changing support geometry.
    actual = model.state(actual_q, states['v'][:1])
    result = control.compute(actual, 0.)
    baseline = control.compute(select_state(states, slice(0, 1)), 0.)
    assert not result['failed'].any()
    assert float(result['joint_rmse'][0]) > 0
    assert float(result['torque'][0, 0]-baseline['torque'][0, 0]) < -1e-4


def test_optional_separation_gate_removes_rising_support_but_keeps_stationary_support(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    cfg = SimpleNamespace(**vars(cfg), contact_separation_velocity=.005)
    control = CrawlForceController(model, cfg, geometry, states['q'], states['v'],
        model.tensor(schedule['actions']), float(schedule['fps']), limits[None],
        reference_contact_labels=torch.tensor(schedule['contact_labels']))
    q = states['q'][:1].clone()
    v = torch.zeros_like(states['v'][:1])
    still = control.compute(model.state(q, v), 0.)
    assert not still['failed'].any()
    assert still['available'][0, :4].any(-1).all()
    v[:, :3] = mv(rotation_xyzw(q[:, 3:7]).transpose(-1, -2), model.tensor([[0., 0., .02]]))
    rising = control.compute(model.state(q, v), 0.)
    assert not rising['failed'].any()
    assert rising['near_floor_available'][0, :4].any(-1).all()
    assert not rising['available'].any()
    assert rising['point_forces'].count_nonzero() == 0


def test_diagnostic_hold_keeps_pose_and_labels_but_zeroes_velocity(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    control = CrawlForceController(model, cfg, geometry, states['q'], states['v'],
        model.tensor(schedule['actions']), float(schedule['fps']),
        limits[None], friction=.2, hold_frame=105, reference_contact_labels=torch.tensor(schedule['contact_labels']))
    for time in (0., 2., 10.):
        reference, index = control.reference(time)
        assert index == 105
        assert not control.labels[index, 5]
        torch.testing.assert_close(reference['q'], states['q'][105:106])
        assert reference['v'].count_nonzero() == 0
    assert control.solver.friction == .2


def test_time_scaling_preserves_label_progress_and_scales_velocity(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    full = controller(crawl)
    slow = CrawlForceController(model, cfg, geometry, states['q'], states['v']*.5,
        model.tensor(schedule['actions']), float(schedule['fps'])*.5, limits[None],
        reference_contact_labels=torch.tensor(schedule['contact_labels']))
    for frame in (70, 104, 105, 106):
        original, index = full.reference((frame+.25)/full.fps)
        reference, slow_index = slow.reference(2*(frame+.25)/full.fps)
        assert index == slow_index
        torch.testing.assert_close(reference['q'], original['q'])
        torch.testing.assert_close(reference['v'], original['v']*.5)


def test_collider_simplification_keeps_all_reference_floor_extrema(crawl):
    *_, annotations = crawl
    body = annotations['body_names'].tolist().index('left_hand_link')
    region = annotations['surface_names'].tolist().index('left_hand')
    mask = (annotations['point_body_index'] == body) & (annotations['point_surface_index'] == region)
    points = annotations['point_position_body'][mask]
    normals = annotations['body_rotation_w'][:, body, 2, :]
    chosen = collider_vertices(points, normals)
    assert len(chosen) <= 64
    np.testing.assert_allclose((points[chosen]@normals.T).min(0), (points@normals.T).min(0), atol=1e-12, rtol=0)


def test_authored_collision_geometry_matches_labels(crawl):
    model, cfg, _, states, _, schedule, _ = crawl
    path = ROOT/'eval_data/reference_crawl_forward/physics_asset/collision_geometry.npz'
    if not path.exists():
        pytest.skip('Run build_crawl_physics_asset.py to validate the generated collision artifact')
    geometry = CrawlContactGeometry(path)
    indices = [0, 70, 104, 105, 106, 453, 497, 540, 598]
    _, available = geometry.at(select_state(states, indices), cfg)
    np.testing.assert_array_equal(available.any(-1), schedule['contact_labels'][indices])


def test_measured_contacts_classified_by_hull_not_scheduled_label():
    planes = np.array([[1,0,0,-1],[-1,0,0,-1],[0,1,0,-1],[0,-1,0,0],[0,0,1,-1],[0,0,-1,0]], float)
    shifted = planes.copy()
    shifted[:, 3] -= shifted[:, 1]
    entries = [dict(region='forearm', planes_body=planes.tolist()), dict(region='hand', planes_body=shifted.tolist())]
    points = torch.tensor([[0., .25, -.0001], [0., 1.8, .0001]], dtype=torch.float64)
    assert classify_contact_points(points, entries, ['forearm', 'hand']).tolist() == [0, 1]


def test_live_physx_state_has_valid_solution_at_runtime_tolerance(crawl):
    model, _, _, references, limits, schedule, _ = crawl
    path = ROOT/'tests/fixtures/crawl_force_live_qp.npz'
    with np.load(path, allow_pickle=False) as data:
        fixture = dict(data)
    cfg = SimpleNamespace(**json.loads(str(fixture['controller_config_json'])))
    control = CrawlForceController(model, cfg, CrawlContactGeometry(path), references['q'], references['v'],
        model.tensor(schedule['actions']), float(schedule['fps']), limits[None],
        reference_contact_labels=torch.tensor(schedule['contact_labels']))
    actual = model.state(model.tensor(fixture['q']), model.tensor(fixture['v']))
    result = control.compute(actual, float(fixture['time_s']))
    assert not result['failed'].any()
    assert float(result['residual'][0]) < 2e-5
    assert result['point_forces'][~result['contact_labels']].count_nonzero() == 0
    assert (result['torque'].abs() <= limits+2e-5).all()


def test_reference_annotations_cannot_change_manual_output_torques(crawl):
    model, cfg, geometry, states, limits, schedule, _ = crawl
    actions = model.tensor(schedule['actions'])
    control = CrawlForceController(model, cfg, geometry, states['q'], states['v'],
        actions, float(schedule['fps']), limits[None])
    actual = select_state(states, slice(0, 1))
    unannotated = control.compute(actual, 0.)
    assert not unannotated['failed'].any()
    assert 'contact_labels' not in unannotated
    assert unannotated['point_forces'].abs().max() > 1.
    for annotation in (torch.zeros(actions.shape, dtype=torch.bool), torch.ones(actions.shape, dtype=torch.bool)):
        control.labels = annotation
        annotated = control.compute(actual, 0.)
        torch.testing.assert_close(annotated['torque'], unannotated['torque'], atol=0, rtol=0)
        torch.testing.assert_close(annotated['point_forces'], unannotated['point_forces'], atol=0, rtol=0)
    control.actions = torch.full_like(actions, -3.)
    disabled = control.compute(actual, 0.)
    assert not disabled['failed'].any()
    assert disabled['contact_labels'].all()
    assert disabled['point_forces'].count_nonzero() == 0
