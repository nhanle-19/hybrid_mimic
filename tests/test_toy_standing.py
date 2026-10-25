"""Verify the toy reference's real collider geometry without Isaac Sim."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'source/whole_body_tracking/whole_body_tracking/utils'))
from toy_standing import heel_up_pose, sway_reference, slide_reference, rotation_reference
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_contact_policy import sole_vertices, available_vertices, FEET


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('angle', [0., 15., 30.])
def test_heel_stays_grounded_and_only_low_edge_supports(side, angle):
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    # Deliberately reverse simulator order to exercise the name mapping.
    names = model.joint_names[::-1]
    points = sole_vertices()
    joints, q, state = heel_up_pose(model, points, torch.zeros(1, len(names)), names, side, angle)
    torch.testing.assert_close(joints.double(), q[:, [7+model.joint_names.index(n) for n in names]],
                               atol=1e-7, rtol=1e-6)
    tilted = 0 if side == 'left' else 1
    for i, foot in enumerate(FEET):
        frame = state['frames'][foot]
        world = points[i] @ frame['rotation'][0].T + frame['position'][0]
        assert abs(float(world[:, 2].min())) < 1e-6
        pitch = torch.rad2deg(torch.atan2(frame['rotation'][0, 2, 0], frame['rotation'][0, 0, 0]))
        assert float(pitch) == pytest.approx(angle if i == tilted else 0., abs=1e-6)
        if i == tilted and angle:
            assert float(world[points[i, :, 0].argmax(), 2]) > .01
        else:
            assert float(world[:, 2].abs().max()) < 1e-6
    cfg = SimpleNamespace(ground_height=0., contact_gap_tolerance=.01,
                          max_contact_penetration=.01, contact_patch_height_tolerance=.0005)
    masks = available_vertices(state, cfg)
    assert int(masks[0, 1-tilted].sum()) == 4
    assert int(masks[0, tilted].sum()) == (2 if angle else 4)


def test_toy_patch_tolerates_small_flat_foot_tilt_but_excludes_raised_toes():
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    points = sole_vertices()
    _, _, state = heel_up_pose(model, points, torch.zeros(1, len(model.joint_names)), model.joint_names)
    frame = state['frames'][FEET[1]]
    angle = torch.tensor(.001, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    rotation = torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]], dtype=torch.float64)
    frame['rotation'] = (rotation @ frame['rotation'][0])[None]
    world = points[1] @ frame['rotation'][0].T + frame['position'][0]
    frame['position'][0, 2] -= world[:, 2].min()
    cfg = SimpleNamespace(ground_height=0., contact_gap_tolerance=.001,
                          max_contact_penetration=.03, contact_patch_height_tolerance=.0005)
    mask = available_vertices(state, cfg)
    assert mask[0, 0].tolist() == [True, True, False, False]
    assert mask[0, 1].all()
    # Being near the lowest corner never overrides the ground-height gate.
    frame['position'][0, 2] += .002
    assert not available_vertices(state, cfg)[0, 1].any()


@pytest.mark.parametrize('side,angle', [('left', 15.), ('right', 15.), ('left', 0.)])
@pytest.mark.parametrize('stance_widen_m', [0., .02])
def test_sway_moves_com_and_leans_torso_without_moving_feet(side, angle, stance_widen_m):
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    names, bodies = model.joint_names[::-1], model.body_names[::-1]
    default = torch.zeros((1, len(names)), dtype=torch.float64)
    values = sway_reference(model, sole_vertices(), default, names, bodies,
        np.tile([-3., 3.], (len(names), 1)), side=side, tilt_deg=angle, seconds=6., fps=20.,
        stance_widen_m=stance_widen_m)
    _, _, nominal = heel_up_pose(model, sole_vertices(), default, names, side, angle)
    root = bodies.index('Trunk')
    q = torch.tensor(np.concatenate((values['body_pos_w'][:, root],
        values['body_quat_w'][:, root][:, [1, 2, 3, 0]],
        values['joint_pos'][:, [names.index(n) for n in model.joint_names]]), -1))
    state = model.state(q, torch.zeros((len(q), model.nv), dtype=torch.float64))
    np.testing.assert_allclose(state['com'], values['com_pos_w'], atol=1e-8)
    np.testing.assert_allclose(state['com'][0], nominal['com'][0], atol=1e-8)
    assert values['stance_widen_m'] == stance_widen_m
    displacement = values['com_pos_w'][:, 1]-values['com_pos_w'][0, 1]
    assert displacement.max() > .015 and displacement.min() < -.015
    np.testing.assert_allclose(state['com'][:, [0, 2]], state['com'][:1, [0, 2]].expand(len(q), -1), atol=1e-8)
    roll = Rotation.from_quat(values['body_quat_w'][:, root][:, [1, 2, 3, 0]]).as_euler('xyz')[:, 0]
    np.testing.assert_allclose(roll, -np.deg2rad(2.4) * displacement/.020, atol=1e-8)
    cfg = SimpleNamespace(ground_height=0., contact_gap_tolerance=.001,
                          max_contact_penetration=.01, contact_patch_height_tolerance=.0005)
    masks = available_vertices(state, cfg)
    for i, foot in enumerate(FEET):
        frame = state['frames'][foot]
        target_position = nominal['frames'][foot]['position'].clone()
        target_position[:, 1] += (1. if i == 0 else -1.) * stance_widen_m / 2.
        torch.testing.assert_close(frame['position'][:1], target_position, atol=1e-8, rtol=0)
        torch.testing.assert_close(frame['rotation'][:1], nominal['frames'][foot]['rotation'], atol=1e-8, rtol=0)
        torch.testing.assert_close(frame['position'], frame['position'][:1].expand(len(q), -1), atol=1e-8, rtol=0)
        torch.testing.assert_close(frame['rotation'], frame['rotation'][:1].expand(len(q), -1, -1), atol=1e-8, rtol=0)
        expected = 2 if angle and i == (0 if side == 'left' else 1) else 4
        assert masks[:, i].sum(-1).eq(expected).all()
        for key in ('body_lin_vel_w', 'body_ang_vel_w'):
            assert abs(values[key][:, bodies.index(foot)]).max() < 1e-7
    for position, velocity in [('joint_pos', 'joint_vel'), ('body_pos_w', 'body_lin_vel_w')]:
        np.testing.assert_allclose(values[velocity], np.gradient(values[position], .05, axis=0), atol=1e-10)
        np.testing.assert_allclose(values[position][-1], values[position][0], atol=1e-8)
        assert abs(values[velocity][[0, -1]]).max() < 1e-8
    assert abs(values['joint_vel']).max() > .01
    assert abs(values['body_ang_vel_w'][:, root]).max() > .01


@pytest.mark.parametrize('kwargs', [{'com_sway_m': float('nan')}, {'torso_roll_deg': 11.},
                                   {'period_s': 0.}, {'seconds': 1.}, {'stance_widen_m': -.01},
                                   {'stance_widen_m': float('nan')}, {'stance_widen_m': .11}])
def test_invalid_sway_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        sway_reference(None, None, None, None, None, None, **kwargs)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_slide_keeps_both_soles_flat_and_rest_of_pose_fixed(side):
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    names, bodies = model.joint_names[::-1], model.body_names[::-1]
    defaults = torch.full((1, len(names)), .1, dtype=torch.float64)
    points = sole_vertices()
    values = slide_reference(model, points, defaults, names, bodies,
        np.tile([-3., 3.], (len(names), 1)), side=side, seconds=8., fps=20.)
    _, neutral, nominal = heel_up_pose(model, points, defaults, names, side, 0.)
    root = bodies.index('Trunk')
    q = torch.tensor(np.concatenate((values['body_pos_w'][:, root],
        values['body_quat_w'][:, root][:, [1, 2, 3, 0]],
        values['joint_pos'][:, [names.index(n) for n in model.joint_names]]), -1))
    state = model.state(q, torch.zeros((len(q), model.nv), dtype=torch.float64))
    moving_names = [f'{side.title()}_{name}' for name in
                   ('Hip_Pitch', 'Hip_Roll', 'Hip_Yaw', 'Knee_Pitch', 'Ankle_Pitch', 'Ankle_Roll')]
    fixed_ids = list(range(7)) + [7+i for i, name in enumerate(model.joint_names) if name not in moving_names]
    torch.testing.assert_close(q[:, fixed_ids], neutral[:, fixed_ids].expand(len(q), -1), atol=1e-8, rtol=0)
    for i, foot in enumerate(FEET):
        frame, initial = state['frames'][foot], nominal['frames'][foot]
        expected = initial['position'].repeat(len(q), 1)
        if foot == f'{side}_foot_link':
            expected[:, 1] += torch.as_tensor(values['slide_displacement_m'])
            displacement = frame['position'][:, 1]-initial['position'][0, 1]
            assert float(displacement.max()) == pytest.approx(.02, abs=1e-8)
            assert float(displacement.min()) == pytest.approx(-.02, abs=1e-8)
            assert abs(values['body_lin_vel_w'][:, bodies.index(foot), 1]).max() > .02
        else:
            assert abs(values['body_lin_vel_w'][:, bodies.index(foot)]).max() < 1e-8
        torch.testing.assert_close(frame['position'], expected, atol=1e-8, rtol=0)
        torch.testing.assert_close(frame['rotation'], initial['rotation'].expand(len(q), -1, -1), atol=1e-8, rtol=0)
        vertices = points[i] @ frame['rotation'].transpose(-1, -2) + frame['position'][:, None, :]
        assert float(vertices[:, :, 2].abs().max()) < 1e-8
        assert abs(values['body_ang_vel_w'][:, bodies.index(foot)]).max() < 1e-7
    for position, velocity in [('joint_pos', 'joint_vel'), ('body_pos_w', 'body_lin_vel_w')]:
        np.testing.assert_allclose(values[velocity], np.gradient(values[position], .05, axis=0), atol=1e-10)
        np.testing.assert_allclose(values[position][-1], values[position][0], atol=1e-8)
        assert abs(values[velocity][[0, -1]]).max() < 1e-8


@pytest.mark.parametrize('kwargs', [{'slide_amplitude_m': float('nan')}, {'slide_amplitude_m': .06},
    {'slide_amplitude_m': 0.}, {'seconds': 1.}, {'fps': 0.}, {'period_s': 1.}, {'side': 'both'}])
def test_invalid_slide_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        slide_reference(None, None, None, None, None, None, **kwargs)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('toe_pitch_deg', [0., 10.])
def test_rotation_holds_grounded_pivot_with_expected_heel_height(side, toe_pitch_deg):
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    names, bodies = model.joint_names[::-1], model.body_names[::-1]
    points = sole_vertices()
    values = rotation_reference(model, points, torch.zeros(1, len(names)), names, bodies,
        np.tile([-3., 3.], (len(names), 1)), side=side, seconds=8., fps=20., toe_pitch_deg=toe_pitch_deg)
    root = bodies.index('Trunk')
    q = torch.tensor(np.concatenate((values['body_pos_w'][:, root],
        values['body_quat_w'][:, root][:, [1, 2, 3, 0]],
        values['joint_pos'][:, [names.index(n) for n in model.joint_names]]), -1))
    state = model.state(q, torch.zeros((len(q), model.nv), dtype=torch.float64))
    moving = 0 if side == 'left' else 1
    contact = available_vertices(state, SimpleNamespace(ground_height=0., contact_gap_tolerance=.001,
        max_contact_penetration=.03, contact_patch_height_tolerance=.001))
    for i, foot in enumerate(FEET):
        frame = state['frames'][foot]
        vertices = points[i]@frame['rotation'].transpose(-1, -2)+frame['position'][:, None]
        toe = points[i, :, 0] == points[i, :, 0].max()
        pivot = vertices[:, toe].mean(1) if i == moving and toe_pitch_deg else vertices.mean(1)
        torch.testing.assert_close(pivot, pivot[:1].expand_as(pivot), atol=1e-8, rtol=0)
        if i == moving and toe_pitch_deg:
            assert int(toe.sum()) == 2
            assert bool(contact[:, i, toe].all()) and not bool(contact[:, i, ~toe].any())
            assert float(vertices[:, toe, 2].abs().max()) < 1e-8
            expected_heel = float(points[i, :, 0].max()-points[i, :, 0].min())*np.sin(np.deg2rad(toe_pitch_deg))
            torch.testing.assert_close(vertices[:, ~toe, 2], torch.full_like(vertices[:, ~toe, 2], expected_heel), atol=1e-8, rtol=0)
            assert expected_heel > .02
            np.testing.assert_allclose(np.rad2deg(np.arcsin(-frame['rotation'][:, 2, 0])), toe_pitch_deg, atol=1e-7)
        else:
            assert bool(contact[:, i].all())
            assert float(vertices[..., 2].abs().max()) < 1e-8
        yaw = torch.atan2(frame['rotation'][:, 1, 0], frame['rotation'][:, 0, 0])
        target = values['yaw_angle_rad'] if i == moving else np.zeros(len(q))
        np.testing.assert_allclose(yaw, target, atol=1e-8)
        if i == moving:
            assert float(torch.rad2deg(yaw).max()) == pytest.approx(15., abs=1e-7)
            assert float(torch.rad2deg(yaw).min()) == pytest.approx(-15., abs=1e-7)
            np.testing.assert_allclose(pivot[0], values['pivot_position_w'], atol=1e-8)
        else:
            torch.testing.assert_close(frame['position'], frame['position'][:1].expand(len(q), -1), atol=1e-8, rtol=0)
    fixed = list(range(7))+[7+i for i,n in enumerate(model.joint_names) if not n.startswith(side.title()+'_')
                           or not any(j in n for j in ('Hip_', 'Knee_', 'Ankle_'))]
    torch.testing.assert_close(q[:, fixed], q[:1, fixed].expand(len(q), -1), atol=1e-8, rtol=0)
    np.testing.assert_allclose(values['joint_vel'], np.gradient(values['joint_pos'], .05, axis=0), atol=1e-10)
    np.testing.assert_allclose(values['joint_pos'][0], values['joint_pos'][-1], atol=1e-8)
    assert abs(values['joint_vel'][[0, -1]]).max() < 1e-8


@pytest.mark.parametrize('kwargs', [{'yaw_amplitude_deg': 0.}, {'yaw_amplitude_deg': 31.},
    {'yaw_amplitude_deg': float('nan')}, {'side': 'both'}, {'support_shift_m': -.01},
    {'support_roll_deg': 11.}, {'seconds': 1.}, {'period_s': 1.}, {'toe_pitch_deg': -1.},
    {'toe_pitch_deg': 31.}, {'toe_pitch_deg': float('nan')}])
def test_invalid_rotation_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        rotation_reference(None, None, None, None, None, None, **kwargs)
