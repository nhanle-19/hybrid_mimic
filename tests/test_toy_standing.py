"""Verify the toy reference's real collider geometry without Isaac Sim."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'source/whole_body_tracking/whole_body_tracking/utils'))
from toy_standing import heel_up_pose
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
