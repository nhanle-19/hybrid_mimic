"""Regression checks for physically supported, stationary motion endings."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('mujoco')
pytest.importorskip('pxr')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_static_motion_endings import EndingBuilder, slowing_phase
from dsms_t1_model import ROOT, build_model, load_reference
from mirror_training_motions import mirror_motion


@pytest.fixture(scope='module')
def editor(tmp_path_factory):
    return EndingBuilder(build_model(tmp_path_factory.mktemp('static_end_model')))


def test_deceleration_is_monotone_and_matches_initial_velocity():
    u = np.linspace(0., 1., 10001)
    phase = slowing_phase(u)
    velocity = np.gradient(phase, u)
    assert phase[0] == 0. and phase[-1] == 1.
    assert np.diff(phase).min() >= 0.
    assert abs(velocity[0]-2.) < 1e-6
    assert abs(velocity[-1]) < 1e-6


def test_static_support_rejects_floating_pose(editor):
    _, state, _ = load_reference(ROOT/'retargeted_motion/training_list/01_walk.npz', editor.model)
    q = state[-1, :editor.model.nq].copy()
    assert editor.support(q)['passed']
    q[2] += .05
    result = editor.support(q)
    assert not result['passed'] and result['contacts'] == 0


def test_nonstatic_front_kick_gets_continuous_supported_ending(editor, tmp_path):
    template, state, fps = load_reference(ROOT/'retargeted_motion/training_list/09_front_kick.npz', editor.model)
    original = state[:, :editor.model.nq]
    target, check = editor.settled_pose(original[-1])
    assert check['passed']
    q, frames, recipe = editor.make_tail(original, len(original)-1, target, fps)
    start = recipe['edited_from_frame']
    np.testing.assert_array_equal(q[:start], original[:start])
    np.testing.assert_array_equal(frames[:start], np.arange(start))
    assert np.diff(frames).min() >= -1e-10
    assert len(q) > len(original)
    result = editor.validate_tail(q, start, fps)
    assert result['passed']
    assert result['planted_settling_frames'] == result['planted_settling_feasible_frames']
    # A moving last pose must not sneak through by relabeling it as a hold.
    altered = q.copy()
    altered[-15:, 0] += np.linspace(0., .01, 15)
    assert not editor.validate_tail(altered, start, fps)['passed']
    path = tmp_path/'reference.npz'
    editor.export(path, q, template, fps, frames)
    with np.load(path, allow_pickle=False) as z:
        output = dict(z)
    for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
        assert np.max(np.abs(output[key][-50:])) < 1e-9
    restored = mirror_motion(mirror_motion(output))
    for key in output:
        np.testing.assert_array_equal(restored[key], output[key])
    # Slow the inherited fast kick while retaining a real one-second hold.
    slow_q, slow_frames, slow_recipe = editor.time_scale_active(q, frames, recipe, 2., fps)
    assert len(slow_q)-51 == 2*(len(q)-51)
    assert slow_frames[-1] == frames[-1]
    assert np.max(np.abs(slow_q[-50:]-slow_q[-1])) == 0.
    speed_fraction = np.max(np.abs(editor.velocities(slow_q, fps)[:, 6:])/editor.speeds)
    assert speed_fraction < .9
    assert editor.validate_tail(slow_q, slow_recipe['edited_from_frame'], fps)['passed']
