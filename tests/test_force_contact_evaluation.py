"""Guard contact-evaluation geometry and checkpoint selection semantics."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_newest_checkpoint_uses_run_and_iteration_not_download_time(tmp_path):
    suite = load('eval_force_motion_suite')
    older = tmp_path/'2026-10-29_walk'
    newer = tmp_path/'2026-10-30_full'
    older.mkdir()
    newer.mkdir()
    (older/'model_9999.pt').touch()
    (newer/'model_4999.pt').touch()
    (newer/'model_500.pt').touch()  # This file is copied last.
    assert suite.newest_checkpoint(tmp_path) == newer/'model_4999.pt'


def test_contact_geometry_retains_stationary_toe_during_rotation():
    module = load('analyze_force_contacts')
    # Toe at x=0.1 stays fixed: origin velocity cancels omega cross r.
    vertices = np.array([[[.1, 0., -.02], [-.1, 0., -.02]]])
    position = np.array([[[0., 0., .02]]])
    rotation = np.eye(3)[None, None]
    velocity = np.array([[[0., -.1, 0., 0., 0., 1.]]])
    clearance, points, speed = module.geometry(position, rotation, velocity, vertices)
    np.testing.assert_allclose(clearance, 0.)
    np.testing.assert_allclose(speed[0, 0, 0], 0., atol=1e-12)
    np.testing.assert_allclose(speed[0, 0, 1], [0., -.2, 0.], atol=1e-12)
    np.testing.assert_allclose(points[0, 0, :, 2], 0.)


def test_swing_events_keep_prefix_suffix_and_empty_denominators():
    module = load('analyze_force_contacts')
    assert module.bouts(np.array([True, True, False, True])) == [(0, 2), (3, 4)]
    assert module.bouts(np.zeros(4, dtype=bool)) == []
    assert module.percent(np.array([True, False]), np.array([False, False])) is None
    assert module.percent(np.array([True, False]), np.array([True, True])) == 50.
