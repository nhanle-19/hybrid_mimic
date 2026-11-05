"""Guard contact-evaluation geometry and checkpoint selection semantics."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_gain_override_is_forwarded_and_old_gain_results_are_not_reused(tmp_path, monkeypatch):
    suite = load('eval_force_motion_suite')
    checkpoint = tmp_path/'model_4999.pt'
    checkpoint.write_bytes(b'checkpoint')
    motions = tmp_path/'motions'
    motions.mkdir()
    (motions/'motion.npz').write_bytes(b'motion')
    digest = hashlib.sha256(b'motion').hexdigest()
    (motions/'manifest.json').write_text(json.dumps({'motions': [{'file': 'motion.npz', 'sha256': digest}]}))
    output = tmp_path/'results'
    arguments = ['suite', '--checkpoint', str(checkpoint), '--motions', str(motions),
                 '--output', str(output), '--posture_kp', '3200', '--posture_kd', '80']
    monkeypatch.setattr(sys, 'argv', arguments)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if '--checkpoint' in command:
            directory = output/'motion'
            (directory/'diagnostics.npz').touch()
            (directory/'run.json').write_text(json.dumps({
                'checkpoint_sha256': hashlib.sha256(b'checkpoint').hexdigest(), 'motion_sha256': digest,
                'qp_contact_model': 'sole_midpoint',
                'controller_overrides': {'posture_position_gain': 3200., 'posture_velocity_gain': 80.}}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, 'run', run)
    suite.main()
    assert calls[0][calls[0].index('--posture_kp')+1] == '3200.0'
    assert calls[0][calls[0].index('--posture_kd')+1] == '80.0'
    assert json.loads((output/'suite.json').read_text())['controller_overrides'] == {
        'posture_position_gain': 3200., 'posture_velocity_gain': 80.}
    assert json.loads((output/'suite.json').read_text())['qp_contact_model'] == 'sole_corners'
    calls.clear()
    monkeypatch.setattr(sys, 'argv', arguments[:arguments.index('--posture_kp')]+['--skip_existing'])
    with pytest.raises(ValueError, match='different controller overrides'):
        suite.main()
    assert not calls
    monkeypatch.setattr(sys, 'argv', arguments+['--skip_existing'])
    with pytest.raises(ValueError, match='different QP contact model'):
        suite.main()
    assert not calls


@pytest.mark.parametrize('gain', ['nan', 'inf', '-1', '0'])
def test_invalid_gain_rejected_before_starting_evaluation(tmp_path, monkeypatch, gain):
    suite = load('eval_force_motion_suite')
    monkeypatch.setattr(sys, 'argv', ['suite', '--output', str(tmp_path), '--posture_kp', gain])
    with pytest.raises(SystemExit):
        suite.main()


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
