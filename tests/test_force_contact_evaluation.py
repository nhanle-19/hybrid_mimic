"""Guard contact-evaluation geometry and checkpoint selection semantics."""
import ast
import hashlib
import importlib.util
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
                'task': 'Tracking-WBC-FORCE-T1-v0',
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


@pytest.mark.parametrize('task,experiment', [
    (None, 'force'),
    ('Tracking-WBC-FORCE-T1-v0', 'force'),
    ('Tracking-WBC-FORCE-LEGACY-T1-v0', 't1_wbc_force'),
    ('Tracking-WBC-FORCE2-T1-v0', 't1_wbc_force2'),
    ('Tracking-WBC-FORCE2-DR-T1-v0', 't1_wbc_force2_dr'),
])
@pytest.mark.parametrize('history', [False, True])
def test_task_selects_its_checkpoint_and_checks_archived_results(tmp_path, monkeypatch, task, experiment, history):
    suite = load('eval_force_motion_suite')
    monkeypatch.setattr(suite, 'ROOT', tmp_path)
    checkpoint = tmp_path/'logs'/('history' if history else 'rsl_rl')/experiment/'2026-11-14/model_6999.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'force2')
    # A newer run in the other tree must not override the requested scope.
    other = tmp_path/'logs'/('rsl_rl' if history else 'history')/experiment/'2026-11-20/model_9999.pt'
    other.parent.mkdir(parents=True)
    other.write_bytes(b'different checkpoint')
    motions = tmp_path/'motions'
    motions.mkdir()
    (motions/'motion.npz').write_bytes(b'motion')
    motion_hash = hashlib.sha256(b'motion').hexdigest()
    (motions/'manifest.json').write_text(json.dumps({
        'motions': [{'file': 'motion.npz', 'sha256': motion_hash}]}))
    output = tmp_path/'results'
    arguments = ['suite', '--motions', str(motions), '--output', str(output)]
    if history:
        arguments.append('--history')
    if task is not None:
        arguments.extend(['--task', task])
    task = task or 'Tracking-WBC-FORCE-T1-v0'
    monkeypatch.setattr(sys, 'argv', arguments)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if '--checkpoint' in command:
            (output/'motion/diagnostics.npz').touch()
            (output/'motion/run.json').write_text(json.dumps({
                'task': task, 'checkpoint_sha256': hashlib.sha256(b'force2').hexdigest(),
                'motion_sha256': motion_hash, 'qp_contact_model': 'sole_corners',
                'force_cap_scope': 'per_sole_vertex'}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, 'run', run)
    suite.main()
    assert calls[0][calls[0].index('--task')+1] == task
    assert calls[0][calls[0].index('--checkpoint')+1] == str(checkpoint)
    assert json.loads((output/'suite.json').read_text())['task'] == task
    monkeypatch.setattr(sys, 'argv', arguments+['--skip_existing'])
    calls.clear()
    suite.main()
    assert len(calls) == 1 and '--checkpoint' not in calls[0]
    metadata = output/'motion/run.json'
    saved = json.loads(metadata.read_text())
    saved.pop('task')  # Historical FORCE evaluations did not record the task.
    metadata.write_text(json.dumps(saved))
    calls.clear()
    if task == 'Tracking-WBC-FORCE-LEGACY-T1-v0':
        suite.main()
        assert len(calls) == 1 and '--checkpoint' not in calls[0]
    else:
        with pytest.raises(ValueError, match='different task'):
            suite.main()
        assert not calls


def test_default_evaluation_does_not_silently_select_history(tmp_path, monkeypatch):
    suite = load('eval_force_motion_suite')
    monkeypatch.setattr(suite, 'ROOT', tmp_path)
    checkpoint = tmp_path/'logs/history/force/2026-11-14/model_6999.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    monkeypatch.setattr(sys, 'argv', ['suite', '--output', str(tmp_path/'evaluation')])
    with pytest.raises(ValueError, match='No checkpoints.*logs/rsl_rl/force'):
        suite.main()


def test_nominal_evaluation_removes_all_dr_without_relying_on_task_name():
    # Load the helper without launching Isaac Sim at module import.
    tree = ast.parse((ROOT/'scripts/rsl_rl/eval_force_contacts.py').read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'configure_nominal_evaluation')
    nominal = SimpleNamespace(events=object(), scene=SimpleNamespace(robot=object(), terrain=object()))
    lookups = []

    def load_config(task, key):
        lookups.append((task, key))
        return nominal

    namespace = {'load_cfg_from_registry': load_config}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<nominal_eval>', 'exec'), namespace)
    configure = namespace['configure_nominal_evaluation']
    cfg = SimpleNamespace(
        actions=SimpleNamespace(wbc_force=SimpleNamespace(domain_randomization=True, controller=object())),
        events=object(), scene=SimpleNamespace(robot=object(), terrain=object()),
        commands=SimpleNamespace(motion=SimpleNamespace(pose_range={'x': (-.005, .005)},
            velocity_range={'x': (-.03, .03)}, joint_position_range=(-.005, .005))),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True,
            actual_state=SimpleNamespace(noise=object()))))
    controller = cfg.actions.wbc_force.controller
    configure(cfg)
    assert cfg.events is nominal.events
    assert cfg.scene.robot is nominal.scene.robot and cfg.scene.terrain is nominal.scene.terrain
    assert not cfg.actions.wbc_force.domain_randomization
    assert cfg.actions.wbc_force.controller is controller
    assert cfg.commands.motion.pose_range == cfg.commands.motion.velocity_range == {}
    assert cfg.commands.motion.joint_position_range == (0., 0.)
    assert cfg.observations.policy.actual_state.noise is None
    assert not cfg.observations.policy.enable_corruption
    configure(cfg)  # Already nominal: no further registry lookup or mutation.
    configure(SimpleNamespace(actions=SimpleNamespace(wbc_force=SimpleNamespace())))  # Legacy.
    assert lookups == [('Tracking-WBC-FORCE2-T1-v0', 'env_cfg_entry_point')]


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
