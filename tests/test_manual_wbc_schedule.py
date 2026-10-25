"""Manual schedules must stay aligned with the exact motion and controller."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from manual_wbc_schedule import (build_schedule, load_schedule, save_schedule, migrate_schedule,
                                 decode_contact_values_numpy, encode_contact_values_numpy)


@pytest.fixture
def example(tmp_path):
    motion = tmp_path/'motion.npz'
    np.savez(motion, fps=50.)
    n = 20
    positions = np.zeros((n, 2, 3))
    positions[:, 0, 1], positions[:, 1, 1] = .1, -.1
    positions[8:14, 1, 2] = .2
    rotations = np.tile(np.eye(3), (n, 2, 1, 1))
    cfg = dict(ground_height=0., contact_force_max=[600., 600.], contact_authority_weights=[100.]*3)
    args = (50., positions, rotations, np.zeros((n, 2, 6)), np.zeros((n, 3)),
            25., np.zeros((2, 4, 3)), cfg, motion, 'prepared')
    return motion, cfg, args


@pytest.mark.parametrize('controller,count', [('force', 2), ('authority', 6)])
def test_schedule_roundtrip_and_support_release(example, tmp_path, controller, count):
    motion, cfg, args = example
    schedule = build_schedule(controller, *args)
    path = save_schedule(tmp_path/'actions.npz', schedule)
    loaded = load_schedule(path, controller, motion, .02, 20, 'prepared', cfg)
    assert loaded['actions'].shape == (20, count)
    assert np.isfinite(loaded['actions']).all()
    np.testing.assert_allclose(decode_contact_values_numpy(loaded['actions']), loaded['fractions'], atol=1e-7)
    assert loaded['phase'][10] == 'left_support'
    if controller == 'force':
        assert loaded['fractions'][9, 0] > loaded['fractions'][9, 1]
        np.testing.assert_array_equal(loaded['fractions'][8:14, 1], 0.)
        assert 0. < loaded['fractions'][7, 1] < loaded['fractions'][6, 1]
        assert 0. < loaded['fractions'][14, 1] < loaded['fractions'][15, 1]
    else:
        assert np.all(loaded['fractions'][10, :3] > loaded['fractions'][10, 3:])
    assert path.with_suffix('.csv').exists()


def test_schedule_rejects_mismatches(example, tmp_path):
    motion, cfg, args = example
    path = save_schedule(tmp_path/'actions.npz', build_schedule('force', *args))
    for options, match in [({'control_dt': .01}, 'fps/control'), ({'frame_count': 19}, 'length'),
                            ({'prepared_sha256': 'other'}, 'Prepared reference'),
                            ({'controller_config': {}}, 'Controller configuration')]:
        with pytest.raises(ValueError, match=match):
            load_schedule(path, 'force', motion, **options)
    with pytest.raises(ValueError, match='controller mismatch'):
        load_schedule(path, 'authority', motion)
    motion.write_bytes(b'different motion')
    with pytest.raises(ValueError, match='different motion'):
        load_schedule(path, 'force', motion)


def test_schedule_rejects_reordered_frames(example, tmp_path):
    motion, cfg, args = example
    schedule = build_schedule('force', *args)
    schedule['reference_frame'][2] = 3
    path = save_schedule(tmp_path/'actions.npz', schedule)
    with pytest.raises(ValueError, match='exactly once'):
        load_schedule(path, 'force', motion)


def test_authority_labels_follow_load_shift_with_stationary_feet(example, tmp_path):
    motion, cfg, args = example
    fps, positions, rotations, velocities, com, mass, points, *_ = args
    positions[:, :, 2] = 0.
    # Hold each load distribution long enough to check beyond the smoothing window.
    com[:10, 1], com[10:, 1] = .02, -.02
    schedule = build_schedule('authority', fps, positions, rotations, velocities, com,
                              mass, points, cfg, motion, 'prepared')
    authority = schedule['fractions'].reshape(-1, 2, 3)
    assert (authority[4, 0] > authority[4, 1]).all()
    assert (authority[15, 1] > authority[15, 0]).all()
    np.testing.assert_allclose(authority[4, 0], authority[15, 1])
    np.testing.assert_allclose(authority[4, 1], authority[15, 0])
    assert (np.ptp(authority, axis=0) > .01).all()
    assert (schedule['phase'] == 'double_support').all()
    path = save_schedule(tmp_path/'load_shift.npz', schedule)
    load_schedule(path, 'authority', motion, .02, 20, 'prepared', cfg)
    # Equal load retains the existing stationary-contact baseline.
    com[:] = 0.
    balanced = build_schedule('authority', fps, positions, rotations, velocities, com,
                              mass, points, cfg, motion, 'prepared')
    np.testing.assert_allclose(balanced['fractions'], np.tile([.9, .9, .95]*2, (20, 1)))


@pytest.mark.parametrize('controller', ['force', 'authority'])
def test_schedule_accepts_explicit_zero_and_full_labels(example, tmp_path, controller):
    motion, _, args = example
    schedule = build_schedule(controller, *args)
    schedule['fractions'][0] = 0.
    schedule['fractions'][1] = 1.
    schedule['actions'] = encode_contact_values_numpy(schedule['fractions']).astype(np.float32)
    path = save_schedule(tmp_path/'endpoints.npz', schedule)
    loaded = load_schedule(path, controller, motion)
    assert (loaded['actions'][0] == -3.).all() and (loaded['actions'][1] == 3.).all()
    assert (decode_contact_values_numpy(loaded['actions'][0]) == 0.).all()
    assert (decode_contact_values_numpy(loaded['actions'][1]) == 1.).all()


@pytest.mark.parametrize('controller', ['force', 'authority'])
def test_legacy_migration_preserves_fractions_without_reusing_validation(example, tmp_path, controller):
    motion, _, args = example
    schedule = build_schedule(controller, *args)
    # Legacy finite sigmoid logits could not encode the new exact-zero swing.
    schedule['fractions'] = np.clip(schedule['fractions'], .025, .98)
    schedule['actions'] = np.log(schedule['fractions']/(1.-schedule['fractions'])).astype(np.float32)
    metadata = json.loads(str(schedule['metadata_json']))
    metadata.update(schema_version=1, status='old_test', validation={'steps_completed': 7})
    metadata.pop('action_transform')
    schedule['metadata_json'] = np.asarray(json.dumps(metadata))
    old_path = save_schedule(tmp_path/'legacy.npz', schedule)
    old_bytes = old_path.read_bytes()
    with pytest.raises(ValueError, match='different action mapping'):
        load_schedule(old_path, controller, motion)
    new_path = migrate_schedule(old_path, tmp_path/'new.npz')
    loaded = load_schedule(new_path, controller, motion)
    np.testing.assert_array_equal(loaded['fractions'], schedule['fractions'])
    np.testing.assert_allclose(decode_contact_values_numpy(loaded['actions']), schedule['fractions'], atol=1e-7)
    assert old_path.read_bytes() == old_bytes
    metadata = json.loads(str(loaded['metadata_json']))
    assert metadata['status'] == 'mapping_migrated_requires_replay'
    assert 'validation' not in metadata
    assert metadata['migration']['previous_validation']['steps_completed'] == 7
    with pytest.raises(ValueError, match='already exists'):
        migrate_schedule(old_path, new_path)
