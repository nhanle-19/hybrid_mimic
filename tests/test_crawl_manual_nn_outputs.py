"""Manual policy values are independent of geometric reference annotations."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from crawl_manual_nn_outputs import DEFAULT, load_manual_nn_outputs, save_manual_nn_outputs, save_replay_labels


def test_manual_outputs_support_nonbinary_values_and_independent_off_frames(tmp_path):
    spec = json.loads(DEFAULT.read_text())
    n = spec['frame_count']*spec.get('sample_rate_multiple', 1)
    # The reference hand is grounded here; the policy can still request zero.
    spec['channels']['left_hand']['intervals'] = [[0, 0, -3.], [1, 1, 0.], [2, n-1, 3.]]
    path = tmp_path/'manual.json'
    path.write_text(json.dumps(spec))
    data = load_manual_nn_outputs(path)
    np.testing.assert_array_equal(data['fractions'][:3, 0], [0., .5, 1.])
    assert data['actions'].shape == (n, 8)
    assert not json.loads(str(data['metadata_json']))['reference_contact_mask_applied']
    save_manual_nn_outputs(tmp_path/'outputs.npz', data)
    with np.load(tmp_path/'outputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['actions'], data['actions'])
    np.testing.assert_array_equal(load_manual_nn_outputs(tmp_path/'outputs.npz')['actions'], data['actions'])


def test_control_rate_table_roundtrip_and_reference_frame_indices(tmp_path):
    spec = json.loads(DEFAULT.read_text())
    spec['sample_rate_multiple'] = 4
    spec['channels'] = {name: dict(default_raw=-3., intervals=[]) for name in spec['region_order']}
    spec['channels']['left_hand']['intervals'] = [[0, 0, 3.], [2, 3, 3.]]
    path = tmp_path/'control_rate.json'
    path.write_text(json.dumps(spec))
    data = load_manual_nn_outputs(path)
    assert data['actions'].shape == (599*4, 8)
    np.testing.assert_array_equal(data['reference_frame'][:8], [0,0,0,0,1,1,1,1])
    np.testing.assert_array_equal(data['fractions'][:4, 0], [1.,0.,1.,1.])
    save_manual_nn_outputs(tmp_path/'control_rate.npz', data)
    np.testing.assert_array_equal(load_manual_nn_outputs(tmp_path/'control_rate.npz')['actions'], data['actions'])


def test_replay_labels_do_not_claim_unapplied_stop_record_or_tail(tmp_path):
    planned = load_manual_nn_outputs()
    dt = 1/float(planned['fps'])
    diagnostics = dict(time_s=np.array([0., dt/2, dt, 2*dt]), reference_frame=np.array([0, 0, 1, 2]),
        raw_actions=planned['actions'][[0, 0, 1, 2]], actual_available=np.ones((4, 8, 4), dtype=bool),
        normal_force_limits=np.ones((4, 8))*600)
    report = dict(simulated_seconds=2*dt, full_motion_completed=False, root_error_limit_m=.5)
    save_replay_labels(tmp_path, planned, diagnostics, report)
    with np.load(tmp_path/'labels.npz', allow_pickle=False) as labels:
        np.testing.assert_array_equal(np.flatnonzero(labels['executed_mask']), [0, 1])
        np.testing.assert_array_equal(labels['first_control_update'][:3], [0, 2, -1])
        assert np.isnan(labels['normal_force_cap_n'][2:]).all()
        assert not json.loads(str(labels['metadata_json']))['full_motion_completed']


def test_npz_replay_rejects_inconsistent_fraction(tmp_path):
    planned = load_manual_nn_outputs()
    planned['fractions'][0, 0] = .5
    save_manual_nn_outputs(tmp_path/'bad.npz', planned)
    with pytest.raises(ValueError, match='fractions disagree'):
        load_manual_nn_outputs(tmp_path/'bad.npz')


@pytest.mark.parametrize('change,match', [('overlap', 'overlapping'), ('nan', 'finite'),
    ('motion', 'different motion'), ('order', 'region order')])
def test_invalid_manual_outputs_fail_before_execution(tmp_path, change, match):
    spec = json.loads(DEFAULT.read_text())
    if change == 'overlap':
        spec['channels']['left_hand']['intervals'] = [[0, 3, 1.], [3, 10, -3.]]
    elif change == 'nan':
        spec['channels']['left_hand']['intervals'][0][2] = float('nan')
    elif change == 'motion':
        spec['motion_sha256'] = 'wrong'
    else:
        spec['region_order'].reverse()
    path = tmp_path/'manual.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_manual_nn_outputs(path)
