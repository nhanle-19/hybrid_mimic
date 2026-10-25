"""Editable eight-output manual policy schedule; no contact-label multiplier."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from crawl_force_schedule import ROOT, digest
from wbc_crawl_contact import CRAWL_REGIONS
from wbc_contact_mapping import contact_action_transform, decode_contact_values_numpy

DEFAULT = ROOT/'configs/crawl_manual_nn_outputs.json'


def load_manual_nn_outputs(path=DEFAULT, motion_path=None):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        metadata = json.loads(str(data['metadata_json']))
        motion_path = Path(metadata['motion_file']) if motion_path is None else Path(motion_path)
        with np.load(motion_path, allow_pickle=False) as motion:
            n, fps = len(motion['root_pos']), float(motion['fps'])
        multiple = metadata.get('sample_rate_multiple', 1)
        if type(multiple) is not int or multiple < 1:
            raise ValueError('Manual output sample rate must be a positive integer multiple')
        n, fps = n*multiple, fps*multiple
        if (metadata.get('schema_version') != 2 or metadata.get('controller') != 'crawl_force'
                or metadata.get('action_transform') != contact_action_transform()
                or metadata.get('action_layout') != [name+'_fz_limit' for name in CRAWL_REGIONS]):
            raise ValueError('Manual NN output schema/action mapping mismatch')
        if metadata.get('motion_sha256') != digest(motion_path):
            raise ValueError('Manual NN outputs belong to a different motion')
        actions = data['actions']
        if (actions.shape != (n, 8) or not np.isfinite(actions).all()
                or np.max(abs(actions)) > np.finfo(np.float32).max):
            raise ValueError('Manual NN outputs must contain eight finite values per policy sample')
        if (data['fractions'].shape != actions.shape or not np.allclose(
                data['fractions'], decode_contact_values_numpy(actions), atol=1e-8, rtol=0)):
            raise ValueError('Manual NN fractions disagree with raw outputs')
        if (data['region_names'].tolist() != list(CRAWL_REGIONS)
                or not np.array_equal(data['reference_frame'], np.arange(n)//multiple)
                or not np.isclose(float(data['fps']), fps, atol=1e-8, rtol=0)
                or not np.allclose(data['time_s'], np.arange(n)/fps, atol=1e-8, rtol=0)):
            raise ValueError('Manual NN output timing or region order mismatch')
        return data
    spec = json.loads(path.read_text())
    if spec.get('schema_version') != 1 or spec.get('action_transform') != contact_action_transform():
        raise ValueError('Manual NN output schema/action mapping mismatch')
    if spec.get('region_order') != list(CRAWL_REGIONS) or set(spec['channels']) != set(CRAWL_REGIONS):
        raise ValueError('Manual NN output region order mismatch')
    motion_path = ROOT/spec['motion'] if motion_path is None else Path(motion_path)
    if digest(motion_path) != spec['motion_sha256']:
        raise ValueError('Manual NN outputs belong to a different motion')
    with np.load(motion_path, allow_pickle=False) as motion:
        n, fps = len(motion['root_pos']), float(motion['fps'])
    if spec['frame_count'] != n or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Manual NN output timing mismatch')
    multiple = spec.get('sample_rate_multiple', 1)
    if type(multiple) is not int or multiple < 1:
        raise ValueError('Manual output sample rate must be a positive integer multiple')
    reference_fps = fps
    n, fps = n*multiple, fps*multiple
    actions = np.empty((n, 8), dtype=np.float64)
    for i, name in enumerate(CRAWL_REGIONS):
        channel = spec['channels'][name]
        actions[:, i] = channel['default_raw']
        previous = -1
        for start, end, value in channel['intervals']:
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < n or start <= previous:
                raise ValueError(f'Invalid or overlapping NN-output interval for {name}')
            actions[start:end+1, i] = value
            previous = end
    if not np.isfinite(actions).all() or np.max(abs(actions)) > np.finfo(np.float32).max:
        raise ValueError('Manual NN outputs must be finite float32-range numbers')
    # Values are independent of annotated contact state; OFF is encoded in the
    # output itself, and live geometry separately checks physical availability.
    fractions = decode_contact_values_numpy(actions)
    metadata = dict(schema_version=2, controller='crawl_force',
        action_layout=[name+'_fz_limit' for name in CRAWL_REGIONS],
        action_transform=contact_action_transform(), motion_file=str(motion_path.resolve()),
        motion_sha256=spec['motion_sha256'], output_spec_sha256=digest(path),
        design='manually_specified_raw_NN_output_intervals', review=spec['review'],
        sample_rate_multiple=multiple, reference_fps=reference_fps,
        status='manual_candidate_requires_physics_validation', reference_contact_mask_applied=False)
    return dict(actions=actions, fractions=fractions, reference_frame=np.arange(n)//multiple,
        time_s=np.arange(n)/fps, fps=np.asarray(fps), region_names=np.asarray(CRAWL_REGIONS),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def save_manual_nn_outputs(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        names = data['region_names'].tolist()
        execution = 'executed_mask' in data
        high_rate = json.loads(str(data['metadata_json'])).get('sample_rate_multiple', 1) > 1
        writer.writerow((['sample', 'reference_frame', 'time_s'] if high_rate else ['frame', 'time_s'])+(['executed'] if execution else [])
                        +['raw_'+n for n in names]+['fraction_'+n for n in names])
        for i in range(len(data['actions'])):
            writer.writerow(([i, data['reference_frame'][i], data['time_s'][i]] if high_rate else [i, data['time_s'][i]])+([bool(data['executed_mask'][i])] if execution else [])
                            +list(data['actions'][i])+list(data['fractions'][i]))


def save_replay_labels(output, planned, diagnostics, report):
    """Keep full planned rows, explicitly flag which policy samples executed.

    A final QP/termination record does not mean its torque was applied. Actual
    availability/caps use the first applied update of each policy sample;
    all control updates remain in diagnostics.npz.
    """
    labels = dict(planned)
    n = len(planned['actions'])
    labels['executed_mask'] = np.zeros(n, dtype=bool)
    labels['first_control_update'] = np.full(n, -1, dtype=int)
    labels['actual_contact_available'] = np.zeros((n, 8), dtype=bool)
    labels['normal_force_cap_n'] = np.full((n, 8), np.nan)
    for update, time_s in enumerate(diagnostics.get('time_s', [])):
        if time_s >= report['simulated_seconds']-1e-10:
            continue
        frame = int(diagnostics.get('policy_sample', diagnostics['reference_frame'])[update])
        if labels['executed_mask'][frame]:
            continue
        np.testing.assert_array_equal(diagnostics['raw_actions'][update], planned['actions'][frame])
        labels['executed_mask'][frame] = True
        labels['first_control_update'][frame] = update
        labels['actual_contact_available'][frame] = diagnostics['actual_available'][update].any(-1)
        labels['normal_force_cap_n'][frame] = diagnostics['normal_force_limits'][update]
    metadata = json.loads(str(planned['metadata_json']))
    metadata.update(status='preliminary_rollout_completed' if report['full_motion_completed'] else 'partial_preliminary_rollout',
        full_motion_completed=report['full_motion_completed'], executed_samples=int(labels['executed_mask'].sum()),
        executed_frames=int(np.unique(labels['reference_frame'][labels['executed_mask']]).size),
        root_error_limit_m=report['root_error_limit_m'],
        execution_note='Rows with executed_mask=false are planned only. Caps/availability use first applied update per policy sample; diagnostics contains every QP update.')
    labels['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    save_manual_nn_outputs(Path(output)/'labels.npz', labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/crawl_manual_export/planned_labels.npz')
    args = parser.parse_args()
    data = load_manual_nn_outputs(args.input)
    save_manual_nn_outputs(args.output, data)
    print('Exported', len(data['actions']), 'samples of eight raw NN outputs:', args.output)
