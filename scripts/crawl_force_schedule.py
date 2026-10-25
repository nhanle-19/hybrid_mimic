"""Build and validate the manually specified eight-region crawl contact table."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_crawl_contact import CRAWL_REGIONS
from wbc_contact_mapping import encode_contact_values_numpy, contact_action_transform


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manual_schedule(path=ROOT/'configs/crawl_force_schedule.json'):
    path = Path(path)
    specification = json.loads(path.read_text())
    if specification.get('schema_version') != 1 or specification['region_order'] != list(CRAWL_REGIONS):
        raise ValueError('Manual crawl schema/region order mismatch')
    if specification['normal_world'] != [0., 0., 1.]:
        raise ValueError('Crawl FORCE currently supports the horizontal floor normal only')
    for key in ('motion', 'geometry'):
        if digest(ROOT/specification[key]) != specification[key+'_sha256']:
            raise ValueError(f'Manual crawl schedule has stale {key} hash')
    with np.load(ROOT/specification['motion'], allow_pickle=False) as data:
        motion = dict(data)
    with np.load(ROOT/specification['geometry'], allow_pickle=False) as data:
        geometry = dict(data)
    n = len(motion['root_pos'])
    fps = float(motion['fps'])
    if specification['frame_count'] != n or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Manual crawl frame count/fps mismatch')
    if not np.isclose(float(geometry['fps']), fps) or len(geometry['time_s']) != n:
        raise ValueError('Crawl annotation timing mismatch')
    caps = np.asarray(specification['force_max_n'], dtype=float)
    if caps.shape != (8,) or not np.isfinite(caps).all() or (caps <= 0).any():
        raise ValueError('Crawl schedule requires eight positive finite force maxima')
    labels = np.zeros((n, 8), dtype=bool)
    if set(specification['contact_intervals']) != set(CRAWL_REGIONS):
        raise ValueError('Crawl interval table must specify every region exactly once')
    for region, name in enumerate(CRAWL_REGIONS):
        end = -1
        for interval in specification['contact_intervals'][name]:
            if (len(interval) != 2 or any(type(x) is not int for x in interval)
                    or not 0 <= interval[0] <= interval[1] < n or interval[0] <= end):
                raise ValueError(f'Invalid/overlapping contact interval for {name}: {interval}')
            start, end = interval
            labels[start:end+1, region] = True
    names = geometry['surface_names'].tolist()
    columns = [names.index(name) for name in CRAWL_REGIONS]
    omitted = [i for i, name in enumerate(names) if name not in CRAWL_REGIONS]
    if geometry['surface_active'][:, omitted].any():
        raise ValueError('A contacting region is omitted from the eight-region QP')
    expected = geometry['surface_active'][:, columns].astype(bool)
    gaps = geometry['surface_clearance_m'][:, columns]
    geometric = ((gaps <= specification['contact_gap_tolerance_m'])
                 & (gaps >= -specification['max_contact_penetration_m']))
    if not np.array_equal(labels, expected) or not np.array_equal(labels, geometric):
        mismatch = np.argwhere((labels != expected) | (labels != geometric)).tolist()
        raise ValueError(f'Manual contact labels disagree with geometric evidence: {mismatch[:10]}')
    fractions = labels.astype(float)
    metadata = dict(schema_version=1, controller='crawl_force', region_order=list(CRAWL_REGIONS),
                    action_transform=contact_action_transform(), motion_sha256=specification['motion_sha256'],
                    geometry_sha256=specification['geometry_sha256'], interval_table_sha256=digest(path),
                    design='manually_specified_intervals_with_geometric_validation', review=specification['review'],
                    force_meaning='Upper bounds, not desired or measured forces', physical_rollout_verified=False)
    schedule = dict(reference_frame=np.arange(n), time_s=np.arange(n)/fps, fps=np.asarray(fps),
                    region_names=np.asarray(CRAWL_REGIONS), contact_labels=labels, fractions=fractions,
                    actions=encode_contact_values_numpy(fractions), force_max_n=caps,
                    normal_force_limits=fractions*caps, surface_clearance_m=gaps,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    return specification, motion, geometry, schedule


def save_schedule(output, schedule):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'manual_schedule.npz', **schedule)
    with (output/'manual_schedule.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'time_s']+[f'{name}_{field}' for name in CRAWL_REGIONS
                                             for field in ('contact', 'raw_action', 'normal_limit_n', 'clearance_mm')])
        for frame, time in enumerate(schedule['time_s']):
            writer.writerow([frame, time]+[value for i in range(8) for value in
                (int(schedule['contact_labels'][frame, i]), schedule['actions'][frame, i],
                 schedule['normal_force_limits'][frame, i], schedule['surface_clearance_m'][frame, i]*1000)])
    # This figure exposes every one-frame interruption; interpolation is absent.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.imshow(schedule['contact_labels'].T, aspect='auto', interpolation='nearest', cmap='Greys', vmin=0, vmax=1,
              extent=(-.5, len(schedule['time_s'])-.5, 7.5, -.5))
    ax.set_yticks(np.arange(8), [name.replace('_', ' ') for name in CRAWL_REGIONS])
    ax.set(xlabel='Reference frame (zero-based)', title='Manual crawl contact intervals | black = enabled, white = zero force')
    fig.tight_layout()
    fig.savefig(output/'contact_schedule.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, default=ROOT/'configs/crawl_force_schedule.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/crawl_contact_review')
    args = parser.parse_args()
    specification, motion, geometry, schedule = load_manual_schedule(args.schedule)
    save_schedule(args.output, schedule)
    print('Saved', len(schedule['time_s']), 'frames with eight manually specified contact channels to', args.output)
