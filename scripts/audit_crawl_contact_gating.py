"""Audit reference labels, live contact masks and measured support separately."""
import argparse
import json
from pathlib import Path

import numpy as np

from crawl_force_schedule import ROOT, load_manual_schedule
from wbc_contact_mapping import decode_contact_values_numpy


def intervals(mask, times, frames):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    return [dict(first_update=int(g[0]), last_update=int(g[-1]), updates=len(g),
                 start_s=float(times[g[0]]), end_s=float(times[g[-1]]),
                 first_reference_frame=int(frames[g[0]]), last_reference_frame=int(frames[g[-1]])) for g in groups]


def audit(path, schedule):
    d = dict(np.load(path/'diagnostics.npz', allow_pickle=False))
    report = json.loads((path/'validation.json').read_text())
    frame, names = d['reference_frame'], d['region_names'].tolist()
    np.testing.assert_array_equal(d['contact_labels'], schedule['contact_labels'][frame])
    manual_path = path/'planned_labels.npz'
    if not manual_path.exists():
        manual_path = path/'manual_nn_outputs.npz'
    manual = dict(np.load(manual_path, allow_pickle=False)) if manual_path.exists() else schedule
    np.testing.assert_array_equal(d['raw_actions'], manual['actions'][d.get('policy_sample', frame)])
    assert names == schedule['region_names'].tolist()
    enabled = d['available'].any(-1)
    actual = d['actual_available'].any(-1)
    near_floor = d.get('near_floor_available', d['actual_available']).any(-1)
    labels = d['contact_labels']
    output_on = decode_contact_values_numpy(d['raw_actions']) > 0
    assert not (enabled & ~output_on).any()
    assert not (enabled & ~actual).any()
    assert not np.count_nonzero(d['point_forces'][~d['available']])
    assert not np.count_nonzero(d['normal_force_limits'][~enabled])
    rows = {}
    for i, name in enumerate(names):
        measured = d['measured_forces'][:, d['measured_region_names'].tolist().index(name), 2]
        force = d['point_forces'][:, i, :, 2].sum(-1)
        unsupported = (force > 5.) & (measured < 1.)
        separation = (d['point_forces'][:, i, :, 2]*d['point_velocity_w'][:, i, :, 2]).sum(-1)
        separation = np.divide(separation, force, out=np.zeros_like(force), where=force > 0)
        rows[name] = dict(reference_on_updates=int(labels[:, i].sum()),
            output_on_updates=int(output_on[:, i].sum()),
            positive_output_while_geometry_unavailable_updates=int((output_on[:, i] & ~actual[:, i]).sum()),
            output_off_but_qp_enabled_updates=int((~output_on[:, i] & enabled[:, i]).sum()),
            label_off_but_qp_enabled_updates=int((~labels[:, i] & enabled[:, i]).sum()),
            geometrically_off_but_qp_enabled_updates=int((~near_floor[:, i] & enabled[:, i]).sum()),
            separation_gate_disabled_updates=int((near_floor[:, i] & ~actual[:, i]).sum()),
            predicted_force_when_off_max_n=float(np.max(force[~enabled[:, i]], initial=0)),
            label_on_but_actual_geometry_off=intervals(labels[:, i] & ~near_floor[:, i], d['time_s'], frame),
            label_on_but_live_support_off=intervals(labels[:, i] & ~actual[:, i], d['time_s'], frame),
            label_off_but_measured_load=intervals(~labels[:, i] & (measured > 1.), d['time_s'], frame),
            label_off_measured_load_peak_n=float(np.max(measured[~labels[:, i]], initial=0)),
            predicted_load_without_measured_load=intervals(unsupported, d['time_s'], frame),
            predicted_load_while_clear_and_separating=intervals(
                unsupported & (d['clearance'][:, i] > 0.) & (separation > .005), d['time_s'], frame))
    outside = [i for i, name in enumerate(d['measured_region_names']) if name not in names]
    return dict(run=str(path.relative_to(ROOT)), control_updates=len(frame),
        all_reference_rows_match=True, output_and_geometry_force_gates_pass=True,
        raw_manual_outputs_exactly_preserved=True,
        positive_outputs_only_on_available_geometry=bool(not (output_on & ~actual).any()),
        measured_force_time_note='Normally one physics step after QP state; final failed/stop record is at QP time.',
        unsupported_thresholds=dict(predicted_above_n=5., measured_below_n=1., separating_velocity_above_m_s=.005),
        contact_interpretation='Geometric proximity allows QP support; it does not prove positive physical load. Zero measured load alone is not proof the reference label is wrong.',
        unmodeled_region_measured_force_peak_n=float(np.max(d['measured_forces'][:, outside, 2], initial=0)),
        completed_motion=report['full_motion_completed'], regions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schedule', type=Path, default=ROOT/'configs/crawl_force_schedule.json')
    args = parser.parse_args()
    spec, _, geometry, schedule = load_manual_schedule(args.schedule)
    reference = dict(frames=spec['frame_count'], region_order=spec['region_order'],
        reference_label_geometry_mismatches=0,
        labels_on_frames=schedule['contact_labels'].sum(0).tolist(),
        source_caveat='Both hands and knees were constrained to stay grounded during repair; these are repaired-robot geometric labels, not independently recovered human support phases.',
        all_anatomical_regions_checked=geometry['surface_names'].tolist())
    results = [audit(p.resolve(), schedule) for p in args.directories]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(reference=reference, runs=results), indent=2)+'\n')
    for r in results:
        print(r['run'], ': OFF force gates pass')
        for name, row in r['regions'].items():
            unloaded = sum(v['updates'] for v in row['predicted_load_without_measured_load'])
            separating = sum(v['updates'] for v in row['predicted_load_while_clear_and_separating'])
            if unloaded or row['label_off_measured_load_peak_n'] > 1.:
                print(f'  {name}: predicted load without measurement={unloaded} updates; '
                      f'clear and separating={separating}; off-label measured peak={row["label_off_measured_load_peak_n"]:.2f} N')


if __name__ == '__main__':
    main()
