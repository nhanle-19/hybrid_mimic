"""Summarize deterministic FORCE evaluations, preserving stance and moving contacts."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def percent(value, mask):
    return float(np.asarray(value)[mask].mean()*100) if np.any(mask) else None


def bouts(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def geometry(position, rotation, velocity, vertices):
    offsets = np.einsum('tfij,fvj->tfvi', rotation, vertices)
    points = offsets + position[:, :, None]
    speeds = velocity[:, :, None, :3] + np.cross(velocity[:, :, None, 3:], offsets)
    return points[..., 2].min(-1), points, speeds


def analyze(directory):
    data = dict(np.load(directory/'diagnostics.npz', allow_pickle=False))
    meta = json.loads(str(data['metadata_json']))
    n, total = meta['attempted_frames'], meta['expected_frames']
    np.testing.assert_array_equal(data['audit_frame'], np.arange(n))
    np.testing.assert_allclose(data['audit_cap'], data['support_activation'], atol=1e-6)
    for key in ('audit_actual_q', 'actual_contact_wrench', 'audit_cap', 'audit_foot_position'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{directory}: nonfinite {key}')
    if len(data['actual_contact_wrench']) != n:
        raise ValueError('Every attempted frame must have a diagnostic wrench')
    vertices = data['sole_vertices']
    ref_height, ref_points, ref_speed = geometry(data['full_reference_foot_position'],
        data['full_reference_foot_rotation'], data['full_reference_foot_velocity'], vertices)
    actual_height, actual_points, actual_speed = geometry(data['audit_foot_position'],
        data['audit_foot_rotation'], data['audit_foot_velocity'], vertices)
    ground = json.loads(str(data['controller_config_json']))['ground_height']
    ref_height -= ground
    actual_height -= ground
    ref_points[..., 2] -= ground
    actual_points[..., 2] -= ground
    caps = data['audit_cap']
    fz = data['actual_contact_wrench'][..., 2]
    stance, swing = ref_height[:n] <= .001, ref_height[:n] >= .005
    feet = {}
    for i, side in enumerate(('left', 'right')):
        # Initial reset has no settled sensor history. First-substep wrench is
        # available for every executed action, including the failure frame.
        zero = caps[:, i] == 0
        events = []
        for start, end in bouts(ref_height[:, i] >= .005):
            observed_end = min(end, n)
            observed = max(0, observed_end-start)
            events.append(dict(start_frame=int(start), end_frame_exclusive=int(end),
                reference_peak_m=float(ref_height[start:end, i].max()),
                observed_frames=int(observed), expected_frames=int(end-start),
                actual_peak_m=float(actual_height[start:observed_end, i].max()) if observed else None,
                airborne_unloaded_frames=int(((actual_height[start:observed_end, i] >= .005)
                                              & (fz[start:observed_end, i] <= 5)).sum()) if observed else 0))
        near_floor = ref_points[:n, i, :, 2] <= .001
        stationary = near_floor & (np.linalg.norm(ref_speed[:n, i, :, :2], axis=-1) <= .02)
        loaded_stationary = stationary & (fz[:, i, None] > 10)
        ref_patch = near_floor & (ref_points[:n, i, :, 2] <= ref_height[:n, i, None]+.001)
        actual_patch = ((actual_points[:, i, :, 2] <= .001)
                        & (actual_points[:, i, :, 2] <= actual_height[:, i, None]+.001))
        patch_union = (ref_patch | actual_patch).sum(-1)
        patch_iou = (ref_patch & actual_patch).sum(-1)/np.maximum(patch_union, 1)
        tangential_error = np.linalg.norm(actual_speed[:, i, :, :2]-ref_speed[:n, i, :, :2], axis=-1)
        contact_speed = np.linalg.norm(actual_speed[:, i, :, :2], axis=-1)
        full_stance = ref_height[:, i] <= .001
        full_swing = ref_height[:, i] >= .005
        feet[side] = dict(
            stance_frames=int(stance[:, i].sum()), full_reference_stance_frames=int(full_stance.sum()),
            swing_frames=int(swing[:, i].sum()), full_reference_swing_frames=int(full_swing.sum()),
            stance_loaded_percent_at_10N=percent(fz[:, i] > 10, stance[:, i]),
            stance_loaded_percent_at_1N=percent(fz[:, i] > 1, stance[:, i]),
            stance_loaded_percent_at_5N=percent(fz[:, i] > 5, stance[:, i]),
            stance_geometric_touch_percent=percent((actual_height[:, i] <= .001) & (actual_height[:, i] >= -.03), stance[:, i]),
            stance_unintended_airborne_percent=percent(actual_height[:, i] >= .005, stance[:, i]),
            unintended_stance_lift_count=len(bouts(stance[:, i] & (actual_height[:, i] >= .005))),
            stance_geometric_and_loaded_percent=percent((actual_height[:, i] <= .001) & (fz[:, i] > 10), stance[:, i]),
            stance_geometric_patch_iou_percent=percent(patch_iou, stance[:, i]),
            stance_reference_low_vertices_mean=float(ref_patch[stance[:, i]].sum(-1).mean()) if stance[:, i].any() else None,
            stance_actual_low_vertices_mean=float(actual_patch[stance[:, i]].sum(-1).mean()) if stance[:, i].any() else None,
            stance_loaded_full_clip_coverage_percent=(float(((fz[:, i] > 10) & stance[:, i]).sum()/full_stance.sum()*100)
                                                     if full_stance.any() else None),
            stance_zero_cap_percent=percent(zero, stance[:, i]),
            stance_measured_fz_mean_N=float(fz[stance[:, i], i].mean()) if stance[:, i].any() else None,
            stance_measured_fz_max_N=float(fz[stance[:, i], i].max()) if stance[:, i].any() else None,
            swing_zero_cap_percent=percent(zero, swing[:, i]),
            swing_airborne_percent=percent(actual_height[:, i] >= .005, swing[:, i]),
            swing_unloaded_percent=percent(fz[:, i] <= 5, swing[:, i]),
            swing_airborne_and_unloaded_percent=percent((actual_height[:, i] >= .005) & (fz[:, i] <= 5), swing[:, i]),
            swing_airborne_unloaded_full_clip_coverage_percent=(float(((actual_height[:, i] >= .005) & (fz[:, i] <= 5) & swing[:, i]).sum()/full_swing.sum()*100)
                                                              if full_swing.any() else None),
            swing_measured_fz_mean_N=float(fz[swing[:, i], i].mean()) if swing[:, i].any() else None,
            swing_cap_mean=float(caps[swing[:, i], i].mean()) if swing[:, i].any() else None,
            longest_zero_cap_hold_s=float(max([b-a for a, b in bouts(zero)] or [0])*meta['policy_dt_s']),
            cap_change_mean=float(np.abs(np.diff(caps[:, i])).mean()) if n > 1 else None,
            foot_position_rmse_m=float(np.sqrt(np.square(data['audit_foot_position'][:, i]-data['full_reference_foot_position'][:n, i]).sum(-1).mean())),
            sole_clearance_rmse_m=float(np.sqrt(np.square(actual_height[:, i]-ref_height[:n, i]).mean())),
            stationary_contact_slip_mean_m_s=float(contact_speed[loaded_stationary].mean()) if loaded_stationary.any() else None,
            contact_vertex_velocity_error_mean_m_s=float(tangential_error[near_floor].mean()) if near_floor.any() else None,
            reference_foot_xy_excursion_m=float(np.linalg.norm(data['full_reference_foot_position'][:n, i, :2]-data['full_reference_foot_position'][0, i, :2], axis=-1).max()),
            actual_foot_xy_excursion_m=float(np.linalg.norm(data['audit_foot_position'][:, i, :2]-data['audit_foot_position'][0, i, :2], axis=-1).max()),
            swing_events=events)
    root_error = np.linalg.norm(data['audit_actual_q'][:, :3]-data['audit_reference_q'][:, :3], axis=-1)
    rot_error = np.rad2deg(np.arccos(np.clip((np.einsum('tfij,tfij->tf', data['audit_foot_rotation'],
                        data['full_reference_foot_rotation'][:n])-1)/2, -1, 1)))
    incidental_ids = [j for j, name in enumerate(data['body_contact_names']) if name not in data['foot_names']]
    body_forces = np.linalg.norm(data['audit_body_contact_force'], axis=-1)
    incidental = {str(data['body_contact_names'][j]): int((body_forces[1:, j] > 10).sum())
                  for j in incidental_ids if (body_forces[1:, j] > 10).any()}
    result = dict(motion=directory.name, completed=meta['completed'], attempted_frames=n, expected_frames=total,
        completion_percent=n/total*100, failure_reason=meta['failure_reason'],
        terminated_at_s=n*meta['policy_dt_s'] if not meta['completed'] else None,
        qp_failure_count=int(data['solver_failed'].sum()), qp_residual_max=float(data['qp_residual'].max()),
        joint_rmse_rad=float(np.sqrt(np.square(data['audit_actual_q'][:, 7:]-data['audit_reference_q'][:, 7:]).mean())),
        root_position_rmse_m=float(np.sqrt(np.square(root_error).mean())), root_position_max_m=float(root_error.max()),
        foot_rotation_mean_deg=rot_error.mean(0).tolist(),
        both_caps_zero_percent=float((caps == 0).all(-1).mean()*100),
        incidental_nonfoot_contact_samples=incidental,
        feet=feet, metadata=meta)
    (directory/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    t = np.arange(n)*meta['policy_dt_s']
    tf = np.arange(total)*meta['policy_dt_s']
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
    for i, side in enumerate(('Left', 'Right')):
        axes[0, i].plot(tf, ref_height[:, i]*1000, '--', label='Reference')
        axes[0, i].plot(t, actual_height[:, i]*1000, label='Actual')
        axes[0, i].set(title=side+' foot', ylabel='Lowest sole vertex (mm)')
        axes[1, i].step(t, caps[:, i], where='post', label='Learned force cap')
        axes[1, i].fill_between(tf, 0, 1, where=ref_height[:, i] >= .005, alpha=.15, color='orange', label='Reference swing')
        axes[1, i].set(ylabel='Cap fraction', ylim=(-.03, 1.03))
        axes[2, i].plot(t, fz[:, i], label='Measured Fz, first substep')
        axes[2, i].plot(t, data['predicted_contact_wrench'][:, i, 2], '--', label='QP Fz')
        axes[2, i].set(ylabel='Vertical force (N)')
        err = np.linalg.norm(data['audit_foot_position'][:, i]-data['full_reference_foot_position'][:n, i], axis=-1)
        axes[3, i].plot(t, err*1000, label='Foot position error')
        axes[3, i].set(ylabel='Position error (mm)')
        axes[4, i].plot(t, rot_error[:, i], label='Foot orientation error')
        axes[4, i].set(ylabel='Orientation error (deg)', xlabel='Reference time (s)')
        for ax in axes[:, i]:
            if not meta['completed']:
                ax.axvspan(n*meta['policy_dt_s'], total*meta['policy_dt_s'], color='gray', alpha=.2, label='Not reached')
            ax.set_xlim(0, total*meta['policy_dt_s'])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle(f'{directory.name}: {n}/{total} frames | '+('completed' if meta['completed'] else meta['failure_reason']), fontsize=11)
    fig.tight_layout()
    fig.savefig(directory/'contacts.png', dpi=140)
    plt.close(fig)
    np.savetxt(directory/'timeseries.csv', np.column_stack((np.arange(n), t, caps, ref_height[:n], actual_height, fz)),
        delimiter=',', comments='', header='frame,time_s,left_cap,right_cap,left_reference_clearance_m,right_reference_clearance_m,left_actual_clearance_m,right_actual_clearance_m,left_fz_N,right_fz_N')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    results = [analyze(p.parent) for p in sorted(args.directory.glob('*/diagnostics.npz'))]
    if not results:
        raise ValueError('No evaluation diagnostics found')
    definitions = dict(stance='Reference lowest collision vertex <=1 mm; preserves toe and sole contacts.',
        swing='Reference lowest collision vertex >=5 mm; intermediate frames excluded from classification.',
        geometric_touch='Actual lowest sole vertex <=1 mm, excluding penetration deeper than 30 mm, during reference contact. A lightly loaded toe can retain contact while remaining below the load thresholds.',
        measured_load='Measured physical foot-ground Fz >10 N; 1 N and 5 N sensitivity also provided. These are load thresholds, not contact-existence tests or imitation pass/fail criteria. Reference geometry alone does not specify a required load distribution.',
        zero_cap='Decoded policy output equals zero. This constrains the QP predicted foot wrench, not the physical collision response. It is not inferred from foot height and does not establish contact loss.',
        airborne_unloaded_swing='Actual lowest sole >=5 mm AND measured Fz <=5 N during reference swing.',
        sliding_rotation='Reference moving contacts are retained: no blanket no-slip requirement. Stationary-contact slip uses only reference vertices <=1 mm with horizontal speed <=0.02 m/s.',
        contact_patch='Patch overlap is a geometric proxy using lowest collision vertices within 1 mm of floor and of the lowest vertex. Physical load is measured separately, not inferred per vertex.',
        scope='One deterministic frame-zero attempt per motion. Contact percentages describe executed prefixes; full-clip coverage also penalizes unreached frames. Completion does not establish imitation quality.',
        sampling='Contact wrenches sampled 2 ms after each policy action, not averaged across the 20 ms control interval.')
    process_notes = []
    if (args.directory/'suite.json').exists():
        suite = json.loads((args.directory/'suite.json').read_text())
        for run in suite['runs']:
            if run.get('exit_code', 0):
                process_notes.append(f"{run['motion']}: process exit {run['exit_code']}; diagnostics and metadata are independently validated.")
    (args.directory/'summary.json').write_text(json.dumps(dict(definitions=definitions, process_notes=process_notes, motions=results), indent=2, allow_nan=False)+'\n')
    def fmt(value):
        return '—' if value is None else f'{value:.1f}'
    rows = []
    contact_rows = []
    for r in results:
        f = r['feet']
        rows.append(dict(motion=r['motion'], complete=r['completed'], frames=f"{r['attempted_frames']}/{r['expected_frames']}",
            left_swing_pct=f['left']['swing_airborne_and_unloaded_percent'], right_swing_pct=f['right']['swing_airborne_and_unloaded_percent'],
            left_foot_rmse_mm=f['left']['foot_position_rmse_m']*1000, right_foot_rmse_mm=f['right']['foot_position_rmse_m']*1000,
            joint_rmse_rad=r['joint_rmse_rad'], qp_failures=r['qp_failure_count']))
        for side in ('left', 'right'):
            foot = f[side]
            contact_rows.append(dict(motion=r['motion'], foot=side,
                geometric_touch_pct=foot['stance_geometric_touch_percent'],
                measured_load_gt_10N_pct=foot['stance_loaded_percent_at_10N'],
                measured_fz_mean_N=foot['stance_measured_fz_mean_N'],
                policy_zero_cap_pct=foot['stance_zero_cap_percent']))
    with (args.directory/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.directory/'contact_summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contact_rows[0]))
        writer.writeheader()
        writer.writerows(contact_rows)
    def table(items):
        header = '<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in items[0])+'</tr>'
        body = ''.join('<tr>'+''.join(f'<td>{html.escape(fmt(v) if v is None or isinstance(v, float) else str(v))}</td>' for v in row.values())+'</tr>' for row in items)
        return '<div style="overflow-x:auto"><table>'+header+body+'</table></div>'
    plots = ''.join(f'<h2>{html.escape(r["motion"])}</h2><p>{html.escape(r["failure_reason"] or "Reached the full clip")}</p>'
        + (f'<video controls preload="metadata" src="{r["motion"]}/comparison.mp4" style="max-width:960px;width:100%"></video>'
           if (args.directory/r['motion']/'comparison.mp4').exists() else '')
        + f'<img src="{r["motion"]}/contacts.png" style="max-width:1100px;width:100%">' for r in results)
    page = '<!doctype html><meta charset="utf-8"><title>FORCE contact evaluation</title><style>body{font:15px system-ui;margin:30px;max-width:1400px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}th{background:#eee}</style>'
    checkpoint_name = Path(results[0]['metadata']['checkpoint']).name
    page += f'<h1>{html.escape(checkpoint_name)}: all-motion contact evaluation</h1><p>{sum(r["completed"] for r in results)}/{len(results)} motions completed.</p>'
    if (args.directory/'overlay/all_tasks_overlay.mp4').exists():
        page += '<h2>All tasks with reference overlay</h2><video controls preload="metadata" poster="overlay/preview.png" src="overlay/all_tasks_overlay.mp4" style="max-width:1280px;width:100%"></video><p><a href="overlay/preview.html">Open player with task chapters</a></p>'
    page += '<p><b>Touch, measured load, and the policy force cap are separate measurements.</b> A zero cap or load below 10 N does not establish loss of physical contact. Use geometry and motion tracking to assess whether a lightly loaded toe, rotating sole, or sliding foot retains the reference contact.</p>'
    page += '<ul>'+''.join(f'<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>' for k,v in definitions.items())+'</ul>'
    if process_notes:
        page += '<p>Simulator process warnings: '+html.escape(' '.join(process_notes))+'</p>'
    page += '<h2>Completion and imitation</h2>'+table(rows)
    page += '<h2>Contact geometry, load, and policy output</h2><p>All values below describe executed frames in which the reference foot touches the floor. Load and cap columns do not grade contact preservation.</p>'+table(contact_rows)+plots
    (args.directory/'report.html').write_text(page)
    # One overview keeps survival separate from contact retention and pose error.
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, len(results)*.55)), gridspec_kw={'width_ratios': [1, 4, 2]})
    values = np.array([[r['completion_percent'],
        *[r['feet'][s]['stance_geometric_touch_percent'] for s in ('left', 'right')],
        *[r['feet'][s]['swing_airborne_and_unloaded_percent'] for s in ('left', 'right')],
        *[r['feet'][s]['foot_position_rmse_m']*1000 for s in ('left', 'right')]] for r in results], dtype=float)
    for ax, subset, labels, vmax in [(axes[0], values[:, :1], ['Frames\nattempted %'], 100),
            (axes[1], values[:, 1:5], ['Left touch %', 'Right touch %', 'Left swing %', 'Right swing %'], 100),
            (axes[2], values[:, 5:], ['Left foot\nRMSE mm', 'Right foot\nRMSE mm'], max(30, float(values[:, 5:].max())))]:
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad('#eeeeee')
        ax.imshow(subset, vmin=0, vmax=vmax, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(results)), [r['motion'] for r in results] if ax is axes[0] else ['']*len(results))
        for i in range(subset.shape[0]):
            for j in range(subset.shape[1]):
                value = subset[i, j]
                ax.text(j, i, '—' if np.isnan(value) else f'{value:.1f}', ha='center', va='center',
                        color='white' if value > vmax*.6 else 'black', fontsize=9)
    fig.suptitle('Contact preservation per motion | deterministic checkpoint evaluation', fontsize=12)
    fig.text(.5, .01, 'Touch: lowest sole <=1 mm in reference contact (penetration >30 mm excluded). Swing: sole >=5 mm and Fz <=5 N.\nTouch does not imply load; zero policy cap does not imply lost contact. Metrics cover attempted prefixes; — = no evaluated reference swing.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(args.directory/'overview.png', dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
