"""Measure vertical excursions and hand landings in recorded PhysX rollouts.

Velocities come from the recorded world-space root twist, not video pixels.
Forces are sampled ground-filtered PhysX measurements at QP updates; their
peaks are not an assertion about the maximum between those samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation


def analyze(directory, end_time=None):
    directory = Path(directory)
    with np.load(directory/'diagnostics.npz', allow_pickle=False) as archive:
        data = dict(archive)
    validation = json.loads((directory/'validation.json').read_text())
    mask = np.ones(len(data['time_s']), dtype=bool)
    if end_time is not None:
        mask &= data['time_s'] <= end_time
    time = data['time_s'][mask]
    height = data['actual_q'][mask, 2]-data['reference_q'][mask, 2]
    velocity = Rotation.from_quat(data['actual_q'][mask, 3:7]).apply(data['actual_v'][mask, :3])[:, 2]
    reference_velocity = np.gradient(data['reference_q'][mask, 2], time)
    measured_names = data['measured_region_names'].tolist()
    measured = data['measured_forces'][mask, :, 2]
    normal = measured.sum(-1)
    hands = measured[:, [measured_names.index(name) for name in ('left_hand', 'right_hand')]]
    interval = float(np.median(np.diff(time)))
    peaks, _ = find_peaks(height, prominence=.015, distance=max(1, round(.3/interval)))
    events = []
    # A landing must follow >=25 ms unloaded and a measurable >3 mm lift.
    for hand, name in enumerate(('left_hand', 'right_hand')):
        unloaded_since = None
        for i, load in enumerate(hands[:, hand]):
            if load <= 5.:
                if unloaded_since is None:
                    unloaded_since = i
                continue
            start = unloaded_since
            unloaded_since = None
            if start is None or time[i]-time[start] < .025:
                continue
            region = data['region_names'].tolist().index(name)
            gap = data['clearance'][mask, region]
            if gap[start:i].max(initial=0) < .003:
                continue
            end = min(len(time), i+max(1, round(.1/interval)))
            preceding = max(start, i-max(1, round(.2/interval)))
            events.append(dict(region=name, time_s=float(time[i]),
                preceding_hand_clearance_peak_m=float(gap[start:i].max()),
                preceding_root_down_speed_m_s=float(np.maximum(-velocity[preceding:i+1], 0).max()),
                hand_force_peak_next_100ms_n=float(hands[i:end, hand].max()),
                total_force_peak_next_100ms_n=float(normal[i:end].max())))
    summary = dict(run=str(directory), full_motion_completed=validation['full_motion_completed'],
        duration_s=validation['simulated_seconds'], measured_through_s=float(time[-1]),
        root_height_error_max_m=float(height.max()), root_height_error_min_m=float(height.min()),
        root_height_error_rms_m=float(np.sqrt(np.mean(height**2))),
        root_height_error_p95_m=float(np.percentile(height, 95)),
        root_down_speed_max_m_s=float(np.maximum(-velocity, 0).max()),
        root_down_speed_p99_m_s=float(np.percentile(np.maximum(-velocity, 0), 99)),
        root_vertical_speed_error_rms_m_s=float(np.sqrt(np.mean((velocity-reference_velocity)**2))),
        total_ground_force_peak_n=float(normal.max()), total_ground_force_p99_n=float(np.percentile(normal, 99)),
        hand_force_peak_n=float(hands.max()), hand_force_p99_n=float(np.percentile(hands, 99)),
        actual_forward_progress_m=float(data['trajectory_q'][-1, 0]-data['trajectory_q'][0, 0]),
        peak_root_error_m=validation['max_root_tracking_error_m'],
        peak_joint_rmse_deg=float(np.rad2deg(validation['max_joint_tracking_rmse_rad'])),
        qp_failures=validation['qp_failed_updates'],
        root_excursions=[dict(time_s=float(time[i]), height_error_m=float(height[i])) for i in peaks],
        hand_landings=sorted(events, key=lambda event:event['time_s']),
        note='Forces sampled one physics step after the QP state. Landing detections use force and preceding clearance, never reference labels.')
    return summary, dict(time=time, height=height, velocity=velocity, reference_velocity=reference_velocity,
                         hands=hands, normal=normal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--end-time', type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [analyze(directory, args.end_time) for directory in args.directories]
    (args.output/'comparison.json').write_text(json.dumps([r for r, _ in results], indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for (row, data), directory in zip(results, args.directories):
        name = directory.name if directory.name != 'force' else 'previous selected'
        axes[0].plot(data['time'], 100*data['height'], label=name, lw=1)
        axes[1].plot(data['time'], data['velocity'], lw=1)
        axes[2].plot(data['time'], data['hands'].max(-1), lw=1)
        axes[3].plot(data['time'], data['normal'], lw=1)
        print(json.dumps({k:v for k,v in row.items() if k not in ('root_excursions','hand_landings','note')}), flush=True)
    axes[0].set_ylabel('Root height error (cm)')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Root vertical speed (m/s)')
    axes[2].set_ylabel('Maximum hand load (N)')
    axes[3].set(ylabel='Total ground load (N)', xlabel='Time (s)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png', dpi=150)


if __name__ == '__main__':
    main()
