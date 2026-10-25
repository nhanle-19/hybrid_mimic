"""Plot a manual action schedule and its available first-failure rollout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.schedule, allow_pickle=False) as data:
        schedule = dict(data)
    meta = json.loads(str(schedule['metadata_json']))
    t = schedule['time_s']
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for foot, name in enumerate(('Left', 'Right')):
        axes[0].plot(t, schedule['foot_clearance_m'][:, foot], label=name)
        axes[1].plot(t, schedule['contact_confidence'][:, foot], label=name+' near-ground confidence')
    axes[0].axhline(.001, ls=':', color='gray', label='1 mm geometry gate')
    axes[0].set_ylabel('Reference sole clearance (m)')
    axes[1].set_ylabel('Heuristic confidence')
    if meta['controller'] == 'force':
        limits = schedule['fractions']*meta['controller_config']['contact_force_max']
        for foot, name in enumerate(('Left', 'Right')):
            axes[2].plot(t, limits[:, foot], label=name+' force upper bound')
        axes[2].set_ylabel('Normal-force cap (N)')
    else:
        for i, label in enumerate(meta['action_layout']):
            axes[2].plot(t, schedule['fractions'][:, i], label=label)
        axes[2].set_ylabel('Cost authority')
    if args.diagnostics:
        with np.load(args.diagnostics, allow_pickle=False) as data:
            diag = dict(data)
        td = diag['time']
        error = np.abs(diag['foot_position_w'][..., 2]-diag['reference_foot_position_w'][..., 2])
        for foot, name in enumerate(('Left', 'Right')):
            axes[3].plot(td, error[:, foot], label=name+' foot z error (pre-step)')
        axes[3].axhline(.25, color='red', ls=':', label='Tracking threshold')
        end = float(diag['policy_steps_completed'])*float(diag['qp_dt'])
        for ax in axes:
            ax.axvline(end, ls='--', color='black', alpha=.7)
            if end < len(t)/float(schedule['fps']):
                ax.axvspan(end, len(t)/float(schedule['fps']), color='gray', alpha=.12)
        axes[3].set_title(f"Rollout: {int(diag['policy_steps_completed'])}/{len(t)} steps; "
                          f"complete_motion={bool(diag['complete_motion'])}; shaded portion was not executed")
    axes[3].set_ylabel('Foot height error (m)')
    axes[-1].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8, ncol=3)
    fig.suptitle(f"{meta['controller'].upper()} manual schedule — {meta['design']}")
    fig.tight_layout()
    output = args.output or args.schedule.with_suffix('.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
