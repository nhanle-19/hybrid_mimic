"""Plot physical Atlas diagnostics; never label optimizer predictions as measurements."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('eval_data/atlas/diagnostics.npz'))
    p.add_argument('--output_dir', type=Path, default=Path('eval_data/atlas/plots'))
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        data = dict(archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if 'evaluation_completed' in data and not bool(data['evaluation_completed']):
        print(f"Partial evaluation: {data['failure_reason']}")
    for env in np.unique(data['env_id']):
        mask = data['env_id'] == env
        t = data['time'][mask]
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        for k, ax in enumerate(axes.flat):
            for key, label, style in [('desired_rate', 'Desired', '--'), ('predicted_rate', 'QP prediction', '-'), ('actual_rate', 'Measured-state finite difference', ':')]:
                ax.plot(t, data[key][mask, k], style, label=label)
            ax.set_ylabel(('Linear rate (N)' if k < 3 else 'Angular rate (Nm)')+' '+'xyz'[k%3])
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        complete = bool(data.get('evaluation_completed', True))
        fig.suptitle('Centroidal momentum rate'+(' — partial evaluation' if not complete else '')); fig.tight_layout()
        fig.savefig(args.output_dir/f'env{env}_momentum.png', dpi=160); plt.close(fig)
        fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
        for foot in range(2):
            axes[0, foot].plot(t, data['normal_forces'][mask, foot].sum(axis=-1), label='QP normal')
            axes[0, foot].plot(t, data['tangential_forces'][mask, foot].sum(axis=-1), label='QP sum point tangential magnitudes')
            if 'sensor_normal_force_w' in data:
                axes[0, foot].plot(t, data['sensor_normal_force_w'][mask, foot, 2], ':', label='Sensor normal z')
            axes[0, foot].set_ylabel('Force (N)'); axes[0, foot].set_title(str(data['foot_names'][foot]))
            utilization = data['friction_utilization'][mask, foot].copy()
            loaded = data['normal_forces'][mask, foot] > 1.
            utilization[~loaded] = np.nan
            maxima = np.max(np.where(loaded, utilization, -np.inf), axis=-1)
            maxima[~loaded.any(axis=-1)] = np.nan
            axes[1, foot].plot(t, maxima)
            axes[1, foot].axhline(1, color='r', linestyle='--'); axes[1, foot].set_ylabel('Friction utilization (points > 1 N)')
            axes[2, foot].plot(t, data['cop'][mask, foot, 0], label='CoP local x')
            axes[2, foot].plot(t, data['cop'][mask, foot, 1], label='CoP local y')
            axes[2, foot].set_ylabel('QP CoP (m)')
            axes[3, foot].step(t, data['active_contact'][mask, foot], where='post'); axes[3, foot].set_ylabel('Planned stance')
            axes[3, foot].set_xlabel('Time (s)')
            for ax in axes[:, foot]:
                ax.grid(alpha=.2)
                if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.output_dir/f'env{env}_contacts.png', dpi=160); plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, data['contact_acceleration_residual'][mask], label='QP')
        if 'actual_contact_acceleration_residual' in data:
            axes[0].plot(t, data['actual_contact_acceleration_residual'][mask], ':', label='Measured-state finite difference')
        axes[0].set_ylabel('Stance acceleration max'); axes[0].legend()
        axes[1].plot(t, data['torque_utilization'][mask].max(axis=-1)); axes[1].axhline(1, color='r', linestyle='--'); axes[1].set_ylabel('Maximum torque utilization')
        axes[2].plot(t, np.max(np.abs(data['inverse_dynamics_residual'][mask]), axis=-1)); axes[2].set_ylabel('QP inverse-dynamics residual max')
        axes[2].set_xlabel('Time (s)')
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(args.output_dir/f'env{env}_residuals.png', dpi=160); plt.close(fig)
    print(f'Saved Atlas plots to {args.output_dir}')


if __name__ == '__main__':
    main()
