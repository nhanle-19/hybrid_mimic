"""Compare recorded FORCE ablations, including slip and contact timing errors."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from check_crawl_force_qp import torque_limits
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_crawl_contact import CrawlContactGeometry
from wbc_contact_mapping import decode_contact_values_numpy


def analyze(path):
    d = dict(np.load(path/'diagnostics.npz'))
    report = json.loads((path/'validation.json').read_text())
    names = d['region_names'].tolist()
    measured = d['measured_forces'][:, [d['measured_region_names'].tolist().index(n) for n in names]]
    caps = d['normal_force_limits']
    force_util = np.divide(d['wrenches'][..., 2], caps, out=np.zeros_like(caps), where=caps > 0)
    torque_util = abs(d['torque'])/torque_limits(d['joint_names'])
    need_acceleration = 'acceleration' in d and 'contact_acceleration' not in d
    if 'point_velocity_w' not in d or need_acceleration:
        model = WBCACCTorchModel('cpu')
        geometry = CrawlContactGeometry(report.get('physics_geometry', ROOT/'eval_data/reference_crawl_forward/physics_asset/collision_geometry.npz'))
        cfg = SimpleNamespace(**report['controller_config'])
        velocities, accelerations = [], []
        for start in range(0, len(d['time_s']), 32):
            state = model.state(model.tensor(d['actual_q'][start:start+32]), model.tensor(d['actual_v'][start:start+32]))
            g, _ = geometry.at(state, cfg)
            velocities.append(torch.einsum('ncpij,nj->ncpi', g['J'], state['v']).numpy())
            if need_acceleration:
                accelerations.append((torch.einsum('ncpij,nj->ncpi', g['J'],
                    model.tensor(d['acceleration'][start:start+32]))+g['bias']).numpy())
        d['point_velocity_w'] = np.concatenate(velocities)
        if need_acceleration:
            d['contact_acceleration'] = np.concatenate(accelerations)
    velocity = d['point_velocity_w'][..., :2]
    force = d['point_forces']
    slipping = (np.linalg.norm(velocity, axis=-1) > .02) & (force[..., 2] > 5.)
    power = (force[..., :2]*velocity).sum(-1)
    loaded_normal_accel = d['contact_acceleration'][..., 2][force[..., 2]>5.] if 'contact_acceleration' in d else None
    off_load = np.maximum(measured[..., 2], 0)*(~d['contact_labels'])
    speed = report.get('reference_speed', 1.)
    delta = d['actual_q'][:, :3]-d['reference_q'][:, :3]
    samples = {}
    for time in (1., 2., 3., 4., 5.):
        if report.get('hold_frame') is not None or d['time_s'][-1]*speed < time:
            continue
        i = int(np.argmin(abs(d['time_s']*speed-time)))
        samples[str(time)] = dict(root_error_m=float(d['root_error'][i]), joint_rmse_deg=float(np.rad2deg(d['joint_rmse'][i])))
    return dict(name=path.name, directory=str(path.relative_to(ROOT)),
        simulation_seconds=report['simulated_seconds'], reference_seconds=report.get('reference_seconds', report['simulated_seconds']*speed),
        hold_frame=report.get('hold_frame'),
        stopped_on=report['failure_reason'] or 'prefix_completed', qp_failures=report['qp_failed_updates'],
        reference_speed=speed, friction=report.get('friction_coefficient', .6),
        force_max_n=report['controller_config']['contact_force_max'][0],
        posture_kp=report['controller_config']['posture_position_gain'], posture_kd=report['controller_config']['posture_velocity_gain'],
        max_root_error_m=float(d['root_error'].max()), final_root_delta_xyz_m=delta[-1].tolist(),
        max_joint_rmse_deg=float(np.rad2deg(d['joint_rmse'].max())),
        max_normal_force_utilization=float(force_util.max()),
        torque_saturation_by_joint={str(n):int((torque_util[:, i]>.99).sum()) for i,n in enumerate(d['joint_names']) if (torque_util[:,i]>.99).any()},
        inactive_predicted_force_max_n=float(np.max(abs(force[decode_contact_values_numpy(d['raw_actions']) <= 0]), initial=0)),
        off_label_measured_force_peak_n=float(off_load.max()), off_label_updates_above_1n=int((off_load>1.).any(-1).sum()),
        normal_force_prediction_rmse_n=float(np.sqrt(np.mean((d['wrenches'][...,2]-measured[...,2])**2))),
        tangential_force_prediction_rmse_n=float(np.sqrt(np.mean((d['wrenches'][...,:2]-measured[...,:2])**2))),
        loaded_slip_point_samples=int(slipping.sum()),
        predicted_friction_assists_slip_samples=int((slipping & (power>.1)).sum()),
        predicted_positive_slip_power_peak_w=float(np.max(np.where(slipping, power, 0))),
        loaded_normal_acceleration_median_abs_m_s2=float(np.median(abs(loaded_normal_accel))) if loaded_normal_accel is not None else None,
        loaded_normal_acceleration_peak_abs_m_s2=float(np.max(abs(loaded_normal_accel))) if loaded_normal_accel is not None else None,
        samples_at_reference_seconds=samples), d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/crawl_force_comparison')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    results, data = zip(*(analyze(p.resolve()) for p in args.directories))
    (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for r, d in zip(results, data):
        if r['hold_frame'] is not None:
            continue  # A static hold has elapsed time but no motion progress.
        time = d['time_s']*r['reference_speed']
        axes[0].plot(time, d['root_error']*100, label=r['name'])
        axes[1].plot(time, np.rad2deg(d['joint_rmse']))
    axes[0].set_ylabel('Root error (cm)')
    axes[0].axhline(15., color='gray', linestyle='--')
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Joint RMSE (deg)', xlabel='Reference progress (s)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png', dpi=130)
    for r in results:
        print(f"{r['name']}: reference={r['reference_seconds']:.3f}s, stop={r['stopped_on']}, "
              f"force utilization={r['max_normal_force_utilization']:.3f}, "
              f"slip-assisting predicted forces={r['predicted_friction_assists_slip_samples']}/{r['loaded_slip_point_samples']}")


if __name__ == '__main__':
    main()
