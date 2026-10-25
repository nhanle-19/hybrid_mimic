"""Check reference acceleration against unilateral forces, friction and torques.

Unlike reference-state tracking QPs, these LPs require the reference's actual
finite-difference acceleration. A dissipative check additionally forbids
friction doing positive work on sliding material points. This is a necessary
instantaneous check, not a certificate of physical rollout or Coulomb sticking.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch

from crawl_force_schedule import ROOT, load_manual_schedule
from check_crawl_force_qp import controller_config, motion_state, torque_limits
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_gpu_control import select_state
from wbc_crawl_contact import CrawlContactGeometry


def audit(schedule_path, friction, output):
    torch.set_num_threads(2)
    spec, motion, _, _ = load_manual_schedule(schedule_path)
    model = WBCACCTorchModel()
    states = motion_state(model, motion)
    cfg = controller_config(spec)
    geometry = CrawlContactGeometry(ROOT/spec['geometry'])
    acceleration = np.gradient(states['v'].numpy(), 1/float(motion['fps']), axis=0)
    limits = torque_limits(model.joint_names)
    rays = np.array([[0, 0, -friction, friction], [friction, -friction, 0, 0], [1, 1, 1, 1]])
    rows, forces, speeds = [], [], []
    for i in range(len(acceleration)):
        state = select_state(states, slice(i, i+1))
        points, available = geometry.at(state, cfg)
        J = points['J'][0].numpy()
        velocity = np.einsum('cpij,j->cpi', J, states['v'][i].numpy())
        G = np.einsum('cpij,ik->jcpk', J, rays).reshape(model.nv, -1)
        enabled = np.repeat(available[0].numpy().reshape(-1), 4)
        G = G[:, enabled]
        region_caps = np.repeat(np.eye(8), 16, axis=1)[:, enabled]
        dynamic_force = states['M'][i].numpy()@acceleration[i]+states['bias'][i].numpy()
        static = model.state(states['q'][i:i+1], torch.zeros_like(states['v'][i:i+1]))
        support_force = static['bias'][0].numpy()
        # Positive friction work at any slipping point is inadmissible. The
        # zero vector is allowed; full kinetic-friction magnitude is not forced.
        power = np.zeros((32, 128))
        for point, vel in enumerate(velocity.reshape(-1, 3)):
            if np.linalg.norm(vel[:2]) > .02:
                power[point, point*4:point*4+4] = vel[:2]@rays[:2]
        power = power[:, enabled]
        results = {}
        for name, demanded, dissipative in [('static', support_force, False),
                ('dynamic', dynamic_force, False), ('dissipative', dynamic_force, True)]:
            upper = np.r_[G[6:], -G[6:], region_caps]
            bound = np.r_[demanded[6:]+limits, limits-demanded[6:], spec['force_max_n']]
            if dissipative:
                upper, bound = np.r_[upper, power], np.r_[bound, np.zeros(len(power))]
            solution = linprog(np.ones(G.shape[1])*1e-5, A_ub=upper, b_ub=bound,
                A_eq=G[:6], b_eq=demanded[:6], bounds=(0, None), method='highs')
            results[name] = bool(solution.success)
            if name == 'dissipative':
                coeff = np.zeros(128)
                if solution.success:
                    coeff[enabled] = solution.x
                forces.append(np.einsum('ik,cpk->cpi', rays, coeff.reshape(8, 4, 4)))
        rows.append(results)
        speeds.append(velocity)
        if i % 100 == 0:
            print('Dynamics check', i, '/', len(acceleration), flush=True)
    masks = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    report = dict(frames=len(rows), friction=friction,
        passed_frames={key: int(value.sum()) for key, value in masks.items()},
        failed_frames={key: np.flatnonzero(~value).tolist() for key, value in masks.items()},
        method=__doc__, motion_sha256=spec['motion_sha256'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'dynamic_feasibility.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output/'dynamic_feasibility.npz', **masks, forces=np.asarray(forces),
        point_velocity_w=np.asarray(speeds), acceleration=acceleration, time_s=np.arange(len(rows))/float(motion['fps']))
    print(json.dumps({k:v for k,v in report.items() if k!='failed_frames'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, default=ROOT/'configs/crawl_force_schedule.json')
    parser.add_argument('--friction', type=float, default=.2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.schedule, args.friction, args.output)
