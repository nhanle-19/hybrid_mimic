"""Small hand-path corrections during reviewed source stationary-hand phases.

This retains root path/timing and the grounded surface constraints. Corrections
are capped at 25 mm and blended across phase boundaries; it does not assert
that a stationary source marker proves a load-bearing contact.
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from repair_crawl_reference import Geometry, project
from smooth_crawl_reference import metrics


PHASES = {
    'left_hand': [[0,85], [144,204], [233,288], [315,380], [408,467], [497,525], [555,598]],
    'right_hand': [[0,68], [90,160], [190,249], [277,335], [366,424], [457,482], [498,598]],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-correction', type=float, default=.025)
    args = parser.parse_args()
    source = dict(np.load(args.input, allow_pickle=False))
    geometry = Geometry()
    qpos = np.c_[source['root_pos'], source['root_rot'][:, [3,0,1,2]], source['dof_pos']]
    points, bodies, positions, targets = {}, {}, {}, {}
    for name, phases in PHASES.items():
        geom, vertices = geometry.groups[name][0]
        points[name] = Rotation.from_quat(geometry.model.geom_quat[geom][[1,2,3,0]]).apply(vertices.mean(0))+geometry.model.geom_pos[geom]
        bodies[name] = int(geometry.model.geom_bodyid[geom])
        positions[name] = []
        for q in qpos:
            geometry.data.qpos[:] = q
            mj.mj_forward(geometry.model, geometry.data)
            b = bodies[name]
            positions[name].append(geometry.data.xpos[b]+geometry.data.xmat[b].reshape(3,3)@points[name])
        positions[name] = np.asarray(positions[name])
        correction = np.zeros((len(qpos), 2))
        for start, end in phases:
            if end >= len(qpos):
                continue
            window = positions[name][start:end+1, :2]
            anchor = np.median(window[:min(12,len(window))], axis=0)
            delta = anchor-window
            delta *= np.minimum(1., args.max_correction/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[:,None]
            blend = np.minimum(np.clip(np.arange(len(window))/8.,0,1), np.clip(np.arange(len(window))[::-1]/8.,0,1))
            correction[start:end+1] = delta*blend[:,None]
        correction = gaussian_filter1d(correction, 2., axis=0)
        targets[name] = positions[name][:,:2]+correction
    output = qpos.copy()
    for i, base in enumerate(qpos):
        anchors = [(bodies[name], points[name], targets[name][i]) for name in PHASES]
        q, _, row = project(geometry, base, np.zeros(24), hand_targets=anchors)
        if not row['success']:
            raise ValueError(f'Anchor projection failed at frame {i}: {row}')
        output[i] = q
        if i % 60 == 0:
            print('Hand phase projection',i,'/',len(qpos),flush=True)
    result = dict(source)
    result.update(root_pos=output[:,:3],root_rot=output[:,[4,5,6,3]],dof_pos=output[:,7:])
    report = dict(phases=PHASES, max_requested_xy_correction_m=args.max_correction,
        before=metrics(source),after=metrics(result),
        max_joint_change_deg=float(np.rad2deg(abs(result['dof_pos']-source['dof_pos'])).max()),
        rms_joint_change_deg=float(np.rad2deg(np.sqrt(np.mean((result['dof_pos']-source['dof_pos'])**2)))),
        source_review='Intervals reviewed against source finger-marker displacement/speed; bounded soft hand-path correction, not a force measurement.',
        physical_execution_verified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
