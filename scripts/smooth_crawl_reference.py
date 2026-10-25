"""Modestly smooth a crawl, then project back onto joint/floor constraints."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from repair_crawl_reference import Geometry, project, ROOT


def metrics(motion):
    fps = float(motion['fps'])
    result = {}
    rot = Rotation.from_quat(motion['root_rot'])
    omega = (rot[1:]*rot[:-1].inv()).as_rotvec()*fps
    for name, velocity in (('joints', np.gradient(motion['dof_pos'], 1/fps, axis=0)),
            ('root_translation', np.gradient(motion['root_pos'], 1/fps, axis=0)), ('root_rotation', omega)):
        acceleration = np.gradient(velocity, 1/fps, axis=0)
        jerk = np.gradient(acceleration, 1/fps, axis=0)
        result[name] = dict(velocity_p99=float(np.percentile(abs(velocity), 99)),
            acceleration_p99=float(np.percentile(abs(acceleration), 99)),
            acceleration_max=float(abs(acceleration).max()), jerk_rms=float(np.sqrt(np.mean(jerk**2))))
    return result


def smooth(source, sigma, passes):
    motion = {k: v.copy() for k, v in source.items()}
    geometry = Geometry()
    rows = []
    for iteration in range(passes):
        xyz = gaussian_filter1d(motion['root_pos'], sigma, axis=0, mode='nearest')
        joints = gaussian_filter1d(motion['dof_pos'], sigma, axis=0, mode='nearest')
        quat = motion['root_rot'].copy()
        for i in range(1, len(quat)):
            if np.dot(quat[i-1], quat[i]) < 0:
                quat[i] *= -1
        quat = gaussian_filter1d(quat, sigma, axis=0, mode='nearest')
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        qpos = np.c_[xyz, quat[:, [3, 0, 1, 2]], joints]
        # Only root Z/joints move during projection; smoothed XY/orientation
        # remain fixed. The same four-support floor convention is retained here.
        for frame, base in enumerate(qpos):
            pose, delta, row = project(geometry, base, np.zeros(24), maxiter=100)
            if not row['success']:
                raise ValueError(f'Floor/joint projection failed at frame {frame}: {row}')
            qpos[frame] = pose
            rows.append(dict(row, frame=frame, iteration=iteration))
            if frame % 60 == 0:
                print(f'Projection pass {iteration+1}/{passes}: {frame}/{len(qpos)}', flush=True)
        motion['root_pos'], motion['root_rot'], motion['dof_pos'] = qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]
    before, after = metrics(source), metrics(motion)
    report = dict(sigma_frames=sigma, passes=passes, frames=len(qpos), fps=float(motion['fps']),
        before=before, after=after, all_floor_projections_pass=True,
        max_root_translation_change_m=float(np.linalg.norm(motion['root_pos']-source['root_pos'], axis=-1).max()),
        max_root_rotation_change_deg=float(np.rad2deg((Rotation.from_quat(motion['root_rot'])*Rotation.from_quat(source['root_rot']).inv()).magnitude()).max()),
        rms_joint_change_deg=float(np.rad2deg(np.sqrt(np.mean((motion['dof_pos']-source['dof_pos'])**2)))),
        max_joint_change_deg=float(np.rad2deg(abs(motion['dof_pos']-source['dof_pos'])).max()),
        note='Geometric projection only. Independent label/collision audit and dynamic force feasibility must follow; source recovery phases are not inferred by smoothing.')
    return motion, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'eval_data/reference_crawl_forward/reference.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sigma', type=float, default=2.5)
    parser.add_argument('--passes', type=int, default=1)
    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma <= 0 or args.passes < 1:
        parser.error('Positive finite smoothing width and pass count required')
    with np.load(args.input, allow_pickle=False) as archive:
        motion, report = smooth(dict(archive), args.sigma, args.passes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **motion)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
