"""Add sagittal (left/right) mirrors of the active T1 clips, except walking.

Run with NumPy only: python scripts/mirror_training_motions.py
Original references remain byte-for-byte intact; reruns verify existing mirrors.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / 'retargeted_motion/training_list'
MODEL_PATH = ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/booster/t1/wbc_acc_dynamics.json'
EXCLUDED = {'01_walk.npz'}
SUFFIX = '_mirrored'


def opposite_joint(name):
    if name.startswith('Left_'):
        return 'Right_' + name[5:]
    if name.startswith('Right_'):
        return 'Left_' + name[6:]
    return name


def mirror_motion(values):
    """Reflect world Y and exchange paired limbs, preserving native array order.

    For S=diag(1,-1,1), positions/linear velocities use S, rotations use
    S R S, and angular velocities use -S. WXYZ quaternions become (w,-x,y,-z).
    T1 pitch axes retain their sign; roll/yaw axes reverse, including waist/head.
    """
    result = {key: np.array(value, copy=True) for key, value in values.items()}
    model = json.loads(MODEL_PATH.read_text())
    joints = {joint['name']: joint for joint in model['joints']}
    joint_names, body_names = result['joint_names'].tolist(), result['body_names'].tolist()
    if len(joint_names) != len(joints) or set(joint_names) != set(joints):
        raise ValueError('Expected the complete 23-joint T1 joint_names')
    if len(body_names) != len(model['bodies']) or set(body_names) != set(model['bodies']):
        raise ValueError('Expected the complete 24-body T1 body_names')
    joint_order = [joint_names.index(opposite_joint(name)) for name in joint_names]
    body_pairs = {joint['child']: joints[opposite_joint(name)]['child'] for name, joint in joints.items()}
    body_pairs[model['root']] = model['root']
    body_order = [body_names.index(body_pairs[name]) for name in body_names]
    joint_sign = [1 if joints[name]['axis'] == 'Y' else -1 for name in joint_names]
    frames = len(result['joint_pos'])
    shapes = dict(joint_pos=(frames, len(joints)), joint_vel=(frames, len(joints)),
                  body_pos_w=(frames, len(body_names), 3), body_quat_w=(frames, len(body_names), 4),
                  body_lin_vel_w=(frames, len(body_names), 3), body_ang_vel_w=(frames, len(body_names), 3))
    for key, shape in shapes.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all() or frames == 0:
            raise ValueError(f'Invalid motion array: {key}')
    fps = float(result['fps'].item())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Expected a positive finite frame rate')
    for key in ('joint_pos', 'joint_vel'):
        result[key] = result[key][:, joint_order] * np.asarray(joint_sign, dtype=result[key].dtype)
    for key, sign in (('body_pos_w', (1, -1, 1)), ('body_lin_vel_w', (1, -1, 1)),
                      ('body_ang_vel_w', (-1, 1, -1)), ('body_quat_w', (1, -1, 1, -1))):
        result[key] = result[key][:, body_order] * np.asarray(sign, dtype=result[key].dtype)
    # Mirror descriptive geometry as well as the arrays consumed by training.
    for key in ('tilted_side', 'slide_side', 'rotation_side'):
        if key in result:
            side = str(result[key].item())
            if side not in ('left', 'right'):
                raise ValueError(f'Unexpected {key}: {side}')
            result[key] = np.asarray('right' if side == 'left' else 'left')
    for key in ('com_pos_w', 'pivot_position_w'):
        if key in result:
            result[key][..., 1] *= -1
    for key in ('slide_displacement_m', 'yaw_angle_rad', 'support_root_shift_y_m', 'support_root_roll_deg'):
        if key in result:
            result[key] *= -1
    return result


def generate(directory=DEFAULT_DIRECTORY):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['motions']
    listed = {record['file'] for record in records}
    existing = {path.name for path in directory.glob('*.npz')}
    missing_originals = {name for name in listed - existing if not Path(name).stem.endswith(SUFFIX)}
    if len(listed) != len(records) or existing - listed or missing_originals:
        raise ValueError('Training manifest must list each motion exactly once; original files must exist')
    originals = [record for record in records if not Path(record['file']).stem.endswith(SUFFIX)]
    updated, pending = list(originals), []
    for record in originals:
        source = directory / record['file']
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_hash != record['sha256']:
            raise ValueError(f'Source hash differs from manifest: {source}')
        if source.name in EXCLUDED:
            continue
        with np.load(source, allow_pickle=False) as archive:
            mirrored = mirror_motion(archive)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **mirrored)
        payload = buffer.getvalue()
        destination = source.with_stem(source.stem + SUFFIX)
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f'Refusing to overwrite a differing mirror: {destination}')
        pending.append((destination, payload))
        updated.append(dict(file=destination.name, source=source.name, mirrored_from=source.name,
                            source_sha256=source_hash, sha256=hashlib.sha256(payload).hexdigest(),
                            frames=len(mirrored['joint_pos']), fps=float(mirrored['fps'].item())))
    if listed - {record['file'] for record in updated}:
        raise ValueError('Unexpected existing mirrored files in training selection')
    manifest['motions'] = sorted(updated, key=lambda record: record['file'])
    manifest['total_frames'] = sum(record['frames'] for record in updated)
    manifest['mirroring'] = dict(generator='scripts/mirror_training_motions.py', plane='world XZ (y = 0)',
                                 excluded=sorted(EXCLUDED), mirrored_clips=len(pending),
                                 transform='Swap left/right limbs and reflect poses, velocities and directional metadata.')
    # Finish all validation before writing any generated artifacts.
    for destination, payload in pending:
        if not destination.exists():
            destination.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    result = generate(args.directory)
    print(f"Training selection: {len(result['motions'])} clips, {result['total_frames']} frames; "
          f"{result['mirroring']['mirrored_clips']} non-walking mirrors verified.")
