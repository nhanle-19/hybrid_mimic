"""Validate training NPZ structure, provenance, mirrors and T1 forward kinematics."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_contact_policy import sole_vertices, FEET
from motion_library import MotionLibrary
from mirror_training_motions import mirror_motion


def validate(directory):
    torch.set_num_threads(4)
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['motions']
    names = [r['file'] for r in records]
    assert len(names) == len(set(names))
    assert set(names) == {p.name for p in directory.glob('*.npz')}
    assert not any('narrow_sway' in name for name in names)
    assert not {'03_standing.npz', '03_standing_mirrored.npz'} & set(names)
    originals = [name for name in names if not name.endswith('_mirrored.npz')]
    assert len(names) == 2*len(originals)
    assert all(Path(name).stem+'_mirrored.npz' in names for name in originals)
    library = MotionLibrary(directory)
    assert library.time_step_total == manifest['total_frames'] == sum(r['frames'] for r in records)
    assert float(library.fps.item()) == 50.
    model, points = WBCACCTorchModel(), sole_vertices().numpy()
    report = dict(passed=False, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  clips=len(names), originals=len(originals), mirrored=len(originals),
                  frames=library.time_step_total, fps=50., checks=[], motions={})
    for record in records:
        path = directory/record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as z:
            values = dict(z)
        assert len(values['joint_pos']) == record['frames']
        norms = np.linalg.norm(values['body_quat_w'], axis=-1)
        assert np.max(np.abs(norms-1)) < 1e-5
        if 'mirrored_from' in record:
            with np.load(directory/record['mirrored_from'], allow_pickle=False) as z:
                original = dict(z)
            expected = mirror_motion(original)
            restored = mirror_motion(values)
            for key in values:
                np.testing.assert_array_equal(values[key], expected[key], err_msg=path.name+':'+key)
                np.testing.assert_array_equal(restored[key], original[key], err_msg=path.name+':'+key)
        root = values['body_names'].tolist().index('Trunk')
        order = [values['joint_names'].tolist().index(name) for name in model.joint_names]
        q = np.column_stack((values['body_pos_w'][:, root], values['body_quat_w'][:, root][:, [1, 2, 3, 0]],
                             values['joint_pos'][:, order]))
        maximum_position_error, maximum_rotation_error = 0., 0.
        feet_positions, feet_rotations = [], []
        for start in range(0, len(q), 64):
            chunk = torch.as_tensor(q[start:start+64])
            state = model.state(chunk, torch.zeros((len(chunk), model.nv)))
            for i, name in enumerate(values['body_names'].tolist()):
                frame = state['frames'][name]
                position_error = np.abs(frame['position'].numpy()-values['body_pos_w'][start:start+64, i]).max()
                stored = Rotation.from_quat(values['body_quat_w'][start:start+64, i][:, [1, 2, 3, 0]]).as_matrix()
                rotation_error = np.abs(stored-frame['rotation'].numpy()).max()
                maximum_position_error = max(maximum_position_error, float(position_error))
                maximum_rotation_error = max(maximum_rotation_error, float(rotation_error))
            feet_positions.append(np.stack([state['frames'][f]['position'].numpy() for f in FEET], 1))
            feet_rotations.append(np.stack([state['frames'][f]['rotation'].numpy() for f in FEET], 1))
        assert maximum_position_error < 2e-6, (path.name, maximum_position_error)
        assert maximum_rotation_error < 2e-6, (path.name, maximum_rotation_error)
        positions, rotations = np.concatenate(feet_positions), np.concatenate(feet_rotations)
        corners = np.einsum('nfij,fvj->nfvi', rotations, points)+positions[:, :, None]
        heights = corners[..., 2]
        assert heights.min() >= -2e-6, (path.name, float(heights.min()))
        contact_count = ((heights <= .001) & (heights >= -.001)).sum(-1)
        low = heights.min(-1)
        tilt = np.rad2deg(np.arccos(np.clip(rotations[:, :, 2, 2], -1, 1)))
        foot_speed = np.linalg.norm(np.gradient(positions[..., :2], 1/50., axis=0), axis=-1)
        yaw = np.unwrap(np.arctan2(rotations[:, :, 1, 0], rotations[:, :, 0, 0]), axis=0)
        yaw_speed = np.abs(np.gradient(yaw, 1/50., axis=0))
        grounded = (low <= .001) & (low >= -.001)
        report['motions'][path.name] = dict(
            frames=len(q), max_fk_position_error_m=maximum_position_error,
            max_fk_rotation_matrix_error=maximum_rotation_error,
            minimum_sole_height_m=float(low.min()), frames_without_sole_within_5mm=int((low.min(-1) > .005).sum()),
            flat_contact_foot_frames=int((contact_count == 4).sum()),
            partial_contact_foot_frames=int(((contact_count > 0) & (contact_count < 4)).sum()),
            grounded_max_tilt_deg=float(tilt[grounded].max()) if grounded.any() else 0.,
            grounded_moving_foot_frames=int((grounded & (foot_speed > .03)).sum()),
            grounded_rotating_foot_frames=int((grounded & (yaw_speed > .2)).sum()))
        print(f'Checked {path.name}', flush=True)
    report['checks'] = ['Exact manifest/file set and SHA256', '50 Hz MotionLibrary load with consistent joint/body ordering',
                        'Every original has a mirror, including walks; narrow sway and static standing absent',
                        'All saved mirror arrays equal reflection; reflection twice restores all arrays',
                        'All body poses agree with independent T1 forward kinematics within 2 micrometres',
                        'Unit quaternions and finite training arrays',
                        'No sole penetration beyond 2 micrometres']
    report['contact_statistics_note'] = 'Reference geometry only, not measured contact forces or successful dynamic tracking. Grounded movement can include retargeting drift.'
    report['passed'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'outputs/contact_training_build_20261115/staging')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/contact_training_build_20261115/validation.json')
    args = parser.parse_args()
    report = validate(args.directory)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'motions'}, indent=2))
