"""Physical consistency and provenance of the non-walking training mirrors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'retargeted_motion/training_list'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mirror = load_module('mirror_training', ROOT / 'scripts/mirror_training_motions.py')
kinematics = load_module('mirror_kinematics', ROOT /
                        'source/whole_body_tracking/whole_body_tracking/utils/wbc_acc_torch_model.py')


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def model_state(model, motion):
    root = motion['body_names'].tolist().index('Trunk')
    order = [motion['joint_names'].tolist().index(name) for name in model.joint_names]
    quat = motion['body_quat_w'][:, root][:, [1, 2, 3, 0]]
    rotation = Rotation.from_quat(quat).as_matrix()
    local = lambda key: np.einsum('nji,nj->ni', rotation, motion[key][:, root])
    q = np.concatenate((motion['body_pos_w'][:, root], quat, motion['joint_pos'][:, order]), axis=-1)
    v = np.concatenate((local('body_lin_vel_w'), local('body_ang_vel_w'), motion['joint_vel'][:, order]), axis=-1)
    return model.state(torch.as_tensor(q), torch.as_tensor(v))


def test_reflected_rotations_and_velocities_match_robot_kinematics():
    # Independent FK with arbitrary root orientations and joint rates catches
    # incorrect quaternion conventions and polar/axial velocity signs.
    model = kinematics.WBCACCTorchModel()
    rng = np.random.default_rng(42)
    q = rng.normal(0, .3, (12, model.nv + 1))
    q[:, 3:7] = Rotation.random(12, random_state=rng).as_quat()
    v = rng.normal(0, .5, (12, model.nv))
    state = model.state(torch.tensor(q), torch.tensor(v))
    frames = [state['frames'][name] for name in model.body_names]
    rotations = np.stack([frame['rotation'].numpy() for frame in frames], axis=1)
    quat = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(12, -1, 4)
    motion = dict(fps=np.asarray(50.), joint_names=np.asarray(model.joint_names),
                  body_names=np.asarray(model.body_names), joint_pos=q[:, 7:], joint_vel=v[:, 6:],
                  body_pos_w=np.stack([frame['position'].numpy() for frame in frames], axis=1),
                  body_quat_w=quat[..., [3, 0, 1, 2]],
                  body_lin_vel_w=np.stack([frame['velocity'][:, :3].numpy() for frame in frames], axis=1),
                  body_ang_vel_w=np.stack([frame['velocity'][:, 3:].numpy() for frame in frames], axis=1))
    reflected = mirror.mirror_motion(motion)
    expected = model_state(model, reflected)
    for i, name in enumerate(model.body_names):
        frame = expected['frames'][name]
        np.testing.assert_allclose(reflected['body_pos_w'][:, i], frame['position'], atol=1e-12)
        rotation = Rotation.from_quat(reflected['body_quat_w'][:, i][:, [1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(rotation, frame['rotation'], atol=1e-12)
        velocity = np.concatenate((reflected['body_lin_vel_w'][:, i], reflected['body_ang_vel_w'][:, i]), -1)
        np.testing.assert_allclose(velocity, frame['velocity'], atol=1e-12)


@pytest.mark.parametrize('stem', ['02_tilted_sway', '03_standing', '04_narrow_sway',
                                 '05_slide', '06_sole_rotation', '07_toe_rotation', '09_front_kick'])
def test_bundled_mirror_is_reversible_and_has_consistent_body_poses(stem):
    original = read_npz(DIRECTORY / f'{stem}.npz')
    reflected = read_npz(DIRECTORY / f'{stem}_mirrored.npz')
    restored = mirror.mirror_motion(reflected)
    assert restored.keys() == original.keys()
    for key in original:
        np.testing.assert_array_equal(restored[key], original[key], err_msg=key)
    # Validate every generated frame against the existing T1 model.
    model = kinematics.WBCACCTorchModel()
    for start in range(0, len(reflected['joint_pos']), 64):
        chunk = {key: value[start:start+64] if key.startswith(('joint_', 'body_'))
                 and not key.endswith('_names') else value for key, value in reflected.items()}
        state = model_state(model, chunk)
        for i, name in enumerate(chunk['body_names']):
            frame = state['frames'][name]
            np.testing.assert_allclose(chunk['body_pos_w'][:, i], frame['position'], atol=2e-6, rtol=0)
            rotation = Rotation.from_quat(chunk['body_quat_w'][:, i][:, [1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(rotation, frame['rotation'], atol=2e-6, rtol=0)


def test_generator_preserves_originals_excludes_walk_and_is_repeatable(tmp_path):
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    manifest['motions'] = [record for record in manifest['motions'] if 'mirrored_from' not in record]
    for record in manifest['motions']:
        source = DIRECTORY / record['file']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256']
        assert source.read_bytes() == (ROOT / record['collection_source']).read_bytes()
        shutil.copyfile(source, tmp_path / source.name)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    generated = mirror.generate(tmp_path)
    assert len(generated['motions']) == 15
    assert generated['total_frames'] == 6491
    assert not (tmp_path / '01_walk_mirrored.npz').exists()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    mirror.generate(tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    (tmp_path / '05_slide_mirrored.npz').unlink()
    mirror.generate(tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    for record in generated['motions']:
        payload = (tmp_path / record['file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
        assert payload == (DIRECTORY / record['file']).read_bytes()
