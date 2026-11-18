"""Check motion timing and named joints at the SPIDER import/export boundary."""
from pathlib import Path
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
pytest.importorskip('pxr')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dsms_t1_model import ROOT, build_model, load_reference
from retarget_force2_spider import resample_reference, export_training


@pytest.fixture(scope='module')
def motion(tmp_path_factory):
    model = mujoco.MjModel.from_xml_path(str(build_model(tmp_path_factory.mktemp('spider_model'))))
    template, states, fps = load_reference(ROOT / 'retargeted_motion/training_list/09_front_kick.npz', model)
    return model, template, states, fps


def test_resampling_keeps_every_source_frame_and_exact_duration(motion):
    model, _, states, fps = motion
    qpos, qvel = resample_reference(model, states, fps, .002)
    assert len(qpos) == (len(states) - 1) * 10 + 1
    np.testing.assert_allclose(qpos[::10], states[:, :model.nq], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1., atol=1e-12)
    assert qvel.shape == (len(qpos), model.nv)
    assert (len(qpos) - 1) * .002 == pytest.approx((len(states) - 1) / fps)
    assert (len(qpos) - 1) % 50 != 0  # Final control block must be shorter.


def test_training_export_preserves_named_joints_and_body_origins(motion, tmp_path):
    model, template, states, fps = motion
    ids = [0, 40, len(states) - 1]
    path = tmp_path / 'candidate.npz'
    export_training(path, model, states[ids, :model.nq], states[ids, model.nq:], template, fps)
    out = dict(np.load(path))
    assert out['joint_names'].tolist() == template['joint_names'].tolist()
    assert out['body_names'].tolist() == template['body_names'].tolist()
    np.testing.assert_allclose(out['joint_pos'], template['joint_pos'][ids], atol=1e-12)
    np.testing.assert_allclose(out['body_pos_w'], template['body_pos_w'][ids], atol=2e-6)
    assert out['body_lin_vel_w'].shape == out['body_pos_w'].shape
    assert np.isfinite(out['body_lin_vel_w']).all()
