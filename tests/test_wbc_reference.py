"""Reference preparation preserves kinematics and avoids hidden reset offsets."""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_reference import prepare_wbc_reference
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_contact_policy import sole_vertices, FEET


def test_grounded_reference_matches_joint_limits_fk_and_velocities():
    torch.set_num_threads(2)
    model = WBCACCTorchModel()
    n = 5
    joints = np.zeros((n, 23))
    knee = model.joint_names.index('Left_Knee_Pitch')
    joints[:, knee] = [-.1, 0., .1, .2, .3]
    limits = np.tile([-3., 3.], (23, 1)); limits[knee, 0] = 0.
    root = np.tile([0., 0., .5], (n, 1)); quat = np.tile([1., 0., 0., 0.], (n, 1))
    result, report = prepare_wbc_reference(joints, root, quat, model.joint_names, model.body_names, limits, 50.)
    assert report['joint_values_clamped'] == 1
    assert report['initial_root_lift_m'] > 0
    assert result['joint_pos'][1, knee] == 0.  # No soft-limit knee bend.
    root_index = model.body_names.index('Trunk')
    q = np.concatenate((result['body_pos_w'][:, root_index], result['body_quat_w'][:, root_index, [1,2,3,0]], result['joint_pos']), -1)
    state = model.state(torch.tensor(q), torch.zeros((n,29), dtype=torch.float64))
    for i, name in enumerate(model.body_names):
        np.testing.assert_allclose(state['frames'][name]['position'], result['body_pos_w'][:,i], atol=1e-12)
    heights=[]
    for i, foot in enumerate(FEET):
        frame=state['frames'][foot]
        heights.append((sole_vertices()[i]@frame['rotation'].transpose(-1,-2)+frame['position'][:,None])[...,2])
    heights=torch.cat(heights,-1)
    assert heights.min() >= -1e-12
    torch.testing.assert_close(heights.amin(-1), torch.zeros(n,dtype=torch.float64), atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['joint_vel'], np.gradient(result['joint_pos'], .02, axis=0))
    np.testing.assert_allclose(result['body_lin_vel_w'], np.gradient(result['body_pos_w'], .02, axis=0))
    rotations=Rotation.from_quat(result['body_quat_w'][..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,-1,3,3)
    np.testing.assert_allclose(rotations[:,root_index], np.tile(np.eye(3),(n,1,1)),atol=1e-12)


def test_airborne_reference_is_not_pulled_to_ground():
    model=WBCACCTorchModel(); n=3
    root=np.tile([0.,0.,2.],(n,1));root[:,0]=[0.,.01,.02]
    result,report=prepare_wbc_reference(np.zeros((n,23)),root,np.tile([1.,0.,0.,0.],(n,1)),
        model.joint_names,model.body_names,np.tile([-3.,3.],(23,1)),50.)
    assert report['max_root_lift_m']==0.
    np.testing.assert_allclose(result['body_pos_w'][:,model.body_names.index('Trunk')],root)
