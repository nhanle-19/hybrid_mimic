"""Prepare a kinematically consistent WBC reference against the actual sole geometry."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    from .wbc_acc_torch_model import WBCACCTorchModel
    from .wbc_acc_contact_policy import sole_vertices, FEET
except ImportError:
    from wbc_acc_torch_model import WBCACCTorchModel
    from wbc_acc_contact_policy import sole_vertices, FEET


def prepare_wbc_reference(joint_pos, root_pos, root_quat, joint_names, body_names,
                          joint_limits, fps, ground_height=0.):
    """Clamp to supplied physical limits, remove floor penetration, rebuild FK/velocities.

    Preparation runs once on CPU, independently of the training device. Only
    penetrating frames are raised; airborne frames and relative foot placement
    are preserved. No contact timing, foot-leveling, or policy action is imposed.
    Quaternions in the input/output archive use WXYZ.
    """
    model = WBCACCTorchModel('cpu')
    joints = torch.as_tensor(np.asarray(joint_pos), dtype=torch.float64).clone()
    limits = torch.as_tensor(np.asarray(joint_limits), dtype=torch.float64)
    if not np.isfinite(fps) or fps <= 0 or len(joints) < 1:
        raise ValueError('WBC reference needs at least one frame and a positive frame rate')
    if not bool(torch.isfinite(joints).all() & torch.isfinite(limits).all()):
        raise ValueError('Non-finite WBC reference or joint limits')
    corrected = joints.clamp(limits[:, 0], limits[:, 1])
    order = [list(joint_names).index(name) for name in model.joint_names]
    quat = torch.as_tensor(np.asarray(root_quat), dtype=torch.float64)[:, [1, 2, 3, 0]]
    pos = torch.as_tensor(np.asarray(root_pos), dtype=torch.float64)
    q = torch.cat((pos, quat, corrected[:, order]), -1)
    points = sole_vertices()
    positions, rotations, shifts = [], [], []
    for start in range(0, len(q), 64):
        chunk = q[start:start+64].clone()
        velocity = torch.zeros((len(chunk), 29), dtype=torch.float64)
        state = model.state(chunk, velocity)
        heights = torch.cat([(points[i]@state['frames'][foot]['rotation'].transpose(-1, -2)
                              +state['frames'][foot]['position'][:, None])[..., 2]
                             for i, foot in enumerate(FEET)], -1)
        shift = (ground_height-heights.amin(-1)).clamp_min(0.)
        chunk[:, 2] += shift
        state = model.state(chunk, velocity)
        positions.append(torch.stack([state['frames'][name]['position'] for name in body_names], 1).numpy())
        rotations.append(torch.stack([state['frames'][name]['rotation'] for name in body_names], 1).numpy())
        shifts.append(shift.numpy())
    positions, rotations, shifts = np.concatenate(positions), np.concatenate(rotations), np.concatenate(shifts)
    quaternions = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(len(q), len(body_names), 4)
    # Use short relative rotations for world-frame angular velocity, including
    # endpoints, rather than differentiating sign-ambiguous quaternion entries.
    previous = np.maximum(np.arange(len(q))-1, 0)
    following = np.minimum(np.arange(len(q))+1, len(q)-1)
    relative = rotations[following]@rotations[previous].swapaxes(-1, -2)
    angular = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(len(q), len(body_names), 3)
    angular *= (fps/np.maximum(following-previous, 1))[:, None, None]
    derivative = lambda value: np.gradient(value, 1./fps, axis=0) if len(q) > 1 else np.zeros_like(value)
    result = dict(joint_pos=corrected.numpy(), joint_vel=derivative(corrected.numpy()),
                  body_pos_w=positions, body_quat_w=quaternions[..., [3, 0, 1, 2]],
                  body_lin_vel_w=derivative(positions), body_ang_vel_w=angular)
    report = dict(version=1, joint_values_clamped=int((corrected-joints).abs().gt(1e-7).sum()),
                  max_joint_correction_rad=float((corrected-joints).abs().max()),
                  max_root_lift_m=float(shifts.max()), initial_root_lift_m=float(shifts[0]))
    return result, report
