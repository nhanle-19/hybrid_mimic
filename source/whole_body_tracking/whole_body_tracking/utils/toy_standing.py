"""Kinematic heel-contact fixture shared by PD and later WBC experiments."""
import math

import torch
from scipy.optimize import least_squares


def heel_up_pose(model, points, default_joints, joint_names, side='left', tilt_deg=15.):
    """Return model q/state with one heel grounded and the other sole flat.

    Preserve the tilted foot's nominal heel x/z by solving hip/knee pitch;
    changing only the ankle would push its heel through the floor.
    ``default_joints`` and returned simulator joints use ``joint_names`` order.
    """
    if side not in ('left', 'right') or not math.isfinite(tilt_deg) or not 0 <= tilt_deg <= 30:
        raise ValueError('Expected left/right side and a finite tilt in [0, 30] degrees')
    joints = default_joints.clone().reshape(1, -1)
    for prefix in ('Left', 'Right'):
        for name, value in (('Hip_Pitch', -.2), ('Hip_Roll', 0.), ('Hip_Yaw', 0.),
                            ('Knee_Pitch', .42), ('Ankle_Pitch', -.22), ('Ankle_Roll', 0.)):
            joints[:, joint_names.index(f'{prefix}_{name}')] = value
    joints[:, joint_names.index('Waist')] = 0.
    q = torch.zeros((1, model.nv+1), device=model.device, dtype=model.dtype)
    q[:, 6] = 1.
    q[:, 7:] = joints[:, [joint_names.index(n) for n in model.joint_names]]
    v = torch.zeros((1, model.nv), device=model.device, dtype=model.dtype)
    feet = ('left_foot_link', 'right_foot_link')
    index = 0 if side == 'left' else 1

    def vertices(state, i):
        frame = state['frames'][feet[i]]
        return points[i] @ frame['rotation'][0].T + frame['position'][0]

    state = model.state(q, v)
    q[:, 2] -= vertices(state, 1-index)[:, 2].min()
    state = model.state(q, v)
    heel = int(points[index, :, 0].argmin())
    target = vertices(state, index)[heel, [0, 2]].clone()
    prefix = side.title()
    ids = [7+model.joint_names.index(f'{prefix}_{name}')
           for name in ('Hip_Pitch', 'Knee_Pitch', 'Ankle_Pitch')]
    tilt = math.radians(tilt_deg)

    def residual(x):
        q[0, ids[0]], q[0, ids[1]], q[0, ids[2]] = float(x[0]), float(x[1]), -tilt-float(x[0]+x[1])
        return (vertices(model.state(q, v), index)[heel, [0, 2]]-target).cpu().numpy()

    result = least_squares(residual, [-.2, .42], bounds=([-.8, .05], [.4, 1.4]),
                           diff_step=1e-4, gtol=1e-12, ftol=1e-12, xtol=1e-12)
    error = residual(result.x)
    if not result.success or abs(error).max() > 1e-6:
        raise ValueError('Could not construct the requested grounded heel pose')
    state = model.state(q, v)
    for i in range(2):
        if float(vertices(state, i)[:, 2].min()) < -1e-6:
            raise ValueError('Constructed foot penetrates the floor')
    joints[:] = q[:, [7+model.joint_names.index(n) for n in joint_names]]
    return joints, q, state
