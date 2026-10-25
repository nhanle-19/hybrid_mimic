"""Kinematic standing, sway, and grounded foot-motion fixtures for WBC experiments."""
import math

import numpy as np
import torch
from scipy.optimize import least_squares

try:
    from .wbc_acc_contact_policy import FEET
    from .wbc_reference import prepare_wbc_reference
except ImportError:
    from wbc_acc_contact_policy import FEET
    from wbc_reference import prepare_wbc_reference


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


def sway_reference(model, points, default_joints, joint_names, body_names, joint_limits,
                   side='left', tilt_deg=15., seconds=10., fps=50., com_sway_m=.020,
                   torso_roll_deg=2.4, period_s=4., stance_widen_m=0.):
    """Shift the actual mass-weighted CoM and lean the torso with fixed foot poses.

    Positive sway moves toward world +Y (left); negative root roll leans left.
    Root translation and the twelve leg joints solve both six-dimensional foot
    constraints and the three-dimensional CoM target. CoM x/z remain constant.
    The first/last 0.5 s hold the neutral pose, with one-second quintic ramps.
    ``stance_widen_m`` adds to the total foot separation, moving each foot
    outward by half that distance while preserving its height and rotation.
    This constructs a kinematic reference, not a dynamic balance guarantee.
    """
    if not all(math.isfinite(x) for x in (seconds, fps, com_sway_m, torso_roll_deg, period_s, stance_widen_m)):
        raise ValueError('Sway parameters must be finite')
    if seconds < 4. or fps < 10. or period_s < 2.:
        raise ValueError('Sway needs seconds >= 4, fps >= 10, and period_s >= 2')
    if not 0 <= com_sway_m <= .06 or not 0 <= torso_roll_deg <= 10.:
        raise ValueError('Expected com_sway_m in [0, 0.06] and torso_roll_deg in [0, 10]')
    if not 0 <= stance_widen_m <= .10:
        raise ValueError('Expected stance_widen_m in [0, 0.10]')
    if model.device.type != 'cpu':
        raise ValueError('Construct the sway reference with a CPU model')
    _, neutral, initial = heel_up_pose(model, points, default_joints, joint_names, side, tilt_deg)
    target_feet = {}
    for foot, direction in zip(FEET, (1., -1.)):
        target_feet[foot] = dict(position=initial['frames'][foot]['position'].clone(),
                                 rotation=initial['frames'][foot]['rotation'])
        target_feet[foot]['position'][:, 1] += direction * stance_widen_m / 2.
    count = math.ceil(seconds * fps)
    time = torch.arange(count, dtype=model.dtype) / fps
    def smoothstep(x):
        x = x.clamp(0., 1.)
        return x**3 * (10. - 15.*x + 6.*x*x)
    envelope = smoothstep(time - .5) * smoothstep(time[-1] - time - .5)
    wave = envelope * torch.sin(2.*math.pi*time / period_s)
    roll = -math.radians(torso_roll_deg) * wave
    target_com = initial['com'].repeat(count, 1)
    target_com[:, 1] += com_sway_m * wave
    leg_ids = [model.joint_names.index(f'{prefix}_{name}')
               for prefix in ('Left', 'Right')
               for name in ('Hip_Pitch', 'Hip_Roll', 'Hip_Yaw', 'Knee_Pitch', 'Ankle_Pitch', 'Ankle_Roll')]
    q_ids = [0, 1, 2] + [7+i for i in leg_ids]
    v_ids = [0, 1, 2] + [6+i for i in leg_ids]
    poses = []
    for start in range(0, count, 64):
        q = neutral.repeat(min(64, count-start), 1)
        q[:, 3] = torch.sin(roll[start:start+len(q)] / 2.)
        q[:, 6] = torch.cos(roll[start:start+len(q)] / 2.)
        zero = torch.zeros((len(q), model.nv), dtype=model.dtype)
        for iteration in range(40):
            state = model.state(q, zero)
            errors, jacobians = [], []
            for foot in FEET:
                frame, target = state['frames'][foot], target_feet[foot]
                relative = target['rotation'] @ frame['rotation'].transpose(-1, -2)
                rotation_error = .5 * torch.stack((relative[:, 2, 1]-relative[:, 1, 2],
                    relative[:, 0, 2]-relative[:, 2, 0], relative[:, 1, 0]-relative[:, 0, 1]), -1)
                errors.extend((target['position']-frame['position'], rotation_error))
                jacobians.append(frame['J'][:, :, v_ids].clone())
            errors.append(target_com[start:start+len(q)]-state['com'])
            jacobians.append((state['Ag'][:, :3, v_ids] / state['mass']).clone())
            error = torch.cat(errors, -1)
            if float(error.abs().max()) < 1e-9:
                break
            # Generalized root velocity is local; the solved translations are world XYZ.
            jacobian = torch.cat(jacobians, 1)
            root_rotation = state['frames'][model.body_names[0]]['rotation']
            jacobian[:, :, :3] = jacobian[:, :, :3] @ root_rotation.transpose(-1, -2)
            transpose = jacobian.transpose(-1, -2)
            step = torch.linalg.solve(transpose@jacobian + 1e-10*torch.eye(15, dtype=model.dtype),
                                      (transpose@error[..., None])).squeeze(-1)
            # Limit each Newton update without clipping the requested trajectory.
            step /= (step.abs().amax(-1, keepdim=True)/.1).clamp_min(1.)
            q[:, q_ids] += step
        else:
            raise ValueError('Sway IK did not converge; reduce CoM shift, torso roll, or stance widening')
        poses.append(q)
    q = torch.cat(poses)
    joints = q[:, [7+model.joint_names.index(n) for n in joint_names]]
    limits = torch.as_tensor(joint_limits, dtype=model.dtype).reshape(len(joint_names), 2)
    if not bool(torch.isfinite(limits).all() & (limits[:, 0] <= limits[:, 1]).all()):
        raise ValueError('Expected finite ordered joint limits')
    if not bool(((joints >= limits[:, 0]) & (joints <= limits[:, 1])).all()):
        raise ValueError('Sway exceeds joint limits; reduce CoM shift, torso roll, or stance widening')
    values, report = prepare_wbc_reference(joints.numpy(), q[:, :3].numpy(), q[:, [6, 3, 4, 5]].numpy(),
        joint_names, body_names, limits.numpy(), fps)
    if report['joint_values_clamped'] or report['max_root_lift_m'] > 1e-7:
        raise ValueError('Sway reference required a joint or ground correction')
    values.update(fps=np.array(fps), joint_names=np.asarray(joint_names), body_names=np.asarray(body_names),
                  tilted_side=side, tilt_deg=tilt_deg, motion_type='sway', com_sway_m=com_sway_m,
                  torso_roll_deg=torso_roll_deg, period_s=period_s, com_pos_w=target_com.numpy(),
                  stance_widen_m=stance_widen_m,
                  stance_width_m=float(target_feet[FEET[0]]['position'][0, 1]
                                       - target_feet[FEET[1]]['position'][0, 1]))
    return values


def slide_reference(model, points, default_joints, joint_names, body_names, joint_limits,
                    side='left', seconds=10., fps=50., slide_amplitude_m=.020, period_s=4.):
    """Slide one flat foot along world Y while the root and other leg stay fixed.

    Only the sliding leg's six joints change. The torso stays upright and the
    upper body keeps its normal standing pose. Both soles remain at floor height.
    The sinusoidal displacement has 0.5-second endpoint holds and one-second
    quintic ramps. This is a kinematic reference, not a guarantee of frictional
    feasibility or controller tracking.
    """
    if side not in ('left', 'right'):
        raise ValueError('Expected left/right slide side')
    if not all(math.isfinite(x) for x in (seconds, fps, slide_amplitude_m, period_s)):
        raise ValueError('Slide parameters must be finite')
    if seconds < 4. or fps < 10. or period_s < 2.:
        raise ValueError('Slide needs seconds >= 4, fps >= 10, and period_s >= 2')
    if not 0 < slide_amplitude_m <= .05:
        raise ValueError('Expected slide_amplitude_m in (0, 0.05]')
    if model.device.type != 'cpu':
        raise ValueError('Construct the slide reference with a CPU model')
    _, neutral, initial = heel_up_pose(model, points, default_joints, joint_names, side, 0.)
    foot = FEET[0 if side == 'left' else 1]
    target_rotation = initial['frames'][foot]['rotation']
    count = math.ceil(seconds * fps)
    time = torch.arange(count, dtype=model.dtype) / fps

    def smoothstep(x):
        x = x.clamp(0., 1.)
        return x**3 * (10. - 15.*x + 6.*x*x)

    envelope = smoothstep(time - .5) * smoothstep(time[-1] - time - .5)
    displacement = slide_amplitude_m * envelope * torch.sin(2.*math.pi*time / period_s)
    target_position = initial['frames'][foot]['position'].repeat(count, 1)
    target_position[:, 1] += displacement
    leg_ids = [model.joint_names.index(f'{side.title()}_{name}')
               for name in ('Hip_Pitch', 'Hip_Roll', 'Hip_Yaw', 'Knee_Pitch', 'Ankle_Pitch', 'Ankle_Roll')]
    q_ids, v_ids = [7+i for i in leg_ids], [6+i for i in leg_ids]
    poses = []
    for start in range(0, count, 64):
        q = neutral.repeat(min(64, count-start), 1)
        zero = torch.zeros((len(q), model.nv), dtype=model.dtype)
        for iteration in range(40):
            frame = model.state(q, zero)['frames'][foot]
            relative = target_rotation @ frame['rotation'].transpose(-1, -2)
            rotation_error = .5 * torch.stack((relative[:, 2, 1]-relative[:, 1, 2],
                relative[:, 0, 2]-relative[:, 2, 0], relative[:, 1, 0]-relative[:, 0, 1]), -1)
            error = torch.cat((target_position[start:start+len(q)]-frame['position'], rotation_error), -1)
            if float(error.abs().max()) < 1e-9:
                break
            jacobian = frame['J'][:, :, v_ids]
            transpose = jacobian.transpose(-1, -2)
            step = torch.linalg.solve(transpose@jacobian + 1e-10*torch.eye(6, dtype=model.dtype),
                                      transpose@error[..., None]).squeeze(-1)
            step /= (step.abs().amax(-1, keepdim=True)/.1).clamp_min(1.)
            q[:, q_ids] += step
        else:
            raise ValueError('Slide IK did not converge; reduce slide amplitude')
        poses.append(q)
    q = torch.cat(poses)
    joints = q[:, [7+model.joint_names.index(n) for n in joint_names]]
    limits = torch.as_tensor(joint_limits, dtype=model.dtype).reshape(len(joint_names), 2)
    if not bool(torch.isfinite(limits).all() & (limits[:, 0] <= limits[:, 1]).all()):
        raise ValueError('Expected finite ordered joint limits')
    if not bool(((joints >= limits[:, 0]) & (joints <= limits[:, 1])).all()):
        raise ValueError('Slide exceeds joint limits; reduce slide amplitude')
    values, report = prepare_wbc_reference(joints.numpy(), q[:, :3].numpy(), q[:, [6, 3, 4, 5]].numpy(),
        joint_names, body_names, limits.numpy(), fps)
    if report['joint_values_clamped'] or report['max_root_lift_m'] > 1e-7:
        raise ValueError('Slide reference required a joint or ground correction')
    values.update(fps=np.array(fps), joint_names=np.asarray(joint_names), body_names=np.asarray(body_names),
                  motion_type='foot_slide', slide_side=side, tilt_deg=0.,
                  slide_amplitude_m=slide_amplitude_m, period_s=period_s,
                  slide_displacement_m=displacement.numpy())
    return values


def rotation_reference(model, points, default_joints, joint_names, body_names, joint_limits,
                       side='left', seconds=10., fps=50., yaw_amplitude_deg=15., period_s=4.,
                       support_shift_m=.08, root_lower_m=.02, support_roll_deg=5., toe_pitch_deg=0.):
    """Yaw a foot about its sole centre or grounded toe edge.

    Start in a supported pose shifted toward the other leg, as in the selected
    sliding fixture. The root and upper body remain fixed throughout the clip.
    With positive ``toe_pitch_deg``, the heel is raised and the toe-edge midpoint
    stays fixed on the floor. Otherwise all sole vertices remain grounded and
    the sole centre is fixed. The foot-link origin follows the pivot's arc.
    This is kinematics only: full-friction pivoting needs a physical replay.
    """
    if side not in ('left', 'right'):
        raise ValueError('Expected left/right rotation side')
    parameters = (seconds, fps, yaw_amplitude_deg, period_s, support_shift_m,
                  root_lower_m, support_roll_deg, toe_pitch_deg)
    if not all(math.isfinite(x) for x in parameters):
        raise ValueError('Rotation parameters must be finite')
    if seconds < 4. or fps < 10. or period_s < 2.:
        raise ValueError('Rotation needs seconds >= 4, fps >= 10, and period_s >= 2')
    if not 0 < yaw_amplitude_deg <= 30.:
        raise ValueError('Expected yaw_amplitude_deg in (0, 30]')
    if not 0 <= toe_pitch_deg <= 30.:
        raise ValueError('Expected toe_pitch_deg in [0, 30]')
    if not (0 <= support_shift_m <= .12 and 0 <= root_lower_m <= .06
            and 0 <= support_roll_deg <= 10.):
        raise ValueError('Support shift, lowering, or roll is outside the supported range')
    if model.device.type != 'cpu':
        raise ValueError('Construct the rotation reference with a CPU model')
    _, neutral, initial = heel_up_pose(model, points, default_joints, joint_names, side, 0.)
    moving = 0 if side == 'left' else 1
    direction = 1. if side == 'left' else -1.
    neutral[:, 1] -= direction*support_shift_m
    neutral[:, 2] -= root_lower_m
    roll = direction*math.radians(support_roll_deg)
    neutral[:, 3], neutral[:, 6] = math.sin(roll/2.), math.cos(roll/2.)
    count = math.ceil(seconds*fps)
    time = torch.arange(count, dtype=model.dtype)/fps
    def smoothstep(x):
        x = x.clamp(0., 1.)
        return x**3*(10.-15.*x+6.*x*x)
    envelope = smoothstep(time-.5)*smoothstep(time[-1]-time-.5)
    yaw = math.radians(yaw_amplitude_deg)*envelope*torch.sin(2.*math.pi*time/period_s)
    rotation = torch.eye(3, dtype=model.dtype).repeat(count, 1, 1)
    rotation[:, 0, 0] = rotation[:, 1, 1] = yaw.cos()
    rotation[:, 1, 0], rotation[:, 0, 1] = yaw.sin(), -yaw.sin()
    pitch = math.radians(toe_pitch_deg)
    toe_rotation = torch.tensor([[math.cos(pitch), 0., math.sin(pitch)],
                                 [0., 1., 0.], [-math.sin(pitch), 0., math.cos(pitch)]],
                                dtype=model.dtype)
    targets, pivots = {}, []
    for i, foot in enumerate(FEET):
        frame = initial['frames'][foot]
        local_pivot = (points[i, points[i, :, 0] == points[i, :, 0].max()].mean(0)
                       if i == moving and toe_pitch_deg > 0. else points[i].mean(0))
        pivot = frame['position'][0]+frame['rotation'][0]@local_pivot
        pivots.append(pivot)
        target_rotation = (rotation@toe_rotation@frame['rotation'] if i == moving else
                           frame['rotation'].repeat(count, 1, 1))
        target_position = pivot-(target_rotation@local_pivot[:, None]).squeeze(-1)
        targets[foot] = (target_position, target_rotation)
    leg_ids = [model.joint_names.index(f'{prefix}_{name}')
               for prefix in ('Left', 'Right')
               for name in ('Hip_Pitch', 'Hip_Roll', 'Hip_Yaw', 'Knee_Pitch', 'Ankle_Pitch', 'Ankle_Roll')]
    q_ids, v_ids = [7+i for i in leg_ids], [6+i for i in leg_ids]
    poses = []
    for start in range(0, count, 64):
        q = neutral.repeat(min(64, count-start), 1)
        zero = torch.zeros((len(q), model.nv), dtype=model.dtype)
        for _ in range(40):
            state = model.state(q, zero)
            errors, jacobians = [], []
            for foot in FEET:
                frame = state['frames'][foot]
                target_position, target_rotation = [v[start:start+len(q)] for v in targets[foot]]
                relative = target_rotation@frame['rotation'].transpose(-1, -2)
                rotation_error = .5*torch.stack((relative[:, 2, 1]-relative[:, 1, 2],
                    relative[:, 0, 2]-relative[:, 2, 0], relative[:, 1, 0]-relative[:, 0, 1]), -1)
                errors.extend((target_position-frame['position'], rotation_error))
                jacobians.append(frame['J'][:, :, v_ids])
            error = torch.cat(errors, -1)
            if float(error.abs().max()) < 1e-9:
                break
            jacobian = torch.cat(jacobians, 1)
            transpose = jacobian.transpose(-1, -2)
            step = torch.linalg.solve(transpose@jacobian+1e-10*torch.eye(12, dtype=model.dtype),
                                      transpose@error[..., None]).squeeze(-1)
            step /= (step.abs().amax(-1, keepdim=True)/.1).clamp_min(1.)
            q[:, q_ids] += step
        else:
            raise ValueError('Rotation IK did not converge; reduce yaw, toe pitch, or support shift')
        poses.append(q)
    q = torch.cat(poses)
    joints = q[:, [7+model.joint_names.index(n) for n in joint_names]]
    limits = torch.as_tensor(joint_limits, dtype=model.dtype).reshape(len(joint_names), 2)
    if not bool(torch.isfinite(limits).all() & (limits[:, 0] <= limits[:, 1]).all()):
        raise ValueError('Expected finite ordered joint limits')
    if not bool(((joints >= limits[:, 0]) & (joints <= limits[:, 1])).all()):
        raise ValueError('Rotation exceeds joint limits; reduce yaw, toe pitch, or support shift')
    values, report = prepare_wbc_reference(joints.numpy(), q[:, :3].numpy(), q[:, [6, 3, 4, 5]].numpy(),
        joint_names, body_names, limits.numpy(), fps)
    if report['joint_values_clamped'] or report['max_root_lift_m'] > 1e-7:
        raise ValueError('Rotation reference required a joint or ground correction')
    values.update(fps=np.array(fps), joint_names=np.asarray(joint_names), body_names=np.asarray(body_names),
                  motion_type='foot_toe_rotation' if toe_pitch_deg else 'foot_rotation',
                  rotation_side=side, tilt_deg=0., toe_pitch_deg=toe_pitch_deg,
                  pivot_kind='toe_edge' if toe_pitch_deg else 'sole_centre',
                  yaw_amplitude_deg=yaw_amplitude_deg, period_s=period_s, yaw_angle_rad=yaw.numpy(),
                  pivot_position_w=pivots[moving].numpy(),
                  support_root_shift_y_m=-direction*support_shift_m, support_root_lower_m=root_lower_m,
                  support_root_roll_deg=direction*support_roll_deg)
    return values
