"""Current generalized reference with body-only future frames for FORCE2."""
import torch


def reference_window(env, offsets=(0, 1, 3, 5, 10)):
    """Keep current joints/root; expose only all-link kinematics at future offsets.

    Offset zero retains FORCE's joint positions/velocities and anchor-body pose
    and velocities (59 values for T1). Positive offsets contain every body's
    world position, WXYZ quaternion, linear velocity and angular velocity
    (24 * 13 = 312 values for T1), with no explicit joint fields. Fields are
    flattened in that order within each selected frame. Clip endpoints clamp
    lookahead without changing the layout of a requested future frame.
    """
    command = env.command_manager.get_term('motion')
    motion = command.motion
    ids = (command.time_steps[:, None]+torch.tensor(offsets, device=env.device)[None]).clamp(
        0, motion.time_step_total-1)
    if hasattr(motion, 'window_indices'):
        ids = motion.window_indices(command.time_steps, offsets)
    values = []
    for column, offset in enumerate(offsets):
        if offset < 0:
            raise ValueError('FORCE2 reference offsets must be nonnegative')
        frames = ids[:, column]
        if offset == 0:
            values.extend((motion.joint_pos[frames], motion.joint_vel[frames]))
        for name in ('_body_pos_w', '_body_quat_w', '_body_lin_vel_w', '_body_ang_vel_w'):
            data = getattr(motion, name)
            if offset == 0:
                data = data[:, command.robot_anchor_body_index]
            values.append(data[frames].flatten(1))
    return torch.cat(values, dim=-1)


# The ONNX consumer must assemble the same mixed-width reference window.
reference_window.export_metadata = {
    'actor_reference_body_selection': 'offset_0:anchor_only;future_offsets:all_bodies',
    'actor_reference_layout': (
        'frame_major;offset_0:joint_pos,joint_vel,body_pos_w,body_quat_w_wxyz,body_lin_vel_w,body_ang_vel_w;'
        'future_offsets:body_pos_w,body_quat_w_wxyz,body_lin_vel_w,body_ang_vel_w'),
}
