"""Current generalized reference with sole-point lookahead for FORCE2."""
import torch


def _contact_reference(env, command, ids):
    """COM-relative corner positions and world linear velocities, without terrain."""
    from whole_body_tracking.utils.wbc_acc_contact_policy import FEET, sole_vertices
    from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw

    motion = command.motion
    model = env.action_manager.get_term('wbc_force').dynamics
    body_names = command.robot.body_names
    order = [model.body_names.index(name) for name in body_names]
    feet = [body_names.index(name) for name in FEET]
    position = motion._body_pos_w[ids]
    rotation = rotation_xyzw(motion._body_quat_w[ids][..., [1, 2, 3, 0]])
    mass = model.mass[order].to(position)
    local_com = model.com_local[order].to(position)
    body_com = position+(rotation@local_com[..., None]).squeeze(-1)
    com = ((mass/mass.sum())[None, None, :, None]*body_com).sum(-2)
    points = sole_vertices(position.device, position.dtype)
    corner_offset = points@rotation[:, :, feet].transpose(-1, -2)
    corner_position = position[:, :, feet, None]+corner_offset
    linear = motion._body_lin_vel_w[ids][:, :, feet, None]
    angular = motion._body_ang_vel_w[ids][:, :, feet, None]
    corner_velocity = linear+torch.cross(angular.expand_as(corner_offset), corner_offset, dim=-1)
    return torch.cat((corner_position-com[:, :, None, None], corner_velocity), -1).flatten(2)


def reference_window(env, offsets=(0, 1, 3, 5, 10)):
    """Keep current joints/root; expose only sole-point geometry at future offsets.

    Offset zero retains FORCE's joint positions/velocities and anchor-body pose
    and velocities (59 values for T1). Each positive offset contains eight sole
    corners, ordered left then right, C0 through C3. Each corner supplies XYZ
    position relative to the reference whole-body COM and XYZ linear velocity
    in world axes (48 values per future frame). There are no future joint,
    absolute-position, floor-clearance, or separate sole-normal fields.
    Clip endpoints clamp without changing the requested frame's layout.
    """
    if any(offset < 0 for offset in offsets):
        raise ValueError('FORCE2 reference offsets must be nonnegative')
    command = env.command_manager.get_term('motion')
    motion = command.motion
    ids = (command.time_steps[:, None]+torch.tensor(offsets, device=env.device)[None]).clamp(
        0, motion.time_step_total-1)
    if hasattr(motion, 'window_indices'):
        ids = motion.window_indices(command.time_steps, offsets)
    future_columns = [column for column, offset in enumerate(offsets) if offset > 0]
    future = _contact_reference(env, command, ids[:, future_columns]) if future_columns else None
    future_index = 0
    values = []
    for column, offset in enumerate(offsets):
        frames = ids[:, column]
        if offset == 0:
            values.extend((motion.joint_pos[frames], motion.joint_vel[frames]))
            for name in ('_body_pos_w', '_body_quat_w', '_body_lin_vel_w', '_body_ang_vel_w'):
                data = getattr(motion, name)[:, command.robot_anchor_body_index]
                values.append(data[frames].flatten(1))
        else:
            values.append(future[:, future_index])
            future_index += 1
    return torch.cat(values, dim=-1)


# The ONNX consumer must assemble the same mixed-width reference window.
reference_window.export_metadata = {
    'actor_reference_body_selection': 'offset_0:anchor_only;future_offsets:sole_corners',
    'actor_reference_layout': (
        'frame_major;offset_0:joint_pos,joint_vel,body_pos_w,body_quat_w_wxyz,body_lin_vel_w,body_ang_vel_w;'
        'future_offsets:corner_major:position_relative_to_reference_com_xyz,linear_velocity_world_xyz'),
    'actor_reference_contact_point_order': ','.join(
        f'{side}_corner_{corner}' for side in ('left', 'right') for corner in range(4)),
    'actor_reference_future_frame_size': 48,
    'actor_reference_future_floor_clearance': False,
}
