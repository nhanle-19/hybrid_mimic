"""Atlas rewards evaluated at policy rate using physical QP/contact quantities."""
import torch


def qp_failed(env):
    """Any invalid/infeasible solve since reset, including earlier decimation steps."""
    return env.action_manager.get_term('atlas').qp_failed


def alive_reward(env):
    """Survival rate; RewardManager integrates over dt, with timeouts counted as alive."""
    return (~env.termination_manager.terminated).float()


def mapped_action_rate(env):
    """Mean squared changes in unit-range activations and normalized QP weights."""
    def mapped(actions):
        values = actions.reshape(-1, 2, 7)
        # Exactly (weight - min) / (max - min) for decode_actions' mapping.
        return torch.cat((values[..., :1].clamp(0., 1.), values[..., 1:].sigmoid()), -1)

    manager = env.action_manager
    return (mapped(manager.action) - mapped(manager.prev_action)).square().mean(dim=(1, 2))


def excessive_slip(env, speed_threshold=.05, force_threshold=1.):
    """Normal-force-weighted tangential speed penalty at measured support points.

    v(point) = v(link) + omega x (point - link); a stationary toe pivot is free.
    Each foot contributes its mean excess squared speed, independent of the
    number of points PhysX reports. Ground is static in this task.
    """
    robot = env.scene['robot']
    value = torch.zeros(env.num_envs, device=env.device)
    for side in ('left', 'right'):
        sensor = env.scene.sensors[f'{side}_foot_ground_contact']
        forces, points, normals, _, counts, starts = sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
        counts, starts = counts.reshape(-1).long(), starts.reshape(-1).long()
        width = int(counts.max())
        if width == 0:
            continue
        offsets = torch.arange(width, device=env.device)[None]
        ids = (starts[:, None] + offsets).clamp(0, len(points)-1)
        weights = torch.where(offsets < counts[:, None], forces.reshape(-1)[ids].clamp_min(0), 0.)
        body = robot.body_names.index(f'{side}_foot_link')
        arm = points[ids] - robot.data.body_link_pos_w[:, body, None]
        velocity = robot.data.body_link_lin_vel_w[:, body, None] + torch.cross(
            robot.data.body_ang_vel_w[:, body, None].expand_as(arm), arm, dim=-1)
        normal = normals[ids]
        tangent = velocity - (velocity*normal).sum(-1, keepdim=True)*normal
        penalty = (tangent.norm(dim=-1)-speed_threshold).clamp_min(0).square()
        total = weights.sum(-1)
        value += torch.where(total > force_threshold, (weights*penalty).sum(-1)/total.clamp_min(1e-8), 0.)
    return value
