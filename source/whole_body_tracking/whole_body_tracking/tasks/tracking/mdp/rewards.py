from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    threshold = 20.0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact_mask = (torch.linalg.norm(net_forces_w, dim=-1) > threshold)  # (N, |body_ids|)
    contact_mask = torch.cat([torch.zeros(contact_mask.shape[0], 1, device=contact_mask.device, dtype=contact_mask.dtype), 
                              contact_mask], dim=-1) # Set zero contact for the base body.
    pred_contact_mask = torch.sigmoid(env.hybrid_rew_info["w"])
    lse = torch.sum(torch.square(contact_mask.float() - pred_contact_mask), dim=-1)
    return lse

def left_foot_right_foot_collision(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize left-foot collisions with the right foot using filtered contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_history = contact_sensor.data.force_matrix_w_history
    if force_matrix_history is None:
        return torch.zeros(
            contact_sensor.data.net_forces_w.shape[0],
            device=contact_sensor.data.net_forces_w.device,
        )

    max_force = torch.max(torch.norm(force_matrix_history, dim=-1), dim=1)[0]
    is_collision = (max_force > threshold).reshape(max_force.shape[0], -1)
    return is_collision.sum(dim=1).float()

def centroid_velocity(env: ManagerBasedRLEnv):
    command_name = "motion"
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, ["Trunk"])
    linvel = command.body_lin_vel_w[:, body_indexes][:, 0, :]
    des_linvel = env.hybrid_rew_info["des_com_vel"][:, :3]
    lse = torch.sum(torch.square(linvel - des_linvel), dim=-1)

    linvel_rew = torch.exp(-lse / 0.25)
    return linvel_rew

def centroid_angular_velocity(env: ManagerBasedRLEnv):
    command_name = "motion"
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, ["Trunk"])
    angvel = command.body_ang_vel_w[:, body_indexes][:, 0, :]
    des_angvel = env.hybrid_rew_info["des_com_angvel"][:, :3]
    lse = torch.sum(torch.square(angvel - des_angvel), dim=-1)

    linvel_rew = torch.exp(-lse / 3.14**2)
    return linvel_rew

def hybrid_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    joint_count = env.hybrid_controller.joint_count
    end_effector_count = env.hybrid_controller.end_effector_count

    def slice_action(vec):
        des_pos = vec[:, 0:joint_count]
        des_com_vel = vec[:, joint_count : joint_count + 3]
        w = vec[:, joint_count + 3 : joint_count + end_effector_count + 4]
        torque = vec[
            :,
            joint_count + end_effector_count + 4 : joint_count * 2 + end_effector_count + 4,
        ]
        des_com_angvel = vec[
            :,
            joint_count * 2 + end_effector_count + 4 : joint_count * 2 + end_effector_count + 7,
        ]
        non_w = torch.cat([
            des_com_vel, des_com_angvel
        ], axis = -1)
        return des_pos, non_w, torque, w
    c_pos, c_rem, c_torque, c_w = slice_action(env.action_manager.action)
    p_pos, p_rem, p_torque, p_w = slice_action(env.action_manager.prev_action)
    pos_l2_err = torch.sum(torch.square(c_pos - p_pos), dim=-1)
    remaining_l2_err = torch.sum(torch.square(c_rem - p_rem), dim=-1)
    w_l2_err = torch.sum(torch.square(c_w - p_w), dim=-1)
    torque_l2_err = torch.sum(torch.square(c_torque - p_torque), dim=-1)
    return pos_l2_err + remaining_l2_err * 0.10 + w_l2_err * 1e-2 + torque_l2_err * 1e-3

def hybrid_force_correctness(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    pred_force = env.hybrid_rew_info["grf"]
    num_eef = pred_force.shape[1] // 6
    pred_force = pred_force.reshape(pred_force.shape[0], num_eef, 6)
    pred_lin_force = pred_force[:, :, :3]
    force_err = torch.sum(torch.square(net_forces_w - pred_lin_force), dim=-1)
    sigma = 150.0
    exp_err = torch.exp(-torch.sum(force_err, dim = -1) / (sigma ** 2))
    return exp_err

def hybrid_tau_ref(env: ManagerBasedRLEnv) -> torch.Tensor:
    applied_torque = env.scene["robot"].data.applied_torque
    tau_ref = env.hybrid_rew_info["ff_tau"]
    # Reward for minimizing error between applied torque and ff torque
    frc_err = torch.sum(torch.square(applied_torque - tau_ref), dim=-1)
    sigma = 75
    exp_err = torch.exp(-frc_err / (sigma ** 2))
    return exp_err


def hybrid_tau_limit(env: ManagerBasedRLEnv) -> torch.Tensor:
    ff_torque = env.hybrid_rew_info["ff_tau"]
    # Reward for staying within torque limits
    soft_limit = 0.90

    torque_limits = env.hybrid_controller.torque_limits.to(device=ff_torque.device, dtype=ff_torque.dtype)
    over_limit = torch.relu(torch.abs(ff_torque) - soft_limit * torque_limits)
    frc_err = torch.sum(over_limit, dim=-1)
    return frc_err

def com_acc_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    com_acc = env.hybrid_rew_info["com_acc"]
    acc_mag = torch.linalg.norm(com_acc, dim=-1)
    acc_limit = 4.0
    over_limit = torch.relu(acc_mag - acc_limit)
    return over_limit

def com_angacc_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    com_angacc = env.hybrid_rew_info["com_angacc"]
    angacc_mag = torch.linalg.norm(com_angacc, dim=-1)
    angacc_limit = 40.0
    over_limit = torch.relu(angacc_mag - angacc_limit)
    return over_limit

def com_linacc_corectness(env: ManagerBasedRLEnv) -> torch.Tensor:
    lin_acc = env.hybrid_rew_info["lin_acc"]
    des_com_acc = env.hybrid_rew_info["com_acc"]
    lse = torch.sum(torch.square(lin_acc - des_com_acc), dim=-1)
    sigma = 6.0
    exp_err = torch.exp(-lse / (sigma ** 2))
    return exp_err

def com_angacc_corectness(env: ManagerBasedRLEnv) -> torch.Tensor:
    ang_acc = env.hybrid_rew_info["ang_acc"]
    des_com_angacc = env.hybrid_rew_info["com_angacc"]
    lse = torch.sum(torch.square(ang_acc - des_com_angacc), dim=-1)
    sigma = 12.0
    exp_err = torch.exp(-lse / (sigma ** 2))
    return exp_err
