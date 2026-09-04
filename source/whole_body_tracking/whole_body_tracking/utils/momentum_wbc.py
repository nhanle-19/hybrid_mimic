"""Momentum-based whole-body controller used by the T1 momentum task.

The controller follows the structure of Koolen et al.'s momentum-based control
framework: solve a QP for desired generalized accelerations and contact wrenches,
then recover joint torques with inverse dynamics.  It keeps the HybridMimic action
layout so the same RL policy structure can be used for comparison.
"""

from __future__ import annotations

import torch

from whole_body_tracking.utils.hybrid import ctrl2components, f_mag_q, highlvlPD, schur_solve


class MomentumBasedWholeBodyController:
    """Runtime whole-body controller parameters resolved from an Isaac articulation."""

    def __init__(self, articulation, cfg):
        requested_body_names = list(cfg.end_effector_body_names)
        if not requested_body_names:
            raise ValueError("Momentum WBC requires at least one configured end effector.")
        end_effector_ids, resolved_body_names = articulation.find_bodies(
            requested_body_names,
            preserve_order=True,
        )
        if resolved_body_names != requested_body_names:
            raise ValueError(
                "Failed to resolve Momentum WBC end effectors in the configured order: "
                f"requested={requested_body_names}, resolved={resolved_body_names}."
            )

        self.articulation = articulation
        self.joint_count = articulation.num_joints
        self.end_effector_ids = torch.tensor(end_effector_ids, device=articulation.device, dtype=torch.long)
        self.end_effector_names = resolved_body_names
        self.end_effector_count = len(end_effector_ids)

        self.torque_limits = articulation.data.joint_effort_limits
        if self.torque_limits.shape[-1] != self.joint_count:
            raise ValueError(
                "Isaac joint effort limits do not match the articulation joint count: "
                f"{self.torque_limits.shape[-1]} != {self.joint_count}."
            )
        if not torch.all(torch.isfinite(self.torque_limits) & (self.torque_limits > 0)):
            raise ValueError("Momentum WBC requires finite, positive effort limits for every articulation joint.")

        joint_names = list(articulation.joint_names)
        configured_cost_names = set(cfg.torque_limits_cost)
        missing_cost_names = set(joint_names) - configured_cost_names
        extra_cost_names = configured_cost_names - set(joint_names)
        if missing_cost_names or extra_cost_names:
            raise ValueError(
                "Momentum WBC torque cost limits must match the Isaac articulation joints exactly: "
                f"missing={sorted(missing_cost_names)}, extra={sorted(extra_cost_names)}."
            )
        torque_limits_cost = torch.tensor(
            [cfg.torque_limits_cost[name] for name in joint_names],
            device=articulation.device,
            dtype=self.torque_limits.dtype,
        )
        if not torch.all(torch.isfinite(torque_limits_cost) & (torque_limits_cost > 0)):
            raise ValueError("Momentum WBC torque cost limits must be finite and positive.")
        self.torque_limits_cost = torque_limits_cost.unsqueeze(0).expand_as(self.torque_limits)

        self.desired_linear_velocity_scale = cfg.desired_linear_velocity_scale
        self.desired_angular_velocity_scale = cfg.desired_angular_velocity_scale
        self.torque_action_scale = cfg.torque_action_scale
        self.linear_velocity_gain = cfg.linear_velocity_gain
        self.angular_velocity_gain = cfg.angular_velocity_gain
        self.joint_position_gain = cfg.joint_position_gain
        self.joint_velocity_gain = cfg.joint_velocity_gain
        self.base_acceleration_weight = cfg.base_acceleration_weight
        self.joint_acceleration_weight = cfg.joint_acceleration_weight
        self.force_objective_weight = cfg.force_objective_weight
        self.torque_objective_weight = cfg.torque_objective_weight
        self.angular_force_scale = cfg.angular_force_scale
        self.force_logit_clip = cfg.force_logit_clip
        self.qp_regularization = cfg.qp_regularization
        self.default_joint_pos = None
        self.action_scale = None

    @property
    def action_dim(self) -> int:
        return self.joint_count * 2 + self.end_effector_count + 7

    def step(
        self,
        com_pos,
        com_vel,
        jacs,
        body_pos,
        base_quat,
        base_angvel,
        action,
        nle,
        lcc_rand,
    ):
        return momentum_wbc_step(
            self.articulation,
            com_vel,
            jacs,
            action,
            base_quat,
            base_angvel,
            lcc_rand,
            joint_count=self.joint_count,
            end_effector_ids=self.end_effector_ids,
            torque_limits=self.torque_limits,
            torque_limits_cost=self.torque_limits_cost,
            desired_linear_velocity_scale=self.desired_linear_velocity_scale,
            desired_angular_velocity_scale=self.desired_angular_velocity_scale,
            torque_action_scale=self.torque_action_scale,
            linear_velocity_gain=self.linear_velocity_gain,
            angular_velocity_gain=self.angular_velocity_gain,
            joint_position_gain=self.joint_position_gain,
            joint_velocity_gain=self.joint_velocity_gain,
            base_acceleration_weight=self.base_acceleration_weight,
            joint_acceleration_weight=self.joint_acceleration_weight,
            force_objective_weight=self.force_objective_weight,
            torque_objective_weight=self.torque_objective_weight,
            angular_force_scale=self.angular_force_scale,
            force_logit_clip=self.force_logit_clip,
            qp_regularization=self.qp_regularization,
            default_joint_pos=self.default_joint_pos,
            action_scale=self.action_scale,
        )


def _generalized_mass_matrix(articulation, expected_dim: int) -> torch.Tensor:
    if not hasattr(articulation.root_physx_view, "get_generalized_mass_matrices"):
        raise AttributeError(
            "Momentum WBC requires root_physx_view.get_generalized_mass_matrices(), "
            "which is not available on this Isaac articulation."
        )
    mass_matrix = articulation.root_physx_view.get_generalized_mass_matrices()
    if mass_matrix.shape[-2:] != (expected_dim, expected_dim):
        raise ValueError(
            "Unexpected generalized mass matrix shape for Momentum WBC: "
            f"expected (..., {expected_dim}, {expected_dim}), received {mass_matrix.shape}."
        )
    return mass_matrix


def _full_bias_forces(articulation, expected_dim: int) -> torch.Tensor:
    gravity = articulation.root_physx_view.get_gravity_compensation_forces()
    if gravity.shape[-1] != expected_dim:
        raise ValueError(
            "Unexpected gravity compensation shape for Momentum WBC: "
            f"expected (..., {expected_dim}), received {gravity.shape}."
        )

    if hasattr(articulation.root_physx_view, "get_coriolis_and_centrifugal_compensation_forces"):
        coriolis = articulation.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        if coriolis.shape[-1] == expected_dim:
            return gravity + coriolis

    return gravity


def _contact_jacobians(
    jacs: torch.Tensor,
    end_effector_ids: torch.Tensor,
    jacobian_bias: torch.Tensor,
) -> torch.Tensor:
    selected_jacs = jacs.index_select(1, end_effector_ids)
    selected_jacs = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))
    selected_jacs = selected_jacs * jacobian_bias.to(device=selected_jacs.device, dtype=selected_jacs.dtype)

    unaccounted_jac = torch.zeros(
        (selected_jacs.shape[0], 6, selected_jacs.shape[-1]),
        device=selected_jacs.device,
        dtype=selected_jacs.dtype,
    )
    unaccounted_jac[:, :6, :6] = torch.eye(6, device=selected_jacs.device, dtype=selected_jacs.dtype)
    return torch.cat([unaccounted_jac, selected_jacs], dim=1)


def momentum_wbc_step(
    articulation,
    com_vel,
    jacs,
    action,
    base_quat,
    base_angvel,
    lcc_rand,
    *,
    joint_count,
    end_effector_ids,
    torque_limits,
    torque_limits_cost,
    desired_linear_velocity_scale,
    desired_angular_velocity_scale,
    torque_action_scale,
    linear_velocity_gain,
    angular_velocity_gain,
    joint_position_gain,
    joint_velocity_gain,
    base_acceleration_weight,
    joint_acceleration_weight,
    force_objective_weight,
    torque_objective_weight,
    angular_force_scale,
    force_logit_clip,
    qp_regularization,
    default_joint_pos,
    action_scale,
):
    generalized_dim = joint_count + 6
    comp_dict = ctrl2components(
        action,
        joint_count,
        end_effector_ids.numel(),
        torque_limits,
        torque_limits_cost,
        desired_linear_velocity_scale,
        desired_angular_velocity_scale,
        torque_action_scale,
        linear_velocity_gain,
        angular_velocity_gain,
    )

    com_vel_ = com_vel + lcc_rand["com_vel"]
    base_angvel_ = base_angvel + lcc_rand["com_angvel"]
    com_acc, ang_acc, global_vel, global_angvel = highlvlPD(
        base_quat,
        base_angvel_,
        comp_dict["d_gain_lin"],
        comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"],
        comp_dict["des_com_angvel"],
        com_vel_,
    )

    mass_matrix = _generalized_mass_matrix(articulation, generalized_dim)
    nle_full = _full_bias_forces(articulation, generalized_dim).to(
        device=mass_matrix.device,
        dtype=mass_matrix.dtype,
    )
    jacs = jacs.to(device=mass_matrix.device, dtype=mass_matrix.dtype)

    contact_jacs = _contact_jacobians(jacs, end_effector_ids, lcc_rand["jac_fac"])

    n_envs = action.shape[0]
    force_dim = contact_jacs.shape[1]
    total_dim = generalized_dim + force_dim
    device = mass_matrix.device
    dtype = mass_matrix.dtype

    desired_qdd = torch.zeros((n_envs, generalized_dim), device=device, dtype=dtype)
    desired_qdd[:, :3] = com_acc.to(device=device, dtype=dtype)
    desired_qdd[:, 3:6] = ang_acc.to(device=device, dtype=dtype)
    normalized_joint_target = comp_dict["des_pos"].to(device=device, dtype=dtype)
    if default_joint_pos is not None and action_scale is not None:
        default_joint_pos = default_joint_pos.to(device=device, dtype=dtype)
        action_scale = action_scale.to(device=device, dtype=dtype)
        joint_target = normalized_joint_target * action_scale.unsqueeze(0) + default_joint_pos
        joint_error = joint_target - articulation.data.joint_pos.to(device=device, dtype=dtype)
    else:
        joint_error = normalized_joint_target
    desired_qdd[:, 6:] = joint_position_gain * joint_error - joint_velocity_gain * articulation.data.joint_vel.to(
        device=device,
        dtype=dtype,
    )

    qdd_weight = torch.ones((generalized_dim,), device=device, dtype=dtype) * joint_acceleration_weight
    qdd_weight[:6] = base_acceleration_weight
    qp_q = torch.zeros((n_envs, total_dim, total_dim), device=device, dtype=dtype)
    qp_c = torch.zeros((n_envs, total_dim), device=device, dtype=dtype)
    qp_q[:, :generalized_dim, :generalized_dim] += torch.diag(qdd_weight).unsqueeze(0)
    qp_c[:, :generalized_dim] += desired_qdd * qdd_weight

    force_q = f_mag_q(
        comp_dict["w"].to(device=device, dtype=dtype),
        angular_force_scale,
        force_logit_clip,
    )
    qp_q[:, generalized_dim:, generalized_dim:] += force_q * force_objective_weight

    m_joint = mass_matrix[:, 6:, :]
    j_joint_t = contact_jacs[:, :, 6:].transpose(-1, -2)
    torque_map = torch.cat([m_joint, -j_joint_t], dim=-1)
    torque_weight = comp_dict["torque_weight"].to(device=device, dtype=dtype)
    weighted_torque_map = torque_map * torque_weight.unsqueeze(-1)
    torque_ref_offset = comp_dict["torque"].to(device=device, dtype=dtype) - nle_full[:, 6:]
    qp_q += torch.matmul(torque_map.transpose(-1, -2), weighted_torque_map) * torque_objective_weight
    qp_c += torch.matmul(weighted_torque_map.transpose(-1, -2), torque_ref_offset.unsqueeze(-1)).squeeze(
        -1
    ) * torque_objective_weight

    base_force_map = contact_jacs[:, :, :6].transpose(-1, -2)
    cons_lhs = torch.zeros((n_envs, 6, total_dim), device=device, dtype=dtype)
    cons_lhs[:, :, :generalized_dim] = mass_matrix[:, :6, :]
    cons_lhs[:, :, generalized_dim:] = -base_force_map
    cons_rhs = -nle_full[:, :6]

    solution = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs, reg=qp_regularization)
    qdd = solution[:, :generalized_dim]
    f = solution[:, generalized_dim:]

    candidate_tau = torch.matmul(torque_map, solution.unsqueeze(-1)).squeeze(-1) + nle_full[:, 6:]
    torque_limits = torque_limits.to(device=candidate_tau.device, dtype=candidate_tau.dtype)
    tau = torch.clamp(candidate_tau, min=-torque_limits, max=torque_limits)

    info = {
        "f": f[:, 6:],
        "candidate_tau": candidate_tau,
        "w": comp_dict["w"],
        "com_vel": global_vel,
        "com_angvel": global_angvel,
        "com_acc": com_acc,
        "com_angacc": ang_acc,
        "qdd": qdd,
    }
    return comp_dict["des_pos"], tau, info
