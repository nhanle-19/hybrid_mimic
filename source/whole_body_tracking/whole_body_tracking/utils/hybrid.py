import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply


class HybridController:
    """Runtime Hybrid controller parameters resolved from an Isaac articulation."""

    def __init__(self, articulation, cfg):
        requested_body_names = list(cfg.end_effector_body_names)
        if not requested_body_names:
            raise ValueError("Hybrid control requires at least one configured end effector.")
        end_effector_ids, resolved_body_names = articulation.find_bodies(
            requested_body_names,
            preserve_order=True,
        )
        if resolved_body_names != requested_body_names:
            raise ValueError(
                "Failed to resolve Hybrid end effectors in the configured order: "
                f"requested={requested_body_names}, resolved={resolved_body_names}."
            )

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
            raise ValueError("Hybrid control requires finite, positive effort limits for every articulation joint.")

        joint_names = list(articulation.joint_names)
        configured_cost_names = set(cfg.torque_limits_cost)
        missing_cost_names = set(joint_names) - configured_cost_names
        extra_cost_names = configured_cost_names - set(joint_names)
        if missing_cost_names or extra_cost_names:
            raise ValueError(
                "Hybrid torque cost limits must match the Isaac articulation joints exactly: "
                f"missing={sorted(missing_cost_names)}, extra={sorted(extra_cost_names)}."
            )
        torque_limits_cost = torch.tensor(
            [cfg.torque_limits_cost[name] for name in joint_names],
            device=articulation.device,
            dtype=self.torque_limits.dtype,
        )
        if not torch.all(torch.isfinite(torque_limits_cost) & (torque_limits_cost > 0)):
            raise ValueError("Hybrid torque cost limits must be finite and positive.")
        self.torque_limits_cost = torque_limits_cost.unsqueeze(0).expand_as(self.torque_limits)

        self.nominal_mass = articulation.data.default_mass.sum(dim=-1)
        if not torch.all(torch.isfinite(self.nominal_mass) & (self.nominal_mass > 0)):
            raise ValueError("Isaac articulation masses must produce a finite, positive total mass.")
        inertia = torch.as_tensor(
            cfg.nominal_angular_inertia,
            device=articulation.device,
            dtype=self.nominal_mass.dtype,
        )
        if inertia.shape != (3, 3):
            raise ValueError(f"Hybrid nominal angular inertia must have shape (3, 3), received {inertia.shape}.")
        self.nominal_angular_inertia = inertia.unsqueeze(0).expand(self.nominal_mass.shape[0], -1, -1)

        self.desired_linear_velocity_scale = cfg.desired_linear_velocity_scale
        self.desired_angular_velocity_scale = cfg.desired_angular_velocity_scale
        self.torque_action_scale = cfg.torque_action_scale
        self.linear_velocity_gain = cfg.linear_velocity_gain
        self.angular_velocity_gain = cfg.angular_velocity_gain
        self.force_objective_weight = cfg.force_objective_weight
        self.torque_objective_weight = cfg.torque_objective_weight
        self.angular_force_scale = cfg.angular_force_scale
        self.force_logit_clip = cfg.force_logit_clip
        self.gravity_magnitude = cfg.gravity_magnitude

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
        return step(
            com_pos,
            com_vel,
            jacs,
            body_pos,
            base_quat,
            base_angvel,
            action,
            nle,
            lcc_rand,
            joint_count=self.joint_count,
            end_effector_ids=self.end_effector_ids,
            torque_limits=self.torque_limits,
            torque_limits_cost=self.torque_limits_cost,
            nominal_mass=self.nominal_mass,
            nominal_angular_inertia=self.nominal_angular_inertia,
            desired_linear_velocity_scale=self.desired_linear_velocity_scale,
            desired_angular_velocity_scale=self.desired_angular_velocity_scale,
            torque_action_scale=self.torque_action_scale,
            linear_velocity_gain=self.linear_velocity_gain,
            angular_velocity_gain=self.angular_velocity_gain,
            force_objective_weight=self.force_objective_weight,
            torque_objective_weight=self.torque_objective_weight,
            angular_force_scale=self.angular_force_scale,
            force_logit_clip=self.force_logit_clip,
            gravity_magnitude=self.gravity_magnitude,
        )


@torch.compile
def ctrl2logits(act, joint_count, end_effector_count):
    des_pos = act[:, 0:joint_count]
    des_com_vel = act[:, joint_count : joint_count + 3]
    w = act[:, joint_count + 3 : joint_count + end_effector_count + 4]
    torque = act[
        :,
        joint_count + end_effector_count + 4 : joint_count * 2 + end_effector_count + 4,
    ]
    des_com_angvel = act[
        :,
        joint_count * 2 + end_effector_count + 4 : joint_count * 2 + end_effector_count + 7,
    ]
    logits = {
        "des_pos": des_pos,
        "des_com_vel": des_com_vel,
        "des_com_angvel": des_com_angvel,
        "w": w,
        "torque": torque,
    }
    return logits


@torch.compile
def ctrl2components(
    act,
    joint_count,
    end_effector_count,
    torque_limits,
    torque_limits_cost,
    desired_linear_velocity_scale,
    desired_angular_velocity_scale,
    torque_action_scale,
    linear_velocity_gain,
    angular_velocity_gain,
):
    logits = ctrl2logits(act, joint_count, end_effector_count)
    des_pos = logits["des_pos"]
    des_angvel = logits["des_com_angvel"] * desired_angular_velocity_scale
    des_vel = logits["des_com_vel"] * desired_linear_velocity_scale

    w = logits["w"]

    torque_logit = logits["torque"] * torque_action_scale
    torque_limits = torque_limits.to(device=torque_logit.device, dtype=torque_logit.dtype)
    tau = torque_limits * torque_logit

    torque_limits_cost = torque_limits_cost.to(device=torque_logit.device, dtype=torque_logit.dtype)
    torque_weight = torch.square(torch.reciprocal(torque_limits_cost))

    return {
        "des_pos": des_pos,
        "des_com_vel": des_vel,
        "des_com_angvel": des_angvel,
        "w": w,
        "torque": tau,
        "d_gain_lin": linear_velocity_gain,
        "d_gain_angvel": angular_velocity_gain,
        "torque_weight": torque_weight
    }

@torch.compile
def make_centroidal_ag(eefpos, com_pos, base_quat, mass, i_b, grav_vec, gravity_magnitude):
    """
    Vectorized version of make_centroidal_ag without Python loops.

    Args:
      eefpos: (N, E, 3)
      com_pos:(N, 3)

    Returns:
      a: (N, 6, 6*E)
      g: (6,)
    """
    r = eefpos - com_pos[:, None, :]  # (N, E, 3)
    N, E, _ = r.shape
    device, dtype = r.device, r.dtype

    # Skew-symmetric matrices S(r) for all effectors: (N, E, 3, 3)
    rx, ry, rz = r.unbind(-1)
    S = torch.zeros(N, E, 3, 3, device=device, dtype=dtype)
    S[..., 0, 1] = -rz
    S[..., 0, 2] =  ry
    S[..., 1, 0] =  rz
    S[..., 1, 2] = -rx
    S[..., 2, 0] = -ry
    S[..., 2, 1] =  rx

    # Compute the aggregate inertia in the world frame.
    rot_mat = matrix_from_quat(base_quat)  # (N, 3, 3)
    i_w = rot_mat @ i_b @ rot_mat.transpose(-1, -2)
    invI = torch.linalg.inv(i_w)

    # Bottom block per effector: [invI @ S, invI] -> (N, E, 3, 6)
    bot_left = invI.view(N, 1, 3, 3).expand(N, E, 3, 3) @ S                         # (N, E, 3, 3)
    bot_right = invI.view(N, 1, 3, 3).expand(N, E, 3, 3)         # (N, E, 3, 3)
    f_bot = torch.cat([bot_left, bot_right], dim=-1)  # (N, E, 3, 6)

    # Top block per effector: [I/M, 0] -> (N, E, 3, 6)
    I3 = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).expand(N, 3, 3)
    f_top_base = torch.cat([I3 / mass[:, None, None], torch.zeros_like(I3)], dim=-1)  # (3, 6)
    f_top = f_top_base.view(N, 1, 3, 6).expand(N, E, 3, 6)

    # Full per-effector 6x6 block: (N, E, 6, 6)
    f_block = torch.cat([f_top, f_bot], dim=-2)  # (N, E, 6, 6)

    # Concatenate horizontally across effectors: (N, 6, 6*E)
    a = f_block.permute(0, 2, 1, 3).reshape(N, 6, E * 6)

    g_base = eefpos.new_tensor([0.0, 0.0, -gravity_magnitude, 0.0, 0.0, 0.0])
    gtau = torch.zeros_like(grav_vec, device=device, dtype=dtype)  # (N, 3)
    g = torch.cat([grav_vec, gtau], axis = -1) + g_base[None, :]
    g = g * gravity_magnitude / torch.norm(g, dim=-1, keepdim=True)
    return a, g

@torch.compile
def f_mag_q(w: torch.Tensor, angular_force_scale: float, force_logit_clip: float) -> torch.Tensor:
    # Accept (N, E) or (E,)
    if w.ndim == 1:
        w = w.unsqueeze(0)  # (1, E)

    # Same scaling as your original
    logits = -torch.clip(w, min=-force_logit_clip, max=force_logit_clip)  # (N, E)
    scale_lin = torch.exp(logits)                  # (N, E)
    scale_ang = scale_lin * angular_force_scale    # (N, E)

    # Build per-effector 6-tuple = [lin, lin, lin, ang, ang, ang]
    # Shape: (N, E, 6) so each effector's 6 entries stay contiguous
    lin3 = scale_lin.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    ang3 = scale_ang.unsqueeze(-1).expand(-1, -1, 3)  # (N,E,3)
    per_eff = torch.cat([lin3, ang3], dim=-1)         # (N,E,6)

    # Flatten effector+axis to (N, E*6), then put on the diagonal
    diag_vec = per_eff.reshape(per_eff.shape[0], -1)  # (N, E*6)
    qp_q = torch.diag_embed(diag_vec)                 # (N, E*6, E*6)
    return qp_q

@torch.compile
def joint_torque_q(jacs: torch.Tensor, tau_ref: torch.Tensor, w: torch.Tensor | None = None):
    """
    jacs:    (N, 6*num_end_effectors, 6+num_joints) or (6*num_end_effectors, 6+num_joints)
    tau_ref: (N, num_joints)                         or (num_joints,)
    w:       Optional weights for ALL dofs (base+joint):
             (N, 6+num_joints) or (6+num_joints,). Only the joint portion
             is used here since J_j excludes the 6 base dofs.

    Returns:
      big_q:   (N, 6*num_end_effectors, 6*num_end_effectors) = J_j @ W @ J_j^T
      small_q: (N, 6*num_end_effectors)                    = J_j @ (W @ tau_ref)
    where J_j = -jacs[..., :, 6:]  (exclude the 6 base dofs)
    and W is diagonal formed from w[..., 6:].

    Notes:
      Implemented without explicitly constructing W/diag matrices:
        big_q = (J_j * wj) @ (J_j)^T
        small_q = J_j @ (tau_ref * wj)
    """
    device, dtype = jacs.device, jacs.dtype

    # Normalize jacs to (N, F, 6+CTRL)
    if jacs.dim() == 2:
        jacs = jacs.unsqueeze(0)
    jacs = jacs.to(dtype=dtype)
    N, F, _ = jacs.shape

    # J_j: (N, F, CTRL)
    J_j = -jacs[..., :, 6:]
    CTRL = J_j.shape[-1]

    # Normalize tau_ref to (N, CTRL)
    if tau_ref.dim() == 1:
        tau_ref = tau_ref.unsqueeze(0)
    else:
        tau_ref = tau_ref.reshape(-1, tau_ref.shape[-1])
    tau_ref = tau_ref.to(device=device, dtype=dtype)

    if tau_ref.shape[-1] != CTRL:
        raise ValueError(f"CTRL dim mismatch: J_j has {CTRL} but tau_ref has {tau_ref.shape[-1]}")

    # Expand or validate batch
    if tau_ref.shape[0] == 1 and N > 1:
        tau_ref = tau_ref.expand(N, -1)
    elif tau_ref.shape[0] != N:
        raise ValueError(f"Batch mismatch: jacs batch {N} vs tau_ref batch {tau_ref.shape[0]}")

    # --- weights handling (optional) ---
    # w is defined over (6+CTRL); only last CTRL apply to J_j columns.
    if w is None:
        # Unweighted case
        big_q = J_j @ J_j.transpose(-1, -2)
        small_q = torch.bmm(J_j, tau_ref.unsqueeze(-1)).squeeze(-1)
        return big_q, small_q

    # Ensure weights are on the same device/dtype as jacs.
    wj = w
    if wj.dim() == 1:
        wj = wj.unsqueeze(0)
    else:
        wj = wj.reshape(-1, wj.shape[-1])
    wj = wj.to(device=device, dtype=dtype)

    # big_q = J_j @ W @ J_j^T  == (J_j * w_j) @ (J_j)^T
    Jw = J_j * wj.unsqueeze(-2)  # (N, F, CTRL)
    big_q = Jw @ J_j.transpose(-1, -2)

    # small_q = J_j @ (W @ tau_ref)  == J_j @ (tau_ref * wj)
    tau_w = tau_ref * wj  # (N, CTRL)
    small_q = torch.bmm(J_j, tau_w.unsqueeze(-1)).squeeze(-1)  # (N, F)

    return big_q, small_q

@torch.compile
def centroidal_qacc_cons(big_a, g, com_ref):
    lhs = big_a
    rhs = com_ref - g
    return lhs, rhs

@torch.compile
def schur_solve(
    qp_q: torch.Tensor,
    qp_c: torch.Tensor,
    cons_lhs: torch.Tensor,
    cons_rhs: torch.Tensor,
    reg: float = 1e-6,
):
    """
    qp_q:     (..., F, F)
    qp_c:     (..., F)
    cons_lhs: (..., M, F)   (A)
    cons_rhs: (..., M)      (b)

    Returns:
      x: (..., F)
    """
    device, dtype = qp_q.device, qp_q.dtype
    qp_c = qp_c.to(device=device, dtype=dtype)
    cons_lhs = cons_lhs.to(device=device, dtype=dtype)
    cons_rhs = cons_rhs.to(device=device, dtype=dtype)

    squeeze_out = False
    if qp_q.dim() == 2:
        qp_q = qp_q.unsqueeze(0)
        qp_c = qp_c.unsqueeze(0)
        cons_lhs = cons_lhs.unsqueeze(0)
        cons_rhs = cons_rhs.unsqueeze(0)
        squeeze_out = True

    batch_shape = qp_q.shape[:-2]
    F = qp_q.shape[-1]
    M = cons_lhs.shape[-2]

    # Symmetrize Q
    Q = 0.5 * (qp_q + qp_q.transpose(-1, -2))

    # Optional Tikhonov regularization on Q
    if reg > 0.0:
        I = torch.eye(F, device=device, dtype=dtype).expand(*batch_shape, F, F)
        Q = Q + reg * I

    A = cons_lhs                      # (..., M, F)
    AT = A.transpose(-1, -2)          # (..., F, M)
    c = qp_c                          # (..., F)
    b = cons_rhs                      # (..., M)

    # Factor Q once: LU (works for indefinite too)
    LU, pivots, infoQ = torch.linalg.lu_factor_ex(Q, pivot=True, check_errors=False)

    def solve_Q(B: torch.Tensor) -> torch.Tensor:
        # solves Q X = B
        return torch.linalg.lu_solve(LU, pivots, B)

    # Compute Q^{-1} A^T and Q^{-1} c
    Qinv_AT = solve_Q(AT)                              # (..., F, M)
    Qinv_c = solve_Q(c.unsqueeze(-1)).squeeze(-1)      # (..., F)

    # Schur matrix S = A Q^{-1} A^T  and rhs = A Q^{-1} c - b
    S = A @ Qinv_AT                                    # (..., M, M)
    rhs_lam = (A @ Qinv_c.unsqueeze(-1)).squeeze(-1) - b   # (..., M)

    # Solve for lambda (small MxM system)
    lam, _ = torch.linalg.solve_ex(S, rhs_lam.unsqueeze(-1), check_errors=False)
    lam = lam.squeeze(-1)                              # (..., M)

    # Recover x = Q^{-1}(c - A^T lambda)
    x = solve_Q((c - (AT @ lam.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)

    if squeeze_out:
        x = x.squeeze(0)
    return x


def hybrid_ref(
    eefpos_,
    com_pos,
    jacs_,
    tau_ref,
    com_ref,
    w,
    torque_weight,
    base_quat,
    mass,
    i_b,
    grav_vec,
    nle,
    torque_limits,
    force_objective_weight,
    torque_objective_weight,
    angular_force_scale,
    force_logit_clip,
    gravity_magnitude,
):
    # Concat the unaccounted force component
    ctrl_num = tau_ref.shape[-1]
    unaccounted_jac = torch.zeros(
        (jacs_.shape[0], 6, ctrl_num + 6), device = jacs_.device
    )
    unaccounted_jac[:, :6, :6] = torch.eye(6, device = jacs_.device)
    jacs = torch.cat([unaccounted_jac, jacs_], dim = 1)
    eefpos = torch.cat([
        com_pos[:, None, :], eefpos_
    ], dim = 1)

    a, g = make_centroidal_ag(
        eefpos,
        com_pos,
        base_quat,
        mass,
        i_b,
        grav_vec,
        gravity_magnitude,
    )

    qp_q_ = f_mag_q(w, angular_force_scale, force_logit_clip)
    qp_q_ = qp_q_ * force_objective_weight
    
    jt_q_big, jt_q_small = joint_torque_q(jacs, tau_ref, torque_weight)
    jt_q_big = jt_q_big * torque_objective_weight

    qp_q = qp_q_ + jt_q_big
    qp_c = jt_q_small * torque_objective_weight

    cons_lhs, cons_rhs = centroidal_qacc_cons(a, g, com_ref)

    f = schur_solve(qp_q, qp_c, cons_lhs, cons_rhs)

    candidate_tau = -jacs[..., :, 6:].transpose(-1, -2) @ f[..., None]
    candidate_tau = candidate_tau.squeeze(-1)
    candidate_tau = candidate_tau + nle

    torque_limits = torque_limits.to(device=candidate_tau.device, dtype=candidate_tau.dtype)
    tau = torch.clamp(candidate_tau, min=-torque_limits, max=torque_limits)

    f = f[:, 6:] # remove unaccounted force
    info = {
        "f": f,
        "candidate_tau": candidate_tau,
        "w": w,
    }
    return tau, info

def highlvlPD(base_quat, base_angvel, 
              lin_gain, angvel_gain,
              des_vel, des_angvel,
              com_vel):
    q_wb = base_quat
    global_des_vel = quat_apply(q_wb, des_vel)
    global_des_angvel = quat_apply(q_wb, des_angvel)

    com_acc = lin_gain * (global_des_vel - com_vel)

    # com_acc should be clipped to a max of 5

    #acc_mag = torch.linalg.norm(com_acc, dim=-1, keepdim=True)
    #max_acc = 5.0
    #new_acc_mag = torch.clamp(acc_mag, max=max_acc)
    #com_acc = com_acc * (new_acc_mag / (acc_mag + 1e-6))
    #com_acc = torch.clamp(com_acc, min=-3.0, max=3.0)

    com_angvel = base_angvel
    ang_acc = angvel_gain * (global_des_angvel - com_angvel)

    return com_acc, ang_acc, global_des_vel, global_des_angvel

def step(com_pos, com_vel,
         jacs,
         body_pos,
         base_quat, base_angvel,
         action, nle, lcc_rand,
         *,
         joint_count,
         end_effector_ids,
         torque_limits,
         torque_limits_cost,
         nominal_mass,
         nominal_angular_inertia,
         desired_linear_velocity_scale,
         desired_angular_velocity_scale,
         torque_action_scale,
         linear_velocity_gain,
         angular_velocity_gain,
         force_objective_weight,
         torque_objective_weight,
         angular_force_scale,
         force_logit_clip,
         gravity_magnitude):
    end_effector_count = end_effector_ids.numel()
    comp_dict = ctrl2components(
        action,
        joint_count,
        end_effector_count,
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
        base_quat, base_angvel_,
        comp_dict["d_gain_lin"], comp_dict["d_gain_angvel"],
        comp_dict["des_com_vel"], comp_dict["des_com_angvel"],
        com_vel_
    )

    selected_jacs = jacs.index_select(1, end_effector_ids)
    jacs_ = selected_jacs.reshape(selected_jacs.size(0), -1, selected_jacs.size(-1))
    eefpos_ = body_pos.index_select(1, end_effector_ids)

    # Add offsets to pos
    eefpos_offset = lcc_rand["pos"][:, 1:, :]
    eefpos_0 = eefpos_ + eefpos_offset
    com_pos_ = com_pos + lcc_rand["pos"][:, 0, :]

    # Modify jacs
    jacs_0 = jacs_ * lcc_rand["jac_fac"]

    mass = nominal_mass * lcc_rand["mass_fac"]
    i_b = nominal_angular_inertia * lcc_rand["i_fac"]

    tau, info = hybrid_ref(
        eefpos_0, com_pos_, jacs_0,
        comp_dict["torque"],
        torch.cat([com_acc, ang_acc], dim=-1),
        comp_dict["w"],
        comp_dict["torque_weight"],
        base_quat,
        mass,
        i_b,
        lcc_rand["grav_vec"],
        nle,
        torque_limits,
        force_objective_weight,
        torque_objective_weight,
        angular_force_scale,
        force_logit_clip,
        gravity_magnitude,
    )
    info["com_vel"] = global_vel
    info["com_angvel"] = global_angvel
    info["com_acc"] = com_acc
    info["com_angacc"] = ang_acc
    return comp_dict["des_pos"], tau, info
