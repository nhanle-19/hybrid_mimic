"""Reference windows and compact measured state for WBC observations."""
import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg


def reference_window(env, offsets=(0, 1, 5, 10), root_only=False):
    """Frame-major joint and body references; FORCE keeps only the root body.

    For T1, root_only gives 23 joint positions, 23 joint velocities, root
    position (3), WXYZ quaternion (4), and world linear/angular velocities (3+3).
    Each selected frame contains 59 values; FORCE selects five offsets in its config.
    """
    command = env.command_manager.get_term('motion')
    motion = command.motion
    ids = (command.time_steps[:, None]+torch.tensor(offsets, device=env.device)[None]).clamp(0, motion.time_step_total-1)
    if hasattr(motion, 'window_indices'):
        ids = motion.window_indices(command.time_steps, offsets)
    # No robot state, state-relative transforms, previous actions, or sensors.
    values = [motion.joint_pos[ids], motion.joint_vel[ids]]
    for name in ('_body_pos_w', '_body_quat_w', '_body_lin_vel_w', '_body_ang_vel_w'):
        data = getattr(motion, name)
        if root_only:
            # This indexes the full motion body arrays, not the tracked-body subset.
            data = data[:, command.robot_anchor_body_index]
        values.append(data[ids].flatten(2))
    return torch.cat(values, -1).flatten(1)


def actual_root_joint_state(env):
    """One measured 59-value T1 frame, in the compact reference field order.

    The observation manager stacks policy-rate samples and resets each
    environment's history. Positions use world axes with the environment
    origin removed, matching the reference coordinates and the QP state.
    """
    command = env.command_manager.get_term('motion')
    data = command.robot.data
    root = command.robot_anchor_body_index
    return torch.cat((data.joint_pos, data.joint_vel,
                      data.body_link_pos_w[:, root]-env.scene.env_origins,
                      data.body_link_quat_w[:, root],
                      data.body_link_lin_vel_w[:, root],
                      data.body_link_ang_vel_w[:, root]), dim=-1)


@configclass
class WBCACCObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        reference = ObservationTermCfg(func=reference_window)
        enable_corruption = False
        concatenate_terms = True
        history_length = 1

    @configclass
    class CriticCfg(ObservationsCfg.PrivilegedCfg):
        reference = ObservationTermCfg(func=reference_window)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
