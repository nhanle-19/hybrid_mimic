"""Reference-only actor inputs and privileged critic observations."""
import torch
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg


def reference_window(env, offsets=(0, 1, 5, 10)):
    command = env.command_manager.get_term('motion')
    motion = command.motion
    ids = (command.time_steps[:, None]+torch.tensor(offsets, device=env.device)[None]).clamp(0, motion.time_step_total-1)
    # No robot state, state-relative transforms, previous actions, or sensors.
    values = [motion.joint_pos[ids], motion.joint_vel[ids]]
    for name in ('_body_pos_w', '_body_quat_w', '_body_lin_vel_w', '_body_ang_vel_w'):
        values.append(getattr(motion, name)[ids].flatten(2))
    return torch.cat(values, -1).flatten(1)


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
