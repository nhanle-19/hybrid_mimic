from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.t1_hybrid.controller_cfg import T1HybridControllerCfg


@configclass
class T1MomentumControllerCfg(T1HybridControllerCfg):
    """T1 parameters for the momentum-based whole-body controller."""

    joint_position_gain: float = 40.0
    joint_velocity_gain: float = 4.0

    base_acceleration_weight: float = 2.0e2
    joint_acceleration_weight: float = 1.0e-1
    force_objective_weight: float = 1.0e-3
    torque_objective_weight: float = 4.0e2
    qp_regularization: float = 1.0e-5
