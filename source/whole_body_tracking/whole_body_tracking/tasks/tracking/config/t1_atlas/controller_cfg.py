from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.t1_floating_model.controller_cfg import T1FloatingModelControllerCfg


@configclass
class T1AtlasControllerCfg(T1FloatingModelControllerCfg):
    """HybridMimic policy scales and Atlas analytical solver settings."""
    momentum_weights: tuple = (1., 1., 1., 10., 10., 10.)
    force_weight: float = 1e-5
    acceleration_weight: float = 1e-5
    # Posture is handled by joint PD, not a duplicate QP tracking objective.
    joint_acceleration_weight: float = 0.
    friction: float = .6
    contact_height_tolerance: float = .025
    contact_schedule_file: str | None = None
    residual_tolerance: float = 2e-5
    record_diagnostics: bool = False
