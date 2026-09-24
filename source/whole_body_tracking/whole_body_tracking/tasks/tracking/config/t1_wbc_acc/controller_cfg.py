from isaaclab.utils import configclass


@configclass
class T1WBCACCControllerCfg:
    momentum_weights: tuple = (1., 1., 1., 10., 10., 10.)
    force_weight: float = 1e-5
    acceleration_weight: float = 1e-5
    com_position_gain: float = 40.
    linear_momentum_gain: float = 10.
    angular_momentum_gain: float = 15.
    posture_position_gain: float = 40.
    posture_velocity_gain: float = 8.
    posture_weight: float = .1
    pelvis_position_gain: float = 60.
    pelvis_velocity_gain: float = 12.
    pelvis_weight: float = 5.
    swing_position_gain: float = 80.
    swing_velocity_gain: float = 16.
    swing_weight: float = 0.  # Omit the Cartesian swing-foot task; retain joint posture and pelvis tracking.
    # Six nonnegative spatial-motion weights per candidate foot.
    contact_weight_min: float = 0.
    contact_weight_max: float = 100.
    contact_velocity_gain: tuple = (10., 10., 10., 5., 5., 5.)
    contact_force_max: tuple = (600., 600.)
    ground_height: float = 0.
    contact_gap_tolerance: float = .001
    contact_patch_height_tolerance: float = 1e-5
    max_contact_penetration: float = .03
    fallback_joint_damping: float = 2.
