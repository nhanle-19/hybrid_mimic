from isaaclab.utils import configclass


@configclass
class T1AtlasControllerCfg:
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
    swing_weight: float = 10.
    posture_action_scale: float = .15
    com_velocity_action_scale: float = .25
    angular_momentum_action_scale: float = .5
