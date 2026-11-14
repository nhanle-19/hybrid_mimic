"""Shared FORCE gains for two-foot training and eight-region crawl replay."""
from isaaclab.utils import configclass
from ..t1_wbc_acc.controller_cfg import T1WBCACCControllerCfg


@configclass
class T1WBCForceControllerCfg(T1WBCACCControllerCfg):
    posture_position_gain: float = 1600.
    posture_velocity_gain: float = 40.
    contact_force_max: tuple = (600., 600.)
    # Two-foot policy: each of eight corner caps can reach one robot body weight.
    # contact_force_max remains the region-total setting for legacy crawl replay.
    point_force_max_body_weights: float = 1.
    swing_weight: float = 0.
