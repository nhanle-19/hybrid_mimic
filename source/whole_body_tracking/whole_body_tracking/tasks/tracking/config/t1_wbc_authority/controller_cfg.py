"""World-aligned wrench limits at each foot origin (N, N m)."""
from isaaclab.utils import configclass
from ..t1_wbc_acc.controller_cfg import T1WBCACCControllerCfg


@configclass
class T1WBCAuthorityControllerCfg(T1WBCACCControllerCfg):
    contact_wrench_max: tuple = (360., 360., 600., 60., 60., 30.)
    contact_authority_weights: tuple = (100., 100., 100., 100., 100., 100.)
    contact_force_max: tuple = (600., 600.)
    swing_weight: float = 0.
