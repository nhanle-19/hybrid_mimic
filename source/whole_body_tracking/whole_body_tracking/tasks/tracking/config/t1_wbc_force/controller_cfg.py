"""WBC-ACC dynamics/tracking defaults with no foot-acceleration objectives."""
from isaaclab.utils import configclass
from ..t1_wbc_acc.controller_cfg import T1WBCACCControllerCfg


@configclass
class T1WBCForceControllerCfg(T1WBCACCControllerCfg):
    contact_force_max: tuple = (600., 600.)
    swing_weight: float = 0.
