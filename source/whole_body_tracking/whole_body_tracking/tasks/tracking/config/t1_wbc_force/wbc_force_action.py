"""Two policy outputs set hard upper bounds on foot normal forces."""
from isaaclab.utils import configclass
from ..t1_wbc_acc.wbc_acc_action import WBCACCActionCfg
from ..t1_wbc_acc.wbc_acc_gpu_action import WBCACCGPUAction
from .controller_cfg import T1WBCForceControllerCfg
from whole_body_tracking.utils.wbc_force_gpu_control import WBCForceGPUQP


class WBCForceAction(WBCACCGPUAction):
    solver_type = WBCForceGPUQP
    action_name = 'wbc_force'
    log_prefix = 'WBCForce'

    @property
    def action_dim(self):
        return 2


@configclass
class WBCForceActionCfg(WBCACCActionCfg):
    class_type: type = WBCForceAction
    controller: T1WBCForceControllerCfg = T1WBCForceControllerCfg()
