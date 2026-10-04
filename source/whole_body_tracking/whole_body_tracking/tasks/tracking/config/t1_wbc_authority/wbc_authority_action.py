"""Six sigmoid-mapped authorities per foot."""
from isaaclab.utils import configclass
from ..t1_wbc_acc.wbc_acc_action import WBCACCActionCfg
from ..t1_wbc_acc.wbc_acc_gpu_action import WBCACCGPUAction
from .controller_cfg import T1WBCAuthorityControllerCfg
from whole_body_tracking.utils.wbc_authority_gpu_control import WBCAuthorityGPUQP


class WBCAuthorityAction(WBCACCGPUAction):
    solver_type = WBCAuthorityGPUQP
    action_name = 'wbc_authority'
    log_prefix = 'WBCAuthority'

    @property
    def action_dim(self):
        return 12


@configclass
class WBCAuthorityActionCfg(WBCACCActionCfg):
    class_type: type = WBCAuthorityAction
    # Optimality tolerance in equilibrated coordinates; physical residuals
    # still must meet residual_tolerance independently.
    batched_tolerance: float = 1e-7
    controller: T1WBCAuthorityControllerCfg = T1WBCAuthorityControllerCfg()
