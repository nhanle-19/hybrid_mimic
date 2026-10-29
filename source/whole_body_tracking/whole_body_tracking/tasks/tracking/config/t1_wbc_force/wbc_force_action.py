"""Two policy outputs set hard upper bounds on foot normal forces."""
import torch

from isaaclab.utils import configclass
from ..t1_wbc_acc.wbc_acc_action import WBCACCActionCfg
from ..t1_wbc_acc.wbc_acc_gpu_action import WBCACCGPUAction
from .controller_cfg import T1WBCForceControllerCfg
from whole_body_tracking.utils.wbc_force_gpu_control import WBCForceGPUQP


class WBCForceAction(WBCACCGPUAction):
    solver_type = WBCForceGPUQP
    action_name = 'wbc_force'
    log_prefix = 'WBCForce'
    contact_gating = 'learned_force_caps'

    @property
    def action_dim(self):
        return 2

    def _contact_masks(self, state):
        # Fixed model corners supply force moment arms, not contact labels.
        # Only the learned caps disable support in the shared QP assembly.
        return torch.ones((len(state['q']), 2, 4), dtype=torch.bool, device=state['q'].device)

    def set_active_contacts(self, mask):
        if mask is not None:
            raise ValueError('FORCE support is controlled by force caps, not contact-mask overrides')
        super().set_active_contacts(None)

    def _contact_measurements(self):
        # Simulator contacts are optional diagnostics, never controller inputs.
        return super()._contact_measurements() if self.cfg.record_diagnostics else None


@configclass
class WBCForceActionCfg(WBCACCActionCfg):
    class_type: type = WBCForceAction
    controller: T1WBCForceControllerCfg = T1WBCForceControllerCfg()
