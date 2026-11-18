"""Eight policy outputs set independent normal-force bounds on sole corners."""
import torch

from isaaclab.utils import configclass
from ..t1_wbc_acc.wbc_acc_action import WBCACCActionCfg
from ..t1_wbc_acc.wbc_acc_gpu_action import WBCACCGPUAction
from .controller_cfg import T1WBCForceControllerCfg
from whole_body_tracking.utils.wbc_force_gpu_control import WBCCornerForceGPUQP
from whole_body_tracking.utils.wbc_force_randomization import WrenchPulses
from whole_body_tracking.utils.wbc_acc_torch_model import rotation_xyzw, mv


class WBCForceAction(WBCACCGPUAction):
    solver_type = WBCCornerForceGPUQP
    action_name = 'wbc_force'
    log_prefix = 'WBCForce'
    contact_gating = 'learned_force_caps'

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.normal_force_max = self.solver.force_maximum(self.dynamics.mass.sum(), self.dynamics.gravity).expand(8)
        self._wrench_pulses = WrenchPulses(env.num_envs, env.device) if cfg.domain_randomization else None

    def _states(self):
        state = super()._states()
        if self.cfg.domain_randomization and (self.cfg.record_diagnostics or not self.validated):
            data = self._asset.data
            ids = [self._asset.body_names.index(name) for name in self.dynamics.body_names]
            position = data.body_com_pos_w[:, ids].double()
            velocity = data.body_com_vel_w[:, ids].double()
            rotation = rotation_xyzw(data.body_link_quat_w[:, ids][..., [1, 2, 3, 0]].double())
            mass = self._physical_mass[..., None]
            inertia = rotation @ self._physical_inertia @ rotation.transpose(-1, -2)
            com = (mass * position).sum(1) / mass.sum(1)
            linear = mass * velocity[..., :3]
            angular = mv(inertia, velocity[..., 3:]) + torch.cross(position-com[:, None], linear, dim=-1)
            state['measured_momentum'] = torch.cat((linear.sum(1), angular.sum(1)), -1)
        return state

    def _validate_simulator_model(self, states):
        if not self.cfg.domain_randomization:
            return super()._validate_simulator_model(states)
        from .domain_randomization import validate_randomized_physics
        validate_randomized_physics(self, states)

    def apply_actions(self):
        if self._wrench_pulses is not None:
            force, torque = self._wrench_pulses.advance()
            self._asset.set_external_force_and_torque(
                force, torque, body_ids=[self._asset.body_names.index('Trunk')])
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self._wrench_pulses is not None:
            self._wrench_pulses.reset(slice(None) if env_ids is None else env_ids)

    @property
    def action_dim(self):
        return 8

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
    domain_randomization: bool = False
