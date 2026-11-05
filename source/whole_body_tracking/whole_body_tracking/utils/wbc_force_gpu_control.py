"""Independent corner force caps for feet, plus legacy region caps for crawl."""
import math
import torch

try:
    from .wbc_acc_gpu_control import WBCACCGPUQP
    from .wbc_contact_mapping import decode_contact_values, encode_contact_values, contact_action_transform
except ImportError:
    from wbc_acc_gpu_control import WBCACCGPUQP
    from wbc_contact_mapping import decode_contact_values, encode_contact_values, contact_action_transform


def decode_force_actions(actions, contact_count=2):
    """Map raw outputs to force fractions with reachable exact 0/1 regions."""
    if actions.ndim != 2 or actions.shape[1] != contact_count or not bool(torch.isfinite(actions).all()):
        count = 'two' if contact_count == 2 else str(contact_count)
        raise ValueError(f'WBC-FORCE expects {count} finite force-limit actions per environment')
    return decode_contact_values(actions)


def encode_force_capacities(capacities, contact_count=2):
    """Convert manual fractions to raw outputs, preserving both endpoints."""
    if (capacities.ndim != 2 or capacities.shape[1] != contact_count
            or not capacities.is_floating_point()
            or not bool(torch.isfinite(capacities).all())
            or not bool(((capacities >= 0) & (capacities <= 1)).all())):
        raise ValueError(f'Expected {contact_count} finite force capacities in [0,1] per environment')
    return encode_contact_values(capacities)


class WBCForceGPUQP(WBCACCGPUQP):
    """Region-total force bounds used by the separate crawl controller."""
    action_transform = contact_action_transform()

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.swing_weight != 0:
            raise ValueError('WBC-FORCE requires swing_weight=0: foot-acceleration objectives are disabled')
        limits = torch.as_tensor(cfg.contact_force_max)
        if limits.ndim != 1 or len(limits) < 1 or not bool(torch.isfinite(limits).all() & (limits > 0).all()):
            raise ValueError('WBC-FORCE contact_force_max must contain finite positive limits')
        self.contact_count = len(limits)

    def decode_policy(self, actions):
        # None removes foot-acceleration terms from QP assembly entirely.
        return decode_force_actions(actions, self.contact_count), None


class WBCCornerForceGPUQP(WBCACCGPUQP):
    """Eight NN outputs cap four sole points per foot independently."""
    action_transform = contact_action_transform()
    force_cap_scope = 'per_sole_vertex'
    action_names = tuple(f'{side}_corner_{corner}' for side in ('left', 'right') for corner in range(4))

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.swing_weight != 0:
            raise ValueError('WBC-FORCE requires swing_weight=0: foot-acceleration objectives are disabled')
        self.maximum_body_weights = cfg.point_force_max_body_weights
        if not math.isfinite(self.maximum_body_weights) or self.maximum_body_weights <= 0:
            raise ValueError('point_force_max_body_weights must be finite and positive')

    def force_maximum(self, mass, gravity):
        # Floor-normal force is world Z; mass is the complete robot model mass.
        return mass*(-gravity[2])*self.maximum_body_weights

    def force_capacities(self, state, activation):
        return activation*self.force_maximum(state['mass'], state['gravity'])

    def decode_policy(self, actions):
        return decode_force_actions(actions, 8).reshape(-1, 2, 4), None
