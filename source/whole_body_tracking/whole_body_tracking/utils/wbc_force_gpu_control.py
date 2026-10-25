"""WBC-FORCE: per-region normal-force bounds; the default layout is two feet."""
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
