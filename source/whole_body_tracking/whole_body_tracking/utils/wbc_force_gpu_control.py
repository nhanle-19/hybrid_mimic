"""WBC-FORCE: learn per-foot normal-force bounds without foot-motion objectives."""
import torch

try:
    from .wbc_acc_gpu_control import WBCACCGPUQP
except ImportError:
    from wbc_acc_gpu_control import WBCACCGPUQP


def decode_force_actions(actions):
    """Smoothly map two policy logits to positive foot force fractions."""
    if actions.ndim != 2 or actions.shape[1] != 2 or not bool(torch.isfinite(actions).all()):
        raise ValueError('WBC-FORCE expects two finite force-limit actions per environment')
    return actions.sigmoid()


def encode_force_capacities(capacities):
    """Convert manual fractions to logits, preserving endpoints in float64."""
    if (capacities.ndim != 2 or capacities.shape[1] != 2
            or not capacities.is_floating_point()
            or not bool(torch.isfinite(capacities).all())
            or not bool(((capacities >= 0) & (capacities <= 1)).all())):
        raise ValueError('Expected two finite force capacities in [0,1] per environment')
    # Finite endpoint logits whose sigmoid rounds to exactly 0/1 in the QP.
    return torch.where(capacities == 0, -1000.,
                       torch.where(capacities == 1, 1000., torch.logit(capacities)))


class WBCForceGPUQP(WBCACCGPUQP):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.swing_weight != 0:
            raise ValueError('WBC-FORCE requires swing_weight=0: foot-acceleration objectives are disabled')
        limits = torch.as_tensor(cfg.contact_force_max)
        if limits.shape != (2,) or not bool(torch.isfinite(limits).all() & (limits > 0).all()):
            raise ValueError('WBC-FORCE contact_force_max must contain two finite positive limits')

    def decode_policy(self, actions):
        # None removes foot-acceleration terms from QP assembly entirely.
        return decode_force_actions(actions), None
