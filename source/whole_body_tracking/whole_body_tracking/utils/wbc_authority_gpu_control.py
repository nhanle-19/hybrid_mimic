"""Directional contact authority: coupled soft acceleration and wrench limits."""
import torch

try:
    from .wbc_acc_gpu_control import WBCACCGPUQP, FEET
    from .wbc_acc_torch_model import skew
except ImportError:
    from wbc_acc_gpu_control import WBCACCGPUQP, FEET
    from wbc_acc_torch_model import skew


def decode_authority_actions(actions):
    """Twelve logits -> two six-axis authorities, without hard clipping."""
    if actions.ndim != 2 or actions.shape[1] != 12 or not bool(torch.isfinite(actions).all()):
        raise ValueError('WBC-AUTHORITY expects twelve finite logits per environment')
    return actions.reshape(-1, 2, 6).sigmoid()


def encode_authorities(authority):
    """Inverse for manual fractions; finite endpoint logits preserve exact 0/1."""
    if (authority.ndim != 3 or authority.shape[1:] != (2, 6)
            or not authority.is_floating_point() or not bool(torch.isfinite(authority).all())
            or not bool(((authority >= 0) & (authority <= 1)).all())):
        raise ValueError('Expected authority fractions with shape (N,2,6) in [0,1]')
    return torch.where(authority == 0, -1000.,
                       torch.where(authority == 1, 1000., torch.logit(authority))).flatten(1)


class WBCAuthorityGPUQP(WBCACCGPUQP):
    def __init__(self, cfg, friction=.6, tolerance=2e-5, max_iterations=100, solver_tolerance=1e-7, backend="batched"):
        super().__init__(cfg, friction, tolerance, max_iterations, solver_tolerance, backend)
        cap = torch.as_tensor(cfg.contact_wrench_max)
        weight = torch.as_tensor(cfg.contact_authority_weights)
        if cap.shape != (6,) or not bool((torch.isfinite(cap) & (cap > 0)).all()):
            raise ValueError('contact_wrench_max must contain six finite positive bounds')
        if weight.shape != (6,) or not bool((torch.isfinite(weight) & (weight > 0)).all()):
            raise ValueError('contact_authority_weights must contain six finite positive weights')
        if cfg.swing_weight != 0:
            raise ValueError('WBC-AUTHORITY requires swing_weight=0')
        if tuple(cfg.contact_force_max) != (cfg.contact_wrench_max[2],)*2:
            raise ValueError('Normal force bounds must match contact_wrench_max[2]')

    def decode_policy(self, actions):
        c = decode_authority_actions(actions)
        return c[:, :, 2], c.square()

    def contact_objective(self, frame, desired, weights, support, tensor):
        # ||diag(c) (J a + Jdot v)||^2, scaled per axis; no reference velocity
        # target or hard no-slip equality. Airborne feet have no contact task.
        return torch.zeros_like(frame['bias']), weights*tensor(self.cfg.contact_authority_weights)*support[:, None]

    def extra_wrench_constraints(self, state, points, rays, enabled, nv, actions):
        n = len(enabled)
        rows = state['M'].new_zeros((n, 2, 6, nv+32))
        for i, foot in enumerate(FEET):
            offsets = points[i]@state['frames'][foot]['rotation'].transpose(-1, -2)
            wrench_map = torch.cat((rays.expand(n, 4, -1, -1), skew(offsets)@rays), -2)
            rows[:, i, :, nv+i*16:nv+(i+1)*16] = wrench_map.transpose(1, 2).reshape(n, 6, 16)
        rows[..., nv:] *= enabled[:, None, None, :]
        # Fz already has its unilateral cap in the base QP. Bound the other
        # five wrench components in both signs while retaining friction rays.
        axes = [0, 1, 3, 4, 5]
        rows = rows[:, :, axes].flatten(1, 2)
        caps = (decode_authority_actions(actions)*actions.new_tensor(self.cfg.contact_wrench_max))[:, :, axes]
        supported = enabled.reshape(n, 2, 16).any(-1)
        caps = torch.where(supported[:, :, None], caps, 1.).flatten(1)
        return torch.cat((rows, -rows), 1), torch.cat((caps, caps), 1)

    def solve(self, state, reference, actions, masks, limits, external=None):
        result = super().solve(state, reference, actions, masks, limits, external)
        c = decode_authority_actions(actions)
        result['contact_authority'] = c
        result['wrench_limits'] = c*actions.new_tensor(self.cfg.contact_wrench_max)
        result['contact_weights'] = c.square()*actions.new_tensor(self.cfg.contact_authority_weights)*result['available'].any(-1)[..., None]
        return result
