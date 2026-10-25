"""Cost-only translational contact authority; physical support is independent."""
import torch

try:
    from .wbc_acc_gpu_control import WBCACCGPUQP
    from .wbc_contact_mapping import decode_contact_values, encode_contact_values, contact_action_transform
except ImportError:
    from wbc_acc_gpu_control import WBCACCGPUQP
    from wbc_contact_mapping import decode_contact_values, encode_contact_values, contact_action_transform


def decode_authority_actions(actions):
    """Six raw outputs -> left/right xyz cost authorities with exact 0/1 regions."""
    if actions.ndim != 2 or actions.shape[1] != 6 or not bool(torch.isfinite(actions).all()):
        raise ValueError('WBC-AUTHORITY expects six finite logits per environment')
    return decode_contact_values(actions.reshape(-1, 2, 3))


def encode_authorities(authority):
    """Inverse for manual fractions; finite endpoint logits preserve exact 0/1."""
    if (authority.ndim != 3 or authority.shape[1:] != (2, 3)
            or not authority.is_floating_point() or not bool(torch.isfinite(authority).all())
            or not bool(((authority >= 0) & (authority <= 1)).all())):
        raise ValueError('Expected authority fractions with shape (N,2,3) in [0,1]')
    return encode_contact_values(authority).flatten(1)


class WBCAuthorityGPUQP(WBCACCGPUQP):
    action_transform = contact_action_transform()

    def __init__(self, cfg, friction=.6, tolerance=2e-5, max_iterations=100, solver_tolerance=1e-7, backend="batched"):
        super().__init__(cfg, friction, tolerance, max_iterations, solver_tolerance, backend)
        weight = torch.as_tensor(cfg.contact_authority_weights)
        if weight.shape != (3,) or not bool((torch.isfinite(weight) & (weight > 0)).all()):
            raise ValueError('contact_authority_weights must contain three finite positive weights')
        cap = torch.as_tensor(cfg.contact_force_max)
        if cap.shape != (2,) or not bool((torch.isfinite(cap) & (cap > 0)).all()):
            raise ValueError('contact_force_max must contain two finite positive fixed bounds')
        if cfg.swing_weight != 0:
            raise ValueError('WBC-AUTHORITY requires swing_weight=0')

    def decode_policy(self, actions):
        c = decode_authority_actions(actions)
        # Authority never disables contact or changes the fixed force caps.
        # The shared QP uses six spatial weights; angular weights stay zero.
        return torch.ones_like(c[:, :, 0]), torch.cat((c.square(), torch.zeros_like(c)), -1)

    def contact_objective(self, frame, desired, weights, support, tensor):
        # Translational ||diag(c) (J a + Jdot v)||^2; no reference velocity
        # target or hard no-slip equality. Airborne feet have no contact task.
        scale = tensor((*self.cfg.contact_authority_weights, 0., 0., 0.))
        return torch.zeros_like(frame['bias']), weights*scale*support[:, None]

    def solve(self, state, reference, actions, masks, limits, external=None):
        result = super().solve(state, reference, actions, masks, limits, external)
        c = decode_authority_actions(actions)
        result['contact_authority'] = c
        _, weights = self.decode_policy(actions)
        scale = actions.new_tensor((*self.cfg.contact_authority_weights, 0., 0., 0.))
        result['contact_weights'] = weights*scale*result['available'].any(-1)[..., None]
        return result
