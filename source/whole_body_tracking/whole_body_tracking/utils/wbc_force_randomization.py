"""Small simulator/model perturbations and structured FORCE2 observation noise."""
import math
import torch


def uniform(shape, bounds, *, device, dtype=torch.float32):
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f'Invalid randomization range: {bounds}')
    return torch.rand(shape, device=device, dtype=dtype) * (high-low) + low


def perturb_inertials(masses, inertias, coms, trunk_index, mass_range, inertia_range, com_range):
    """Scale each body's mass/inertia; shift only the trunk COM, without changing axes."""
    if min(mass_range) <= 0 or min(inertia_range) <= 0 or com_range < 0:
        raise ValueError('Mass/inertia scales must be positive and COM range nonnegative')
    mass_scale = uniform(masses.shape, mass_range, device=masses.device, dtype=masses.dtype)
    inertia_scale = uniform(masses.shape, inertia_range, device=masses.device, dtype=masses.dtype)
    shifted = coms.clone()
    shifted[:, trunk_index, :3] += uniform((len(masses), 3), (-com_range, com_range),
                                          device=coms.device, dtype=coms.dtype)
    return masses * mass_scale, inertias * inertia_scale[..., None], shifted, mass_scale, inertia_scale


def require_match(actual, expected, name, atol=1e-5):
    actual, expected = torch.as_tensor(actual).cpu(), torch.as_tensor(expected).cpu()
    if not torch.isfinite(actual).all() or not torch.allclose(actual, expected.to(actual), atol=atol, rtol=1e-5):
        raise RuntimeError(f'FORCE2 simulator {name} differs from the registered randomization sample')


def state_observation_noise(data, cfg):
    """Perturb one joint/root frame before history stacking; preserve unit WXYZ quaternions."""
    if data.shape[-1] < 15 or (data.shape[-1]-13) % 2:
        raise ValueError('Expected joint positions/velocities and a 13-value root state')
    n = (data.shape[-1]-13)//2
    result = data.clone()
    for region, amplitude in ((slice(0, n), cfg.joint_position), (slice(n, 2*n), cfg.joint_velocity),
                              (slice(2*n, 2*n+3), cfg.root_position),
                              (slice(2*n+7, 2*n+10), cfg.linear_velocity),
                              (slice(2*n+10, 2*n+13), cfg.angular_velocity)):
        result[..., region] += uniform(result[..., region].shape, (-amplitude, amplitude),
                                       device=data.device, dtype=data.dtype)
    # An axis-angle perturbation avoids adding arbitrary noise to quaternion components.
    axis = torch.randn((*data.shape[:-1], 3), device=data.device, dtype=data.dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    angle = uniform((*data.shape[:-1], 1), (-cfg.orientation, cfg.orientation),
                    device=data.device, dtype=data.dtype)
    w, xyz = torch.cos(angle/2), axis * torch.sin(angle/2)
    q = data[..., 2*n+3:2*n+7]
    rotated = torch.cat((w*q[..., :1]-(xyz*q[..., 1:]).sum(-1, keepdim=True),
                         w*q[..., 1:]+q[..., :1]*xyz+torch.cross(xyz, q[..., 1:], dim=-1)), -1)
    result[..., 2*n+3:2*n+7] = rotated / rotated.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return result


class WrenchPulses:
    """Per-environment finite-duration disturbances, advanced once per physics step."""
    def __init__(self, count, device):
        self.remaining = torch.zeros(count, dtype=torch.long, device=device)
        self.force = torch.zeros((count, 1, 3), device=device)
        self.torque = torch.zeros_like(self.force)

    def start(self, ids, force, torque, steps):
        if steps < 1:
            raise ValueError('A disturbance must last at least one physics step')
        self.force[ids], self.torque[ids], self.remaining[ids] = force, torque, steps

    def advance(self):
        active = (self.remaining > 0)[:, None, None]
        force, torque = self.force * active, self.torque * active
        self.remaining.sub_(1).clamp_min_(0)
        return force, torque

    def reset(self, ids):
        self.force[ids] = 0
        self.torque[ids] = 0
        self.remaining[ids] = 0
