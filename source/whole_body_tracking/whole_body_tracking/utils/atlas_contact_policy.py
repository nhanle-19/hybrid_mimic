"""Reference-contact policy conventions and collider-derived support geometry."""
import json
from pathlib import Path
import torch
from functools import lru_cache

FEET = ('left_foot_link', 'right_foot_link')
CONTACT_PATH = Path(__file__).resolve().parents[1]/'assets/booster/t1/atlas_contacts.json'


@lru_cache(maxsize=8)
def sole_vertices(device='cpu', dtype=torch.float64):
    data = json.loads(CONTACT_PATH.read_text())
    return torch.tensor([data['bodies'][f]['sole_vertices'] for f in FEET], device=device, dtype=dtype)


def decode_actions(actions, cfg):
    if actions.ndim != 2 or actions.shape[1] != 14 or not bool(torch.isfinite(actions).all()):
        raise ValueError('Expected 14 finite actions: [activation, six motion-weight logits] per foot')
    values = actions.reshape(-1, 2, 7)
    # Clipping permits exact c=0/1, including manually supplied actions.
    activation = values[..., 0].clamp(0., 1.)
    low, high = cfg.contact_weight_min, cfg.contact_weight_max
    import math
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low < high:
        raise ValueError('Motion weight bounds must satisfy finite 0 <= min < max')
    return activation, low+(high-low)*torch.sigmoid(values[..., 1:])


def available_vertices(state, cfg, origins=None):
    """Conservative flat-floor bottom-face model, from ACTUAL poses only.

    Vertices above the floor cannot carry force. Tilted feet use only the low
    edge/corner; inverted/side contacts are unsupported, never full-foot patches.
    """
    points = sole_vertices(state['q'].device, state['q'].dtype)
    masks = []
    for i, foot in enumerate(FEET):
        frame = state['frames'][foot]
        world = points[i]@frame['rotation'].transpose(-1, -2)+frame['position'][:, None]
        if origins is not None:
            world = world+origins[:, None]
        gap = world[..., 2]-cfg.ground_height
        mask = (gap <= cfg.contact_gap_tolerance) & (gap >= -cfg.max_contact_penetration)
        # Even within the contact gap tolerance, a tilted high corner must not
        # provide a full-foot moment arm. Keep only the lowest edge/corner.
        mask &= gap <= gap.amin(-1, keepdim=True)+cfg.contact_patch_height_tolerance
        mask &= (world[..., :2].abs() < 100.).all(-1)
        mask &= (frame['rotation'][:, 2, 2] > .5)[:, None]
        masks.append(mask)
    return torch.stack(masks, 1)


def finite_difference(values, dt):
    """Reference spatial acceleration in the same world-aligned frame as J."""
    if len(values) == 1:
        return torch.zeros_like(values)
    result = torch.empty_like(values)
    result[0], result[-1] = (values[1]-values[0])/dt, (values[-1]-values[-2])/dt
    if len(values) > 2:
        result[1:-1] = (values[2:]-values[:-2])/(2*dt)
    return result


def validate_contact_asset():
    """Reject a stale collider export before the first torque command."""
    import hashlib
    data = json.loads(CONTACT_PATH.read_text())
    for name, expected in data['source_layers_sha256'].items():
        matches = list(CONTACT_PATH.parent.rglob(name))
        if len(matches) != 1 or hashlib.sha256(matches[0].read_bytes()).hexdigest() != expected:
            raise ValueError(f'Atlas collider export is stale for {name}; regenerate atlas_contacts.json')
