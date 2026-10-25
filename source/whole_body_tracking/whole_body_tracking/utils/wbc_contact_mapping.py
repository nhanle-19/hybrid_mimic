"""FORCE/AUTH action mapping with reachable endpoints, without simulator imports."""
import numpy as np


CONTACT_DEAD_ZONE = 0.10
# Inside the zero region even after float32 storage and float64 QP conversion.
CONTACT_ZERO_ACTION = -3.0
CONTACT_FULL_ACTION = 3.0


def contact_action_transform():
    return {'type': 'sigmoid_dead_zones', 'epsilon': CONTACT_DEAD_ZONE, 'endpoints': [0., 1.]}


def decode_contact_values(actions):
    """Torch mapping; PPO retains the raw Gaussian actions for log probabilities."""
    return ((actions.sigmoid()-CONTACT_DEAD_ZONE)/(1.-2.*CONTACT_DEAD_ZONE)).clamp(0., 1.)


def encode_contact_values(values):
    """Torch inverse for validated fractions; endpoints use ordinary raw outputs."""
    import torch
    probability = CONTACT_DEAD_ZONE+(1.-2.*CONTACT_DEAD_ZONE)*values
    return torch.where(values == 0, CONTACT_ZERO_ACTION,
                       torch.where(values == 1, CONTACT_FULL_ACTION, torch.logit(probability)))


def decode_contact_values_numpy(actions):
    actions = np.asarray(actions, dtype=np.float64)
    probability = np.exp(-np.logaddexp(0., -actions))
    return np.clip((probability-CONTACT_DEAD_ZONE)/(1.-2.*CONTACT_DEAD_ZONE), 0., 1.)


def encode_contact_values_numpy(capacities):
    """Encode desired fractions, including exact zero, as finite raw outputs."""
    capacities = np.asarray(capacities, dtype=np.float64)
    if not np.isfinite(capacities).all() or not ((capacities >= 0) & (capacities <= 1)).all():
        raise ValueError('Expected finite contact fractions in [0,1]')
    probability = CONTACT_DEAD_ZONE+(1.-2.*CONTACT_DEAD_ZONE)*capacities
    with np.errstate(divide='ignore'):
        actions = np.log(probability)-np.log1p(-probability)
    return np.where(capacities == 0, CONTACT_ZERO_ACTION,
                    np.where(capacities == 1, CONTACT_FULL_ACTION, actions))
