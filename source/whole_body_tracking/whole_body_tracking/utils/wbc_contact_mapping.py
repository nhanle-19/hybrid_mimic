"""Current corner FORCE mapping and legacy region/AUTH mappings, without simulator imports."""
import numpy as np


CONTACT_DEAD_ZONE = 0.10
# Inside the zero region even after float32 storage and float64 QP conversion.
CONTACT_ZERO_ACTION = -3.0
CONTACT_FULL_ACTION = 3.0
FORCE_CAP_DENOMINATOR = 0.90
# sigmoid(40) rounds to 1 in float32 and float64, including the QP conversion.
FORCE_FULL_ACTION = 40.0


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


def force_action_transform():
    return {'type': 'sigmoid_lower_dead_zone', 'epsilon': CONTACT_DEAD_ZONE,
            'denominator': FORCE_CAP_DENOMINATOR, 'endpoints': [0., 1.]}


def decode_force_values(actions):
    """Eight corner caps use /0.9; archived region policies retain their mapping."""
    if actions.shape[-1] != 8:
        return decode_contact_values(actions)
    return ((actions.sigmoid()-CONTACT_DEAD_ZONE)/FORCE_CAP_DENOMINATOR).clamp(0., 1.)


def encode_force_values(capacities):
    """Encode manual corner fractions; the full cap uses floating-point saturation."""
    import torch
    if capacities.shape[-1] != 8:
        return encode_contact_values(capacities)
    probability = CONTACT_DEAD_ZONE+FORCE_CAP_DENOMINATOR*capacities
    return torch.where(capacities == 0, CONTACT_ZERO_ACTION,
                       torch.where(capacities == 1, FORCE_FULL_ACTION, torch.logit(probability)))


def decode_force_values_numpy(actions):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape[-1] != 8:
        return decode_contact_values_numpy(actions)
    probability = np.exp(-np.logaddexp(0., -actions))
    return np.clip((probability-CONTACT_DEAD_ZONE)/FORCE_CAP_DENOMINATOR, 0., 1.)
