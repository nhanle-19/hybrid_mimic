"""Gaussian policy with an unrestricted mean and fixed unit exploration noise."""
import torch
from rsl_rl.modules import ActorCritic


class FixedStdActorCritic(ActorCritic):
    def __init__(self, *args, init_noise_std=1., noise_std_type='scalar', **kwargs):
        if init_noise_std != 1. or noise_std_type != 'scalar':
            raise ValueError('FixedStdActorCritic requires scalar noise fixed at 1.0')
        super().__init__(*args, init_noise_std=1., noise_std_type='scalar', **kwargs)
        # Keep ActorCritic's linear output for sampling, inference, and exports.
        # A buffer follows device moves and checkpoints but is absent from PPO's
        # parameters, so neither the policy loss nor entropy can change the noise.
        fixed_std = torch.ones_like(self.std)
        del self.std
        self.register_buffer('std', fixed_std)

    def load_state_dict(self, state_dict, strict=True):
        std = state_dict.get('std')
        if std is not None and not torch.equal(std, torch.ones_like(std)):
            raise ValueError('FixedStdActorCritic requires checkpoint noise fixed at 1.0')
        return super().load_state_dict(state_dict, strict=strict)
