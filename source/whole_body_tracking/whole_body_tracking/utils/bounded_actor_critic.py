"""Gaussian policy with a bounded mean shared by sampling and exported inference."""
import torch
from torch import nn
from rsl_rl.modules import ActorCritic


class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        # Persist the bound so a legacy unbounded checkpoint cannot silently load.
        self.register_buffer('bound', torch.tensor(2.5))

    def forward(self, values):
        return self.bound * torch.tanh(values)


class BoundedMeanActorCritic(ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the first Linear accessible to the Isaac Lab ONNX/JIT exporters.
        # RSL-RL uses this same actor for Gaussian means and act_inference().
        self.actor.add_module('mean_bound', ScaledTanh())
