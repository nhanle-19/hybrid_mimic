"""Exercise bounded Gaussian means through PPO, checkpointing, and Isaac exports."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from bounded_actor_critic import BoundedMeanActorCritic


def policy(cls=BoundedMeanActorCritic):
    return cls(3, 3, 8, actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=1.)


def test_sampling_inference_and_log_probabilities_share_bounded_mean():
    torch.manual_seed(42)
    model = policy()
    with torch.no_grad():
        model.actor[2].weight.zero_()
        model.actor[2].bias.copy_(torch.linspace(-4., 4., 8))
    obs = torch.randn(256, 3)
    samples = model.act(obs)
    mean = model.act_inference(obs)
    torch.testing.assert_close(model.action_mean, mean)
    assert mean.abs().max() <= 2.5
    assert (samples.abs() > 2.5).any()  # Only the mean is bounded.
    expected = torch.distributions.Normal(mean, model.std).log_prob(samples).sum(-1)
    torch.testing.assert_close(model.get_actions_log_prob(samples), expected)
    (-model.get_actions_log_prob(samples).mean()).backward()
    assert model.actor[2].bias.grad.abs().sum() > 0
    assert model.std.grad.abs().sum() > 0


def test_checkpoint_preserves_bound_and_rejects_unbounded_policy(tmp_path):
    model = policy()
    checkpoint = tmp_path/'policy.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = policy()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    obs = torch.randn(4, 3)
    torch.testing.assert_close(restored.act_inference(obs), model.act_inference(obs))
    with pytest.raises(RuntimeError, match='mean_bound.bound'):
        restored.load_state_dict(policy(ActorCritic).state_dict())


def test_isaac_jit_and_onnx_exports_preserve_bounded_mean(tmp_path):
    # Load the actual exporters without importing Isaac Sim's task registration.
    root = Path(importlib.util.find_spec('isaaclab_rl').submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location('isaac_exporter_for_test', root/'rsl_rl/exporter.py')
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    model = policy()
    with torch.no_grad():
        model.actor[2].bias.copy_(torch.linspace(-10., 10., 8))
    exporter.export_policy_as_jit(model, None, str(tmp_path))
    exporter.export_policy_as_onnx(model, str(tmp_path))
    obs = torch.tensor([[1., -2., 3.]])
    expected = model.act_inference(obs).detach()
    scripted = torch.jit.load(str(tmp_path/'policy.pt'))
    torch.testing.assert_close(scripted(obs), expected)
    graph = onnx.load(tmp_path/'policy.onnx')
    onnx.checker.check_model(graph)
    actual = ReferenceEvaluator(graph).run(None, {'obs': obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-6)
    assert np.abs(actual).max() <= 2.5


def test_real_ppo_runner_can_update_bounded_actor(monkeypatch, tmp_path):
    import rsl_rl.modules
    monkeypatch.setattr(rsl_rl.modules, 'BoundedMeanActorCritic', BoundedMeanActorCritic, raising=False)

    class Env:
        num_envs, num_actions, device = 8, 8, 'cpu'
        cfg = {}

        def get_observations(self):
            return torch.randn(self.num_envs, 3), {'observations': {}}

        def step(self, actions):
            obs, extras = self.get_observations()
            rewards = -(actions-.5).square().mean(-1)
            return obs, rewards, torch.zeros(self.num_envs, dtype=torch.long), extras

    cfg = dict(policy=dict(class_name='rsl_rl.modules.BoundedMeanActorCritic',
                           actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=1.),
               algorithm=dict(class_name='PPO', num_learning_epochs=1, num_mini_batches=2),
               num_steps_per_env=4, save_interval=500, empirical_normalization=True)
    runner = OnPolicyRunner(Env(), cfg, log_dir=str(tmp_path), device='cpu')
    before = runner.alg.policy.actor[2].weight.detach().clone()
    runner.learn(2)
    assert not torch.equal(before, runner.alg.policy.actor[2].weight)
    assert torch.isfinite(runner.alg.policy.std).all()
    actions = runner.get_inference_policy()(torch.randn(16, 3))
    assert actions.abs().max() <= 2.5
    runner.writer.close()
