"""Check unrestricted FORCE means and fixed noise through PPO, resume, and export."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from bounded_actor_critic import BoundedMeanActorCritic
from fixed_std_actor_critic import FixedStdActorCritic


def policy(cls=FixedStdActorCritic, **kwargs):
    return cls(3, 3, 8, actor_hidden_dims=[16], critic_hidden_dims=[16], **kwargs)


def test_sampling_inference_and_log_probabilities_use_unrestricted_mean_and_unit_noise():
    torch.manual_seed(42)
    model = policy()
    with torch.no_grad():
        model.actor[2].weight.zero_()
        model.actor[2].bias.copy_(torch.linspace(-10., 10., 8))
    obs = torch.randn(4096, 3)
    samples = model.act(obs)
    mean = model.act_inference(obs)
    torch.testing.assert_close(mean, model.actor[2].bias.expand_as(mean))
    torch.testing.assert_close(model.action_mean, mean)
    torch.testing.assert_close(model.action_std, torch.ones_like(mean), rtol=0, atol=0)
    torch.testing.assert_close((samples-mean).std(0), torch.ones(8), rtol=0, atol=.05)
    expected = torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(samples).sum(-1)
    torch.testing.assert_close(model.get_actions_log_prob(samples), expected)
    (-model.get_actions_log_prob(samples).mean()).backward()
    assert (model.actor[2].bias.grad.abs() > 0).all()
    assert not model.std.requires_grad
    assert 'std' not in dict(model.named_parameters())
    assert 'std' in dict(model.named_buffers())
    assert not model.entropy.requires_grad


@pytest.mark.parametrize('kwargs', [{'init_noise_std': .5}, {'init_noise_std': float('nan')},
                                   {'noise_std_type': 'log'}])
def test_non_unit_or_trainable_log_noise_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError, match='fixed at 1.0'):
        policy(**kwargs)


def test_checkpoint_preserves_unit_noise_and_rejects_incompatible_policy(tmp_path):
    model = policy().double()
    checkpoint = tmp_path/'policy.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = policy().double()
    assert restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    obs = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(restored.act_inference(obs), model.act_inference(obs))
    torch.testing.assert_close(restored.std, torch.ones(8, dtype=torch.float64), rtol=0, atol=0)
    assert 'std' not in dict(restored.named_parameters())
    invalid = model.state_dict()
    invalid['std'] = torch.full_like(model.std, 40.)
    with pytest.raises(ValueError, match='checkpoint noise fixed at 1.0'):
        restored.load_state_dict(invalid)
    torch.testing.assert_close(restored.std, torch.ones_like(restored.std), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='mean_bound.bound'):
        restored.load_state_dict(policy(BoundedMeanActorCritic).state_dict())


def test_isaac_jit_and_onnx_exports_preserve_unrestricted_mean(tmp_path):
    root = Path(importlib.util.find_spec('isaaclab_rl').submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location('fixed_std_isaac_exporter_for_test', root/'rsl_rl/exporter.py')
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
    assert actual.min() < -2.5 and actual.max() > 2.5


def test_real_ppo_updates_and_resumes_without_changing_noise(monkeypatch, tmp_path):
    import rsl_rl.modules
    monkeypatch.setattr(rsl_rl.modules, 'FixedStdActorCritic', FixedStdActorCritic, raising=False)

    class Env:
        num_envs, num_actions, device = 8, 8, 'cpu'
        cfg = {}

        def get_observations(self):
            return torch.randn(self.num_envs, 3), {'observations': {}}

        def step(self, actions):
            obs, extras = self.get_observations()
            rewards = -(actions-.5).square().mean(-1)
            return obs, rewards, torch.zeros(self.num_envs, dtype=torch.long), extras

    def runner(log_dir):
        cfg = dict(policy=dict(class_name='rsl_rl.modules.FixedStdActorCritic',
                               actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=1.),
                   algorithm=dict(class_name='PPO', num_learning_epochs=2, num_mini_batches=2,
                                  entropy_coef=.5),
                   num_steps_per_env=4, save_interval=500, empirical_normalization=True)
        return OnPolicyRunner(Env(), cfg, log_dir=str(log_dir), device='cpu')

    torch.manual_seed(42)
    first = runner(tmp_path/'first')
    before = first.alg.policy.actor[2].weight.detach().clone()
    first.learn(2)
    assert not torch.equal(before, first.alg.policy.actor[2].weight)
    torch.testing.assert_close(first.alg.policy.std, torch.ones(8), rtol=0, atol=0)
    assert all(p is not first.alg.policy.std for g in first.alg.optimizer.param_groups for p in g['params'])
    checkpoint = tmp_path/'resume.pt'
    first.save(str(checkpoint))
    resumed = runner(tmp_path/'resumed')
    resumed.load(str(checkpoint))
    obs = torch.randn(16, 3)
    torch.testing.assert_close(resumed.get_inference_policy()(obs), first.get_inference_policy()(obs))
    before = resumed.alg.policy.actor[2].weight.detach().clone()
    resumed.learn(2)
    assert not torch.equal(before, resumed.alg.policy.actor[2].weight)
    torch.testing.assert_close(resumed.alg.policy.std, torch.ones(8), rtol=0, atol=0)
    first.writer.close()
    resumed.writer.close()
