"""Regression checks for lift lead-in, weighted metrics, and phase sampling."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/whole_body_tracking/whole_body_tracking'
spec = importlib.util.spec_from_file_location('force_learning_test', SOURCE/'utils/wbc_force_learning.py')
learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning)


def test_unloading_anticipates_liftoff_without_delaying_touchdown_or_wrapping():
    swing = torch.tensor([[0.], [0.], [0.], [1.], [1.], [0.], [0.]])
    actual = learning.unloading_weights(swing, .04, .02)
    torch.testing.assert_close(actual[:, 0], torch.tensor([0., 1/3, 2/3, 1., 1., 0., 0.]))
    torch.testing.assert_close(learning.unloading_weights(swing, 0., .02), swing)
    # A swing at the beginning must never leak into the end of a clip.
    swing[0] = 1.
    assert learning.unloading_weights(swing, .04, .02)[-1] == 0


@pytest.mark.parametrize('lead,dt', [(-.01, .02), (float('nan'), .02), (.02, 0)])
def test_invalid_lead_time_is_rejected(lead, dt):
    with pytest.raises(ValueError):
        learning.unloading_weights(torch.zeros(5, 2), lead, dt)


def test_metrics_weight_samples_instead_of_reset_batches_and_omit_empty_conditions():
    metrics = learning.ForceMetrics()
    metrics.mean('success', torch.ones(9)*100)
    metrics.mean('success', torch.zeros(1))
    metrics.mean('no_swing', torch.ones(4), torch.zeros(4, dtype=torch.bool))
    metrics.count('episodes', torch.ones(9))
    metrics.count('episodes', torch.ones(1))
    assert metrics.drain() == {'WBCForce/success': 90., 'WBCForce/episodes': 10.}
    assert metrics.drain() == {}


def sampling_method():
    tree = ast.parse((SOURCE/'tasks/tracking/mdp/commands.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionCommand')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_adaptive_sampling')
    method.args.args[1].annotation = None
    namespace = {'torch': torch, 'math': math, 'sample_uniform': lambda low, high, shape, device: low+(high-low)*torch.rand(shape, device=device)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<sampling>', 'exec'), namespace)
    return namespace['_adaptive_sampling']


@pytest.mark.parametrize('probability', [0., .2, 1.])
def test_beginning_mixture_keeps_full_motion_attempts_when_last_bin_dominates(probability):
    count = 10000
    cfg = NS(start_from_beginning_probability=probability, adaptive_uniform_ratio=.1, adaptive_kernel_size=1)
    command = NS(cfg=cfg, device='cpu', bin_count=6, bin_failed_count=torch.tensor([0., 0., 0., 0., 0., 1.e6]),
                 _current_bin_failed=torch.zeros(6), kernel=torch.ones(1),
                 time_steps=torch.zeros(count, dtype=torch.long), motion=NS(time_step_total=261),
                 _env=NS(termination_manager=NS(terminated=torch.zeros(count, dtype=torch.bool))),
                 metrics={name: torch.zeros(count) for name in ['sampling_entropy', 'sampling_top1_prob', 'sampling_top1_bin']})
    torch.manual_seed(123)
    sampling_method()(command, torch.arange(count))
    assert abs(float((command.time_steps == 0).float().mean())-probability) < .015
    assert (command.time_steps >= 0).all() and (command.time_steps < 261).all()
    assert command.metrics['sampling_top1_prob'][0] == pytest.approx(max(probability, 1-probability), abs=1e-5)


def test_force_reset_executes_requested_start_before_advancing():
    tree = ast.parse((SOURCE/'tasks/tracking/config/t1_wbc_force/motion.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    class Parent:
        def _update_command(self):
            self.time_steps += 1

    namespace = {'WBCACCTrainingMotion': Parent, 'torch': torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<force_motion>', 'exec'), namespace)
    command = namespace['WBCForceTrainingMotion'].__new__(namespace['WBCForceTrainingMotion'])
    command.motion = NS(clip_bounds=lambda frames: (torch.zeros_like(frames), torch.full_like(frames, 261)))
    command._env = NS(episode_length_buf=torch.tensor([0, 0, 5]))
    command.time_steps = torch.tensor([0, 100, 5])
    command._update_command()
    assert command.time_steps.tolist() == [0, 100, 6]
    command._env.episode_length_buf += 1
    command._update_command()
    assert command.time_steps.tolist() == [1, 101, 7]


def test_force_runner_logs_weighted_metrics_and_preserves_other_tasks(capsys):
    tree = ast.parse((SOURCE/'utils/my_on_policy_runner.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotionOnPolicyRunner')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'log']

    class Parent:
        def log(self, locs, width, pad):
            self.forwarded = locs

    namespace = {'OnPolicyRunner': Parent}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<runner>', 'exec'), namespace)
    runner = namespace['MotionOnPolicyRunner']()
    metrics = learning.ForceMetrics()
    metrics.mean('Episode/suffix_success_percent', torch.tensor([100., 100., 0.]))
    runner.env = NS(unwrapped=NS(_wbc_force_metrics=metrics))
    runner.alg = NS(policy=NS(std=torch.tensor([1., 2.])))
    written = {}
    runner.writer = NS(add_scalar=lambda name, value, iteration: written.update({name: value}))
    infos = [{'stale_episode_average': 123.}]
    locs = {'it': 1, 'ep_infos': infos}
    runner.log(locs)
    assert written['WBCForce/Episode/suffix_success_percent'] == pytest.approx(200/3)
    assert written['WBCForce/Policy/right_noise_std'] == 2.
    assert runner.forwarded['ep_infos'] == []
    assert locs['ep_infos'] is infos
    assert 'n/a' in capsys.readouterr().out
    del runner.env.unwrapped._wbc_force_metrics
    runner.log(locs)
    assert runner.forwarded is locs
