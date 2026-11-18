"""Bounds, physical consistency and reset isolation for FORCE2 randomization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils/wbc_force_randomization.py'
spec = importlib.util.spec_from_file_location('force_randomization', PATH)
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def test_inertial_samples_preserve_positive_definiteness_axes_and_nontrunk_coms():
    mass = torch.ones(1024, 3)
    inertia = torch.diag(torch.tensor([.01, .02, .025])).flatten().expand(1024, 3, 9).clone()
    com = torch.zeros(1024, 3, 7)
    com[..., 6] = 1
    changed_mass, changed_inertia, shifted, _, _ = dr.perturb_inertials(
        mass, inertia, com, 1, (.97, 1.03), (.95, 1.05), .003)
    assert .97 <= changed_mass.min() < changed_mass.max() <= 1.03
    eigenvalues = torch.linalg.eigvalsh(changed_inertia.reshape(-1, 3, 3))
    assert (eigenvalues > 0).all()
    ratios = eigenvalues / torch.tensor([.01, .02, .025])
    assert (ratios >= .95-1e-6).all() and (ratios <= 1.05+1e-6).all()
    torch.testing.assert_close(shifted[:, [0, 2]], com[:, [0, 2]])
    torch.testing.assert_close(shifted[..., 3:], com[..., 3:])
    assert 0 < shifted[:, 1, :3].abs().max() <= .003
    assert (changed_mass.std(0) > .01).all()
    assert not com[..., :3].any()  # Input arrays remain reusable baselines.


@pytest.mark.parametrize('bad', [float('nan'), 1.2])
def test_registered_sample_check_rejects_unexpected_model_changes(bad):
    expected = torch.ones(2, 3)
    actual = expected.clone()
    actual[1, 2] = bad
    with pytest.raises(RuntimeError, match='registered randomization'):
        dr.require_match(actual, expected, 'mass')


def test_observation_noise_preserves_orientation_and_field_units_without_mutation():
    cfg = SimpleNamespace(joint_position=.005, joint_velocity=.05, root_position=.002,
                          orientation=.005, linear_velocity=.02, angular_velocity=.02)
    data = torch.zeros(1024, 59, dtype=torch.float64)
    data[:, 49:53] = torch.tensor([.5, .5, .5, .5])
    before = data.clone()
    result = dr.state_observation_noise(data, cfg)
    torch.testing.assert_close(data, before)
    for region, bound in ((slice(0, 23), .005), (slice(23, 46), .05), (slice(46, 49), .002),
                          (slice(53, 56), .02), (slice(56, 59), .02)):
        assert 0 < (result[:, region]-data[:, region]).abs().max() <= bound
    q = result[:, 49:53]
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(1024, dtype=q.dtype))
    angle = 2*torch.acos((q*data[:, 49:53]).sum(-1).abs().clamp(max=1))
    assert angle.max() <= .005+1e-10
    assert angle.std() > 1e-4


def test_zero_noise_is_identity():
    cfg = SimpleNamespace(**dict.fromkeys(('joint_position', 'joint_velocity', 'root_position',
                                          'orientation', 'linear_velocity', 'angular_velocity'), 0.))
    data = torch.randn(4, 59)
    data[:, 49:53] = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(dr.state_observation_noise(data, cfg), data)


def test_wrench_pulses_end_at_physics_step_and_reset_only_selected_environment():
    pulses = dr.WrenchPulses(3, 'cpu')
    pulses.start(torch.tensor([0, 2]), torch.ones(2, 1, 3)*5, torch.ones(2, 1, 3)*.5, 2)
    force, _ = pulses.advance()
    assert force[0, 0, 0] == 5 and not force[1].any()
    pulses.reset(torch.tensor([0]))
    force, torque = pulses.advance()
    assert not force[0].any() and force[2, 0, 0] == 5 and torque[2, 0, 0] == .5
    force, torque = pulses.advance()
    assert not force.any() and not torque.any() and not pulses.remaining.any()


def test_invalid_sampling_ranges_fail_before_physics_mutation():
    with pytest.raises(ValueError):
        dr.uniform((4,), (1., -1.), device='cpu')
    with pytest.raises(ValueError):
        dr.uniform((4,), (0., float('nan')), device='cpu')
