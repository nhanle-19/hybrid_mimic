"""Ground-only filtering, hysteresis and action contact routing without Isaac Sim."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
from atlas_contact import GroundContactEstimator, read_ground_forces

FEET = ('left_foot_link', 'right_foot_link')


def test_hysteresis_independent_feet_and_reset():
    estimator = GroundContactEstimator(2, on_N=10, off_N=5)
    np.testing.assert_array_equal(estimator.update([[0, 0], [0, 0]]), [[False, False], [False, False]])
    np.testing.assert_array_equal(estimator.update([[10, 9], [2, 20]]), [[True, False], [False, True]])
    # Between thresholds, loaded feet stay loaded and unloaded feet stay unloaded.
    np.testing.assert_array_equal(estimator.update([[7, 7], [7, 7]]), [[True, False], [False, True]])
    estimator.reset([0])
    np.testing.assert_array_equal(estimator.update([[7, 7], [7, 7]]), [[False, False], [False, True]])
    np.testing.assert_array_equal(estimator.update([[0, -1], [0, 5]]), np.zeros((2, 2), dtype=bool))
    estimator.update([[20, 20], [20, 20]])
    estimator.reset()
    assert not estimator.active.any()


@pytest.mark.parametrize('on,off', [(5, 5), (5, 6), (10, -1), (np.nan, 0), (10, np.inf)])
def test_invalid_thresholds(on, off):
    with pytest.raises(ValueError):
        GroundContactEstimator(1, on, off)


@pytest.mark.parametrize('force', [[[np.nan, 0]], [[0, np.inf]], [1, 2], [[1, 2, 3]]])
def test_invalid_measurement(force):
    with pytest.raises(ValueError):
        GroundContactEstimator(1).update(force)


def sensors():
    return {f'{side}_foot_ground_contact': SimpleNamespace(
        body_names=[foot], data=SimpleNamespace(force_matrix_w=torch.zeros(2, 1, 1, 3),
                                               net_forces_w=torch.full((2, 1, 3), 500.)))
            for side, foot in zip(('left', 'right'), FEET)}


def test_filter_excludes_all_object_force_and_preserves_env_order():
    scene_sensors = sensors()
    scene_sensors['left_foot_ground_contact'].data.force_matrix_w[0, 0, 0, 2] = 100
    scene_sensors['right_foot_ground_contact'].data.force_matrix_w[1, 0, 0, 2] = 50
    ground = read_ground_forces(scene_sensors, FEET, 2)
    np.testing.assert_array_equal(ground[:, :, 2], [[100, 0], [0, 50]])
    np.testing.assert_array_equal(GroundContactEstimator(2).update(ground[:, :, 2]),
                                  [[True, False], [False, True]])


@pytest.mark.parametrize('matrix', [None, torch.zeros(2, 1, 0, 3), torch.zeros(2, 2, 1, 3),
                                   torch.full((2, 1, 1, 3), float('nan'))])
def test_missing_or_invalid_ground_filter_fails_without_net_force_fallback(matrix):
    scene_sensors = sensors()
    scene_sensors['left_foot_ground_contact'].data.force_matrix_w = matrix
    with pytest.raises(RuntimeError):
        read_ground_forces(scene_sensors, FEET, 2)


@pytest.fixture
def action(monkeypatch):
    managers = ModuleType('isaaclab.managers')
    managers.ActionTerm = type('ActionTerm', (), {})
    managers.ActionTermCfg = type('ActionTermCfg', (), {})
    utils = ModuleType('isaaclab.utils')
    utils.configclass = lambda cls: cls
    controller = ModuleType('atlas_contact_test.controller_cfg')
    controller.T1AtlasControllerCfg = type('T1AtlasControllerCfg', (), {})
    for name, module in [('isaaclab.managers', managers), ('isaaclab.utils', utils),
                         ('atlas_contact_test.controller_cfg', controller)]:
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_atlas/atlas_action.py'
    spec = importlib.util.spec_from_file_location('atlas_contact_test.atlas_action', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    term = object.__new__(module.AtlasAction)
    term._env = SimpleNamespace(num_envs=2)
    term.cfg = SimpleNamespace(contact_source='ground_force')
    term.contact_estimator = GroundContactEstimator(2)
    term.contact_override = None
    term.planned_contacts = np.array([[True, False], [False, True]])
    return term


def test_action_uses_measured_support_even_when_reference_disagrees(action):
    forces = np.zeros((2, 2, 3))
    forces[:, :, 2] = [[0, 30], [40, 0]]
    chosen, estimated, reference = action._contact_masks(np.array([0, 1]), forces)
    np.testing.assert_array_equal(chosen, [[False, True], [True, False]])
    np.testing.assert_array_equal(chosen, estimated)
    np.testing.assert_array_equal(reference, [[True, False], [False, True]])
    # No fallback to reference support when measurements vanish.
    chosen, _, _ = action._contact_masks(np.array([0, 1]), np.zeros_like(forces))
    assert not chosen.any()


def test_reference_mode_and_explicit_override(action):
    action.cfg.contact_source = 'reference'
    forces = np.zeros((2, 2, 3))
    chosen, _, reference = action._contact_masks(np.array([0, 1]), forces)
    np.testing.assert_array_equal(chosen, reference)
    action.set_active_contacts(np.ones((2, 2), dtype=bool))
    chosen, _, _ = action._contact_masks(np.array([0, 1]), forces)
    assert chosen.all()
    action.set_active_contacts(None)
    action.cfg.contact_source = 'ground_force'
    chosen, _, _ = action._contact_masks(np.array([0, 1]), forces)
    assert not chosen.any()


def test_action_reset_clears_only_reset_env_contact_memory(action):
    action._env = SimpleNamespace(num_envs=2)
    action._raw = torch.ones(2, 2)
    action.pending = ['left', 'right']
    action.external = [[1], [2]]
    action.contact_estimator.update([[20, 20], [20, 20]])
    action.reset(torch.tensor([0]))
    np.testing.assert_array_equal(action.contact_estimator.active, [[False, False], [True, True]])
    assert action.pending == [None, 'right']
    assert action.external == [[], [2]]
