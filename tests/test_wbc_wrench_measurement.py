"""Detailed PhysX contact and friction reads share mutable index buffers."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


@pytest.mark.parametrize('unloaded_second_env', [False, True])
def test_wrench_preserves_normal_contact_indices_before_friction_read(unloaded_second_env):
    path = Path(__file__).resolve().parents[1] / (
        'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/'
        't1_wbc_acc/wbc_acc_gpu_action.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_actual_wrenches')
    namespace = {'torch': torch, 'FEET': ('left', 'right')}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

    class SharedBufferView:
        def __init__(self):
            self.counts = torch.zeros((2, 1), dtype=torch.int64)
            self.starts = self.counts.clone()

        def get_contact_data(self, dt):
            self.counts[:, 0] = torch.tensor([3, 2])
            self.starts[:, 0] = torch.tensor([0, 3])
            if unloaded_second_env:
                self.counts[1, 0] = 0
            forces = torch.tensor([[10.], [20.], [30.], [40.], [50.]])
            points = torch.tensor([[1., 0., 0.]]).repeat(5, 1)
            normals = torch.tensor([[0., 0., 1.]]).repeat(5, 1)
            return forces, points, normals, None, self.counts, self.starts

        def get_friction_data(self, dt):
            self.counts[:, 0] = 1
            self.starts[:, 0] = torch.tensor([0, 1])
            if unloaded_second_env:
                self.counts[1, 0] = 0
            return (torch.tensor([[2., 0., 0.], [4., 0., 0.]]),
                    torch.tensor([[0., 1., 0.], [0., 1., 0.]]), self.counts, self.starts)

    sensors = {name: SimpleNamespace(contact_physx_view=SharedBufferView()) for name in
               ('left_foot_ground_contact', 'right_foot_ground_contact')}
    action = SimpleNamespace(
        _raw=torch.zeros((2, 2)),
        _env=SimpleNamespace(num_envs=2, physics_dt=.002, device='cpu',
                             scene=SimpleNamespace(sensors=sensors)),
        _asset=SimpleNamespace(body_names=['left', 'right'],
                               data=SimpleNamespace(body_link_pos_w=torch.zeros((2, 2, 3)))))
    actual = namespace['_actual_wrenches'](action)
    expected = torch.tensor([[2., 0., 60., 0., -60., -2.],
                             [4., 0., 90., 0., -90., -4.]])
    if unloaded_second_env:
        expected[1] = 0
    torch.testing.assert_close(actual, expected[:, None, :].repeat(1, 2, 1))
