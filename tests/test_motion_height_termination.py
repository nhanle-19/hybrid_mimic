"""Height checks must work before the first command update following a reset."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def height_check():
    # Load the actual tensor-only function without launching Isaac Sim.
    path = Path(__file__).resolve().parents[1] / (
        'source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py')
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'bad_motion_body_pos_z_only')
    namespace = {'torch': torch, '_get_body_indexes': lambda command, names:
                 [i for i, name in enumerate(command.cfg.body_names) if names is None or name in names]}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[function.name]


def test_first_step_uses_live_reference_and_still_detects_height_error():
    reference = torch.tensor([[[0., 0., .65], [0., .2, .7], [0., -.1, .03]]]).repeat(2, 1, 1)
    command = SimpleNamespace(cfg=SimpleNamespace(body_names=['Trunk', 'hand', 'foot']),
                              body_pos_w=reference, robot_body_pos_w=reference.clone(),
                              body_pos_relative_w=torch.zeros_like(reference))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    check = height_check()
    assert not check(env, 'motion', .25, ['hand', 'foot']).any()
    command.robot_body_pos_w[1, 1, 2] -= .3
    assert check(env, 'motion', .25, ['hand', 'foot']).tolist() == [False, True]
    assert not check(env, 'motion', .25, ['foot']).any()
    # A later reset may leave a previous episode's aligned cache behind.
    command.body_pos_relative_w.fill_(99.)
    command.robot_body_pos_w.copy_(reference)
    assert not check(env, 'motion', .25, ['hand', 'foot']).any()
