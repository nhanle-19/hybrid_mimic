"""FORCE2 lookahead exposes body kinematics without explicit future joints."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observations = load('force2_observations', SOURCE/'tasks/tracking/config/t1_wbc_force2/observations.py')
MotionLibrary = load('force2_motion_library', SOURCE/'utils/motion_library.py').MotionLibrary
BODY_FIELDS = (('_body_pos_w', 3), ('_body_quat_w', 4),
               ('_body_lin_vel_w', 3), ('_body_ang_vel_w', 3))


def make_motion():
    motion = NS(time_step_total=12,
                joint_pos=torch.arange(12*23).reshape(12, 23).float(),
                joint_vel=torch.arange(12*23).reshape(12, 23).float()+1000)
    for i, (name, width) in enumerate(BODY_FIELDS):
        setattr(motion, name, torch.arange(12*24*width).reshape(12, 24, width).float()+2000*(i+1))
    return motion


def make_env(motion, frames):
    command = NS(motion=motion, time_steps=torch.tensor(frames), robot_anchor_body_index=7)
    return NS(device='cpu', command_manager=NS(get_term=lambda _: command))


def body_frame(motion, frame):
    return torch.cat([getattr(motion, name)[frame].flatten() for name, _ in BODY_FIELDS])


@pytest.mark.parametrize('start', [0, 10, 11])
def test_current_reference_preserved_and_future_contains_every_body(start):
    motion = make_motion()
    env = make_env(motion, [start])
    result = observations.reference_window(env)
    assert result.shape == (1, 59+4*24*13)
    current = torch.cat((motion.joint_pos[start], motion.joint_vel[start],
                         *(getattr(motion, name)[start, 7] for name, _ in BODY_FIELDS)))
    torch.testing.assert_close(result[0, :59], current)
    for i, offset in enumerate((1, 3, 5, 10)):
        torch.testing.assert_close(result[0, 59+i*312:59+(i+1)*312], body_frame(motion, min(start+offset, 11)))


def test_future_joint_fields_cannot_leak_and_current_joints_remain():
    motion = make_motion()
    env = make_env(motion, [0])
    before = observations.reference_window(env)
    motion.joint_pos[1:] = float('nan')
    motion.joint_vel[1:] = float('nan')
    env.robot_state = torch.full((1, 100), float('nan'))
    torch.testing.assert_close(observations.reference_window(env), before)
    motion.joint_pos[0, 2] += 7.
    after = observations.reference_window(env)
    assert after[0, 2] == before[0, 2]+7.
    torch.testing.assert_close(after[:, 59:], before[:, 59:])


def test_future_only_window_does_not_read_joint_arrays():
    motion = make_motion()
    del motion.joint_pos, motion.joint_vel
    result = observations.reference_window(make_env(motion, [0, 11]), offsets=(1, 3))
    assert result.shape == (2, 624)
    torch.testing.assert_close(result[0], torch.cat((body_frame(motion, 1), body_frame(motion, 3))))


def test_lookahead_clamps_within_each_motion_clip(tmp_path):
    source = make_motion()
    for name, frames in [('a', slice(0, 3)), ('b', slice(3, 12))]:
        values = {key: getattr(source, key)[frames].numpy() for key in ('joint_pos', 'joint_vel')}
        values.update({key[1:]: getattr(source, key)[frames].numpy() for key, _ in BODY_FIELDS})
        np.savez(tmp_path/f'{name}.npz', fps=50., **values,
                 joint_names=np.asarray([f'j{i}' for i in range(23)]),
                 body_names=np.asarray([f'b{i}' for i in range(24)]))
    motion = MotionLibrary(tmp_path)
    result = observations.reference_window(make_env(motion, [1, 2, 3, 11]))
    for row, start in enumerate((1, 2, 3, 11)):
        end = 2 if start < 3 else 11
        for i, offset in enumerate((1, 3, 5, 10)):
            torch.testing.assert_close(result[row, 59+i*312:59+(i+1)*312], body_frame(motion, min(start+offset, end)))


def test_force2_configuration_uses_body_lookahead_for_both_networks():
    # Exercise the configuration function without launching Isaac Sim.
    import ast
    from copy import deepcopy
    tree = ast.parse((SOURCE/'tasks/tracking/config/t1_wbc_force2/flat_env_cfg.py').read_text())
    configure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'configure_wbc_force')
    namespace = dict(reference_window=observations.reference_window,
                     WBCForceActionsCfg=lambda: NS(wbc_force=NS()),
                     WBCForceRewardsCfg=lambda: NS(), ObservationTermCfg=NS, TerminationTermCfg=NS,
                     actual_root_joint_state=object(), qp_failed=object())
    exec(compile(ast.Module(body=[configure], type_ignores=[]), '<force2_config>', 'exec'), namespace)
    group = NS(history_length=1, reference=NS(func=None, params={'root_only': True}))
    cfg = NS(actions=NS(wbc_acc=NS(record_diagnostics=False, standing_only=False)),
             observations=NS(policy=deepcopy(group), critic=deepcopy(group)), terminations=NS())
    namespace['configure_wbc_force'](cfg)
    for group in (cfg.observations.policy, cfg.observations.critic):
        assert group.reference.func is observations.reference_window
        assert group.reference.params == {'offsets': (0, 1, 3, 5, 10)}
        assert group.reference.history_length == 0
        assert group.history_length is None
        assert group.actual_state.history_length == 5
        assert group.actual_state.func is namespace['actual_root_joint_state']
