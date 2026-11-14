"""FORCE2 sole-point lookahead uses reference geometry without terrain data."""
import importlib.util
from pathlib import Path
import sys
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
BODY_NAMES = [f'b{i}' for i in range(24)]
BODY_NAMES[5], BODY_NAMES[18] = 'right_foot_link', 'left_foot_link'


@pytest.fixture(autouse=True)
def geometry_modules(monkeypatch):
    # Load the analytical helpers without importing Isaac Sim's task registry.
    for name in ('wbc_acc_contact_policy', 'wbc_acc_torch_model'):
        full_name = 'whole_body_tracking.utils.'+name
        monkeypatch.setitem(sys.modules, full_name, load(full_name, SOURCE/'utils'/f'{name}.py'))


def make_motion():
    motion = NS(time_step_total=12,
                joint_pos=torch.arange(12*23).reshape(12, 23).float(),
                joint_vel=torch.arange(12*23).reshape(12, 23).float()+1000)
    coordinates = torch.arange(12*24*3).reshape(12, 24, 3).float()
    motion._body_pos_w = coordinates/1000
    motion._body_lin_vel_w = coordinates/200
    motion._body_ang_vel_w = coordinates/500-.4
    angle = torch.arange(12)[:, None]*.07+torch.arange(24)[None]*.03
    motion._body_quat_w = torch.zeros(12, 24, 4)
    motion._body_quat_w[..., 0] = torch.cos(angle/2)
    motion._body_quat_w[..., 2] = torch.sin(angle/2)
    return motion


def make_env(motion, frames):
    command = NS(motion=motion, time_steps=torch.tensor(frames), robot_anchor_body_index=7,
                 robot=NS(body_names=BODY_NAMES))
    # Dynamics and motion use different body orderings, as in the real task.
    model = NS(body_names=BODY_NAMES[::-1], mass=(torch.arange(24).double()+1)/10,
               com_local=(torch.arange(72).reshape(24, 3).double()%5-2)*.01)
    return NS(device='cpu', command_manager=NS(get_term=lambda _: command),
              action_manager=NS(get_term=lambda _: NS(dynamics=model)))


def contact_frame(env, frame):
    """Independent NumPy construction for the pitch-only test references."""
    motion = env.command_manager.get_term('motion').motion
    model = env.action_manager.get_term('wbc_force').dynamics
    position = motion._body_pos_w[frame].numpy().astype(np.float64)
    quat = motion._body_quat_w[frame].numpy().astype(np.float64)
    rotations, body_com, masses = [], [], []
    for index, name in enumerate(BODY_NAMES):
        angle = 2*np.arctan2(quat[index, 2], quat[index, 0])
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        rotations.append(rotation)
        model_index = model.body_names.index(name)
        masses.append(model.mass[model_index].item())
        body_com.append(position[index]+rotation@model.com_local[model_index].numpy())
    com = np.average(body_com, axis=0, weights=masses)
    corners = np.array([[-.1015, -.05, -.03], [-.1015, .05, -.03],
                        [.1215, -.05, -.03], [.1215, .05, -.03]])
    result = []
    for name in ('left_foot_link', 'right_foot_link'):
        index = BODY_NAMES.index(name)
        for corner in corners:
            offset = rotations[index]@corner
            velocity = (motion._body_lin_vel_w[frame, index].numpy()
                        +np.cross(motion._body_ang_vel_w[frame, index].numpy(), offset))
            result.extend(np.concatenate((position[index]+offset-com, velocity)))
    return torch.tensor(result, dtype=motion._body_pos_w.dtype)


@pytest.mark.parametrize('start', [0, 10, 11])
def test_current_reference_preserved_and_future_contains_com_relative_corners(start):
    motion = make_motion()
    env = make_env(motion, [start])
    result = observations.reference_window(env)
    assert result.shape == (1, 59+4*48)
    current = torch.cat((motion.joint_pos[start], motion.joint_vel[start],
                         *(getattr(motion, name)[start, 7] for name, _ in BODY_FIELDS)))
    torch.testing.assert_close(result[0, :59], current)
    for i, offset in enumerate((1, 3, 5, 10)):
        torch.testing.assert_close(result[0, 59+i*48:59+(i+1)*48], contact_frame(env, min(start+offset, 11)))


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
    env = make_env(motion, [0, 11])
    result = observations.reference_window(env, offsets=(1, 3))
    assert result.shape == (2, 96)
    torch.testing.assert_close(result[0], torch.cat((contact_frame(env, 1), contact_frame(env, 3))))
    torch.testing.assert_close(result[1], contact_frame(env, 11).repeat(2))


def test_reference_translation_does_not_supply_floor_height():
    motion = make_motion()
    env = make_env(motion, [0, 7])
    before = observations.reference_window(env, offsets=(1, 3))
    # Translate every frame independently, including vertical translation.
    motion._body_pos_w += torch.arange(12)[:, None, None]*torch.tensor([.3, -.2, .7])
    env.scene = NS(terrain=NS(ground_height=999.), env_origins=torch.full((2, 3), float('nan')))
    env.command_manager.get_term('motion').robot.data = NS(body_pos_w=torch.full((2, 24, 3), float('nan')))
    torch.testing.assert_close(observations.reference_window(env, offsets=(1, 3)), before)


def test_corner_velocity_includes_rotation_and_keeps_world_translation():
    motion = make_motion()
    motion._body_lin_vel_w.zero_()
    motion._body_ang_vel_w.zero_()
    motion._body_ang_vel_w[:, 18] = torch.tensor([1., 2., 3.])
    env = make_env(motion, [0])
    before = observations.reference_window(env, offsets=(1,)).reshape(2, 4, 6)
    torch.testing.assert_close(before.flatten(), contact_frame(env, 1))
    assert before[0, :, 3:].abs().sum() > 0
    assert before[1, :, 3:].count_nonzero() == 0
    boost = torch.tensor([.5, -.3, .2])
    motion._body_lin_vel_w += boost
    after = observations.reference_window(env, offsets=(1,)).reshape(2, 4, 6)
    torch.testing.assert_close(after[..., :3], before[..., :3])
    torch.testing.assert_close(after[..., 3:], before[..., 3:]+boost)


def test_current_only_reference_does_not_require_contact_model():
    env = make_env(make_motion(), [0])
    del env.action_manager
    assert observations.reference_window(env, offsets=(0,)).shape == (1, 59)


def test_lookahead_clamps_within_each_motion_clip(tmp_path):
    source = make_motion()
    for name, frames in [('a', slice(0, 3)), ('b', slice(3, 12))]:
        values = {key: getattr(source, key)[frames].numpy() for key in ('joint_pos', 'joint_vel')}
        values.update({key[1:]: getattr(source, key)[frames].numpy() for key, _ in BODY_FIELDS})
        np.savez(tmp_path/f'{name}.npz', fps=50., **values,
                 joint_names=np.asarray([f'j{i}' for i in range(23)]),
                 body_names=np.asarray(BODY_NAMES))
    motion = MotionLibrary(tmp_path)
    env = make_env(motion, [1, 2, 3, 11])
    result = observations.reference_window(env)
    for row, start in enumerate((1, 2, 3, 11)):
        end = 2 if start < 3 else 11
        for i, offset in enumerate((1, 3, 5, 10)):
            torch.testing.assert_close(result[row, 59+i*48:59+(i+1)*48], contact_frame(env, min(start+offset, end)))


def test_force2_configuration_uses_contact_lookahead_for_both_networks():
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
