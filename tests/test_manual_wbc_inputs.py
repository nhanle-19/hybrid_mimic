"""Manual policy inputs must never silently fall back to controller defaults."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rsl_rl'))
from manual_wbc_inputs import load_manual_inputs
import eval_toy_wbc


def test_checked_in_pose_hold_values_preserve_intended_caps_and_costs():
    from manual_wbc_schedule import decode_contact_values_numpy
    import numpy as np
    path = ROOT/'configs/toy_wbc_inputs.json'
    np.testing.assert_allclose(decode_contact_values_numpy(load_manual_inputs(path, 'force')), [.2, .6])
    np.testing.assert_allclose(decode_contact_values_numpy(load_manual_inputs(path, 'authority')), [.5]*6)


def test_unset_authority_inputs_rejected(tmp_path):
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps({'authority': [None]*6}))
    with pytest.raises(ValueError, match='not set yet'):
        load_manual_inputs(path, 'authority')


def test_legacy_authority_inputs_rejected(tmp_path):
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps({'authority': [0.]*12}))
    with pytest.raises(ValueError, match='6 raw NN outputs'):
        load_manual_inputs(path, 'authority')


@pytest.mark.parametrize('values', [[1], [None, 1], [True, 1], ['1', 1], [float('nan'), 1], [1e100, 1]])
def test_invalid_force_inputs(tmp_path, values):
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps({'force': values}))
    with pytest.raises(ValueError):
        load_manual_inputs(path, 'force')


@pytest.mark.parametrize('controller,count', [('force', 2), ('authority', 6)])
def test_launcher_routes_raw_outputs_to_matching_task(tmp_path, monkeypatch, controller, count):
    path = tmp_path / 'inputs.json'
    values = [-2.5, 3.] * (count // 2)
    path.write_text(json.dumps({controller: values}))
    assert load_manual_inputs(path, controller) == values  # No fraction encoding/clipping.
    reference = tmp_path / 'reference.npz'
    reference.touch()
    monkeypatch.setattr(sys, 'argv', ['eval_toy_wbc.py', '--controller', controller,
                                    '--inputs', str(path), '--motion_file', str(reference),
                                    '--video', '--headless'])
    commands = []
    monkeypatch.setattr(eval_toy_wbc.os, 'execv', lambda executable, command: commands.append(command))
    eval_toy_wbc.main()
    command, = commands
    assert command[command.index('--task')+1] == f'Toy-Standing-WBC-{controller.upper()}-T1-v0'
    assert command[command.index('--manual_inputs')+1] == str(path)
    assert '--qp_only' in command and '--fail_on_reset' in command
    assert '--video' in command and '--headless' in command
    assert '--load_run' not in command and '--nominal_standing' not in command


@pytest.mark.parametrize('controller', ['force', 'authority'])
def test_toy_tracking_uses_whole_clip_and_separate_outputs(tmp_path, monkeypatch, controller):
    import numpy as np
    reference = tmp_path / 'reference.npz'
    np.savez(reference, fps=50., joint_pos=np.zeros((300, 23)))
    monkeypatch.setattr(sys, 'argv', ['eval_toy_wbc.py', '--controller', controller,
                                    '--motion_file', str(reference), '--track_motion'])
    commands = []
    monkeypatch.setattr(eval_toy_wbc.os, 'execv', lambda executable, command: commands.append(command))
    eval_toy_wbc.main()
    command, = commands
    assert '--track_motion' in command
    assert command[command.index('--steps')+1] == '300'
    assert '/toy_standing_sway/' in command[command.index('--output')+1]
    assert command[command.index('--manual_inputs')+1] == str(ROOT/'configs/toy_sway_wbc_inputs.json')
    if controller == 'force':
        index = command.index('--force_max')
        assert command[index+1:index+3] == ['1000.0', '1000.0']
    else:
        assert '--force_max' not in command
    assert '--fail_on_reset' in command


@pytest.mark.parametrize('fps,steps', [(25., 50), (50., 101)])
def test_toy_tracking_rejects_wrong_timing(tmp_path, monkeypatch, fps, steps):
    import numpy as np
    reference = tmp_path / 'reference.npz'
    np.savez(reference, fps=fps, joint_pos=np.zeros((100, 23)))
    monkeypatch.setattr(sys, 'argv', ['eval_toy_wbc.py', '--controller', 'force',
        '--motion_file', str(reference), '--track_motion', '--steps', str(steps)])
    with pytest.raises(SystemExit, match='2'):
        eval_toy_wbc.main()


@pytest.mark.parametrize('option', ['--manual_schedule', '--build_manual_schedule'])
@pytest.mark.parametrize('controller', ['force', 'authority'])
def test_toy_schedule_routes_without_constant_inputs(tmp_path, monkeypatch, option, controller):
    import numpy as np
    from manual_wbc_schedule import build_schedule, save_schedule
    reference = tmp_path / 'reference.npz'
    n = 20
    np.savez(reference, fps=50., joint_pos=np.zeros((n, 23)))
    schedule = tmp_path / 'schedule.npz'
    if option == '--manual_schedule':
        positions = np.zeros((n, 2, 3))
        positions[:, :, 1] = [.1, -.1]
        values = build_schedule(controller, 50., positions, np.tile(np.eye(3), (n, 2, 1, 1)),
            np.zeros((n, 2, 6)), np.zeros((n, 3)), 25., np.zeros((2, 4, 3)),
            dict(ground_height=0., contact_force_max=[1000., 1000.]), reference, 'prepared')
        save_schedule(schedule, values)
    monkeypatch.setattr(sys, 'argv', ['eval_toy_wbc.py', '--controller', controller,
        '--track_motion', '--motion_file', str(reference), option, str(schedule)])
    commands = []
    monkeypatch.setattr(eval_toy_wbc.os, 'execv', lambda executable, command: commands.append(command))
    eval_toy_wbc.main()
    command, = commands
    assert '--manual_inputs' not in command
    assert command[command.index(option)+1] == str(schedule)
    assert command[command.index('--steps')+1] == str(n)
    assert command[command.index('--task')+1] == f'Toy-Standing-WBC-{controller.upper()}-T1-v0'
    if controller == 'force':
        assert command[command.index('--force_max')+1:command.index('--force_max')+3] == ['1000.0', '1000.0']
        assert '/force_scheduled/' in command[command.index('--output')+1]
    else:
        assert '--force_max' not in command
        assert '/authority_cost_only_scheduled/' in command[command.index('--output')+1]


def test_toy_schedule_requires_tracking_and_excludes_constant_inputs(monkeypatch):
    for extra in [[], ['--track_motion', '--inputs', 'constant.json']]:
        monkeypatch.setattr(sys, 'argv', ['eval_toy_wbc.py', '--controller', 'force',
                                        '--manual_schedule', 'schedule.npz', *extra])
        with pytest.raises(SystemExit, match='2'):
            eval_toy_wbc.main()
