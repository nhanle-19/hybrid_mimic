"""Manual policy inputs must never silently fall back to controller defaults."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rsl_rl'))
from manual_wbc_inputs import load_manual_inputs
import eval_toy_wbc


def test_authority_inputs_remain_deliberately_unset():
    with pytest.raises(ValueError, match='not set yet'):
        load_manual_inputs(ROOT / 'configs/toy_wbc_inputs.json', 'authority')


@pytest.mark.parametrize('values', [[1], [None, 1], [True, 1], ['1', 1], [float('nan'), 1], [1e100, 1]])
def test_invalid_force_inputs(tmp_path, values):
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps({'force': values}))
    with pytest.raises(ValueError):
        load_manual_inputs(path, 'force')


@pytest.mark.parametrize('controller,count', [('force', 2), ('authority', 12)])
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
