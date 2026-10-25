"""The editable decoded weights must replay as the intended force-limit inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from turn_reanchor_manual_outputs import DEFAULT, load_manual_outputs, save_schedule
from manual_wbc_schedule import load_schedule


def test_edited_decoded_zero_and_half_survive_export_and_standard_replay(tmp_path):
    spec=json.loads(DEFAULT.read_text())
    spec['decoded_weights'][0]=[0.,.5]
    path=tmp_path/'weights.json';path.write_text(json.dumps(spec))
    data=load_manual_outputs(path)
    np.testing.assert_array_equal(data['actions'][0],[-3.,0.])
    np.testing.assert_array_equal(data['fractions'][0],[0.,.5])
    save_schedule(tmp_path/'labels.npz',data)
    loaded=load_schedule(tmp_path/'labels.npz','force',ROOT/spec['reference'],.02,661,
        controller_config=spec['controller_config'])
    np.testing.assert_array_equal(loaded['actions'],data['actions'])
    assert not json.loads(str(loaded['metadata_json']))['manual_actual_contact_gate']


@pytest.mark.parametrize('bad',[-.01,1.01,float('nan')])
def test_invalid_decoded_weights_rejected(tmp_path,bad):
    spec=json.loads(DEFAULT.read_text());spec['decoded_weights'][20][0]=bad
    path=tmp_path/'bad.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='finite decoded weights'):
        load_manual_outputs(path)


def test_different_reference_rejected_before_replay(tmp_path):
    spec=json.loads(DEFAULT.read_text());spec['reference_sha256']='wrong'
    path=tmp_path/'bad.json';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='Reference differs'):
        load_manual_outputs(path)
