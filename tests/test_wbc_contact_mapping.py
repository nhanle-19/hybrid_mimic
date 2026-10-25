"""Offline/manual labels must produce the same endpoints as the runtime decoder."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_contact_mapping import (decode_contact_values, encode_contact_values,
                                 decode_contact_values_numpy, encode_contact_values_numpy)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_manual_label_roundtrip_matches_qp_precision(dtype):
    fractions = torch.tensor([0., 1e-4, .2, .5, .6, 1.-1e-4, 1.], dtype=dtype)
    actions = encode_contact_values(fractions)
    assert actions[0] == -3. and actions[-1] == 3.
    # The action handler stores float32; the QP evaluates float64.
    actual = decode_contact_values(actions.float().double())
    np.testing.assert_allclose(actual.numpy(), fractions.double().numpy(), atol=1e-7, rtol=0.)
    assert actual[0] == 0. and actual[-1] == 1.
    offline = encode_contact_values_numpy(fractions.numpy()).astype(np.float32)
    np.testing.assert_allclose(decode_contact_values_numpy(offline), actual.numpy(), atol=1e-7, rtol=0.)


def test_offline_decoder_matches_runtime_including_boundary_and_extreme_actions():
    boundary = np.log(.1/.9)
    actions = np.array([-1e30, -3., boundary-1e-4, boundary+1e-4, 0.,
                        -boundary-1e-4, -boundary+1e-4, 3., 1e30])
    actual = decode_contact_values(torch.tensor(actions)).numpy()
    np.testing.assert_allclose(decode_contact_values_numpy(actions), actual, atol=2e-16, rtol=0.)
    assert np.array_equal(actual[[0, 1, 2, 4, 6, 7, 8]], [0., 0., 0., .5, 1., 1., 1.])
    assert 0. < actual[3] < .001 and .999 < actual[5] < 1.
