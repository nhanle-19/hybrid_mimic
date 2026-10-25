"""A manual release must have a smooth approach and an exact-zero plateau."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_contact_envelope import smooth_contact_weights


def test_smooth_release_and_restore_preserve_zero_and_other_channels():
    original = np.full((30, 2), .8)
    off = np.zeros_like(original, dtype=bool)
    off[10:16, 0] = True
    result = smooth_contact_weights(original, off, 4)
    np.testing.assert_array_equal(result[10:16, 0], 0.)
    np.testing.assert_array_equal(result[:, 1], original[:, 1])
    assert (np.diff(result[6:11, 0]) < 0).all()
    assert (np.diff(result[15:20, 0]) > 0).all()
    assert result[9, 0] < .1 and result[16, 0] < .1
    np.testing.assert_allclose(result[6:11, 0], result[15:20, 0][::-1])
    np.testing.assert_array_equal(result[19:, 0], original[19:, 0])


def test_existing_off_intervals_and_clip_edges_are_not_filled():
    original = np.ones((20, 2))
    original[:4, 0] = 0.
    original[9:12, 0] = 0.
    original[-3:, 1] = 0.
    result = smooth_contact_weights(original, np.zeros_like(original, dtype=bool), 4)
    assert not result[original == 0].any()
    assert (result <= original).all()
    assert np.isfinite(result).all()
    assert 0. < result[4, 0] < result[5, 0] < result[6, 0]


def test_reject_invalid_weights():
    with pytest.raises(ValueError, match='finite'):
        smooth_contact_weights(np.array([[1.1]]), np.array([[False]]))
