"""Offline smooth manual force weights with reachable, sustained exact zeros.

This edits a saved policy-input table; it never gates outputs during playback.
"""
import numpy as np


def smooth_contact_weights(fractions, off, ramp_frames=4):
    """Taper around explicit OFF intervals using a quintic smoothstep.

    OFF rows stay exactly zero. Adjacent ON rows ramp over ``ramp_frames``
    samples; overlapping release/restore ramps use the smaller envelope.
    Existing zero weights are also treated as OFF, so smoothing never creates
    support that the original manual table excluded.
    """
    values = np.asarray(fractions, dtype=np.float64)
    off = np.asarray(off, dtype=bool)
    if values.ndim != 2 or off.shape != values.shape or not len(values):
        raise ValueError('Weights and OFF flags must be matching frame/channel matrices')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Decoded weights must be finite values in [0, 1]')
    if type(ramp_frames) is not int or ramp_frames < 1:
        raise ValueError('ramp_frames must be a positive integer')
    excluded = off | (values == 0)
    envelope = np.ones_like(values)
    envelope[excluded] = 0.
    for distance in range(1, ramp_frames):
        u = distance / ramp_frames
        weight = u**3 * (10. - 15.*u + 6.*u*u)
        envelope[distance:] = np.minimum(
            envelope[distance:], np.where(excluded[:-distance], weight, 1.))
        envelope[:-distance] = np.minimum(
            envelope[:-distance], np.where(excluded[distance:], weight, 1.))
    return values * envelope
