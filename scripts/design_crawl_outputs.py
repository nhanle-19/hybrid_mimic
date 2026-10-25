"""Hand-designed eight-channel force-cap candidates informed by measured contact.

These replace raw NN outputs. They do not mask the QP with reference labels or
command a measured force. Designs are saved before independent physics replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from crawl_manual_nn_outputs import load_manual_nn_outputs, save_manual_nn_outputs
from wbc_contact_mapping import encode_contact_values_numpy


def design(manual_path, dynamic_path, actual_path, output):
    baseline = load_manual_nn_outputs(manual_path)
    if json.loads(str(baseline['metadata_json'])).get('sample_rate_multiple', 1) != 1:
        raise ValueError('This reference-frame design tool requires a reference-rate input table')
    dynamic = dict(np.load(dynamic_path, allow_pickle=False))
    actual = dict(np.load(actual_path, allow_pickle=False))
    n = len(baseline['actions'])
    names = baseline['region_names'].tolist()
    allowed = baseline['fractions'] > 0
    normal = dynamic['forces'][..., 2].sum(-1)
    valid = dynamic['dissipative']
    for i in range(8):
        normal[:, i] = np.interp(np.arange(n), np.flatnonzero(valid), normal[valid, i])
    normal = gaussian_filter1d(normal, 1.5, axis=0)
    velocity = np.divide((dynamic['point_velocity_w'][..., 0]*dynamic['forces'][..., 2]).sum(-1),
        dynamic['forces'][..., 2].sum(-1), out=np.zeros((n, 8)), where=dynamic['forces'][..., 2].sum(-1)>1e-5)
    measured = actual['measured_forces'][:, [actual['measured_region_names'].tolist().index(name) for name in names], 2]
    total = actual['point_forces'][..., 2].sum(-1)
    actual_velocity = np.divide((actual['point_velocity_w'][..., 0]*actual['point_forces'][..., 2]).sum(-1), total,
        out=np.zeros_like(total), where=total>1e-5)
    for frame in np.unique(actual['reference_frame']):
        rows = actual['reference_frame'] == frame
        # Add a candidate contact only where it actually carried a measured load.
        observed = measured[rows].max(0) > 5.
        allowed[frame] |= observed
        velocity[frame, observed] = actual_velocity[rows].mean(0)[observed]
        # Geometry may describe a different contact phase from the drifted robot.
        newly_observed = observed & (baseline['fractions'][frame] == 0)
        normal[frame, newly_observed] = measured[rows].mean(0)[newly_observed]
    velocity = gaussian_filter1d(velocity, 1.5, axis=0)
    candidates = {
        'load_caps': (1.2*normal+15., '1.2 times feasible reference normal load plus 15 N headroom.'),
        'load_caps_wide': (1.6*normal+25., '1.6 times feasible reference normal load plus 25 N headroom.'),
    }
    # A forward-sliding point tends to drag backward. Relax its force ceiling
    # while allowing more normal load on the slow/backward stroke that can push.
    forward = np.clip((velocity-.01)/.06, 0., 1.)
    release = (1.6*normal+25.)*(1.-.65*forward)
    for frame in range(n):
        if (release[frame]*allowed[frame]).sum() < 390.:
            weights = allowed[frame]*(1.05-forward[frame])
            release[frame] += weights/max(weights.sum(), 1e-9)*(390.-(release[frame]*allowed[frame]).sum())
    candidates['push_release'] = (release, 'Reduce caps up to 65 percent on observed forward sliding; transfer headroom to slow/backward contacts, with 390 N total nominal capacity.')
    output.mkdir(parents=True, exist_ok=True)
    for name, (caps, description) in candidates.items():
        caps = gaussian_filter1d(caps, 1.5, axis=0)
        # Retain the independently stable initial stance; blend into the manual
        # force sharing over reference frames 35..45 without changing timing.
        blend = np.clip((np.arange(n)-35)/10., 0., 1.)[:, None]
        caps = ((1-blend)*600.+blend*np.clip(caps, 0, 600))*allowed
        data = dict(baseline)
        data['fractions'] = caps/600.
        data['actions'] = encode_contact_values_numpy(data['fractions'])
        data['normal_force_cap_n'] = caps
        metadata = json.loads(str(data['metadata_json']))
        metadata.update(design='manual_'+name, design_description=description,
            status='manual_candidate_requires_physics_validation',
            basis_actual_diagnostics=str(actual_path.resolve()), basis_dynamic_check=str(dynamic_path.resolve()),
            contact_source='Reference geometry plus observed loaded contacts in the preceding baseline; outputs remain independent of QP reference annotations.')
        data['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
        save_manual_nn_outputs(output/(name+'.npz'), data)
        print(name, 'cap ranges N:', np.round(np.c_[caps.min(0), caps.max(0)], 1).tolist(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual', type=Path, required=True)
    parser.add_argument('--dynamic', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design(args.manual, args.dynamic, args.actual, args.output)
