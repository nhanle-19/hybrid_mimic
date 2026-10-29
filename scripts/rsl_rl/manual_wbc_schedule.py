"""Reference-derived manual policy schedules; no simulator imports.

Actions are raw policy logits, indexed by reference frame (one frame per control
step). This is a hand-designed baseline, not a claim of dynamic feasibility.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Import the simulator-independent mapping without registering Isaac tasks.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'source/whole_body_tracking/whole_body_tracking/utils'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_contact_envelope import smooth_contact_weights
from wbc_contact_mapping import (decode_contact_values_numpy, encode_contact_values_numpy,
                                 contact_action_transform)

LAYOUTS = {
    'force': ('left_fz_limit', 'right_fz_limit'),
    'authority': ('left_x', 'left_y', 'left_z', 'right_x', 'right_y', 'right_z'),
}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_digest(joints, positions, quaternions):
    digest = hashlib.sha256()
    for value in (joints, positions, quaternions):
        array = np.ascontiguousarray(value, dtype=np.float32)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def smooth(values, width=5):
    width = min(width, len(values))
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    return np.apply_along_axis(lambda v: np.convolve(np.pad(v, (pad, pad), mode='edge'),
                                                    np.ones(width)/width, mode='valid'), 0, values)


def build_schedule(controller, fps, positions, rotations, velocities, com, mass,
                   sole_points, controller_config, motion_path, prepared_sha256):
    """Use sole clearance, reference speed and CoM projection to assign actions.

    Force limits allow headroom above estimated support loads. They are upper
    bounds, never force targets. FORCE weights smoothly release before reference
    lift-off, stay exactly zero in swing, and restore after reference touchdown.
    Actual geometry still decides whether a physical support point is available.
    """
    positions, rotations, velocities, com = map(np.asarray, (positions, rotations, velocities, com))
    n = len(positions)
    if controller not in LAYOUTS or n < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Schedule requires force/authority, at least two frames, and positive fps')
    points = np.einsum('nfij,fkj->nfki', rotations, sole_points) + positions[:, :, None, :]
    heights = points[..., 2].min(-1) - controller_config['ground_height']
    reference_contact = ((heights <= controller_config.get('contact_gap_tolerance', .001))
                         & (heights >= -controller_config.get('max_contact_penetration', .03))
                         & (rotations[:, :, 2, 2] > .5))
    horizontal_speed = np.linalg.norm(velocities[..., :2], axis=-1)
    # Full support confidence below 3 mm, fading to zero by 43 mm clearance.
    height_blend = np.clip((heights-.003)/.04, 0., 1.)
    contact = 1.-height_blend**2*(3.-2.*height_blend)
    segment = positions[:, 1, :2]-positions[:, 0, :2]
    right_share = np.clip(np.sum((com[:, :2]-positions[:, 0, :2])*segment, axis=-1)
                          / np.maximum(np.sum(segment**2, axis=-1), 1e-8), .1, .9)
    shares = np.stack((1.-right_share, right_share), -1)*contact
    shares /= np.maximum(shares.sum(-1, keepdims=True), 1e-8)
    # Estimate total vertical demand, suppressing frame-to-frame acceleration noise.
    com_acc_z = smooth(np.gradient(np.gradient(com[:, 2], 1./fps), 1./fps), 9)
    vertical_load = mass*np.clip(9.81+com_acc_z, .5*9.81, 1.5*9.81)
    nominal_load = shares*vertical_load[:, None]
    if controller == 'force':
        # Anticipate landing by at most 80 ms, while tapering release over 40 ms.
        anticipation = max(1, round(.08*fps))
        caps = 15. + 1.7*nominal_load + 50.*contact
        for foot in range(2):
            caps[:, foot] = np.array([max(caps[i:min(n, i+anticipation+1), foot]) for i in range(n)])
        caps = smooth(caps)
        fractions = np.clip(caps/np.asarray(controller_config['contact_force_max']), .025, .98)
        fractions = smooth_contact_weights(fractions, ~reference_contact,
                                          max(2, round(.08*fps)))
    else:
        # Penalize stationary support translation most strongly. Release xy
        # when the reference foot is moving; z remains stronger during support.
        planar = .08 + .82*contact*np.exp(-(horizontal_speed/.25)**2)
        vertical = .05 + .90*contact*np.exp(-(np.abs(velocities[..., 2])/.35)**2)
        # Vary the cost even when both feet stay planted and only the CoM moves.
        # AUTH costs are quadratic in authority: sqrt(.5 + share) scales each
        # foot's costs with its estimated load, preserving the equal-load case.
        load_scale = np.sqrt(.5 + shares)
        fractions = smooth((np.stack((planar, planar, vertical), -1)
                            * load_scale[..., None]).reshape(n, 6))
        fractions = np.clip(fractions, .025, .98)
    actions = encode_contact_values_numpy(fractions).astype(np.float32)
    phases = np.where(contact[:, 0] > .5,
                      np.where(contact[:, 1] > .5, 'double_support', 'left_support'),
                      np.where(contact[:, 1] > .5, 'right_support', 'flight_or_uncertain'))
    metadata = {
        'schema_version': 2, 'action_transform': contact_action_transform(),
        'controller': controller, 'action_layout': list(LAYOUTS[controller]),
        'motion_file': str(Path(motion_path).resolve()), 'motion_sha256': file_digest(motion_path),
        'prepared_reference_sha256': prepared_sha256, 'controller_config': controller_config,
        'design': ('reference_geometry_and_com_exact_off_v2' if controller == 'force'
                   else 'reference_geometry_and_load_share_v2'),
        'status': 'hand_designed_unvalidated',
        'phase_definition': 'Soft near-ground support confidence, not a physical contact assertion; see reference_contact_available.',
        'notes': ('Raw logits; one row per reference frame; FORCE caps control support without a runtime geometry gate. No feedback in schedule.'
                  if controller == 'force' else
                  'Raw logits; one row per reference frame; actual geometry gates support. No feedback in schedule.'),
    }
    if controller == 'authority':
        metadata['authority_load_scaling'] = 'Multiply speed/contact authority by sqrt(0.5 + estimated support load share), then smooth and clip.'
    else:
        metadata['force_release'] = 'Offline quintic ramps over 80 ms around reference-contact intervals; exact zero throughout reference swing. Playback does not replace these outputs.'
    return dict(actions=actions, fractions=fractions, time_s=np.arange(n)/fps,
                reference_frame=np.arange(n, dtype=np.int64), fps=np.asarray(fps),
                foot_clearance_m=heights, foot_horizontal_speed_m_s=horizontal_speed,
                contact_confidence=contact, nominal_vertical_load_n=nominal_load,
                reference_contact_available=reference_contact,
                phase=phases, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))


def save_schedule(path, schedule):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **schedule)
    import csv
    metadata = json.loads(str(schedule['metadata_json']))
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'time_s', 'phase', *['raw_'+x for x in metadata['action_layout']],
                         *['fraction_'+x for x in metadata['action_layout']]])
        for i, action in enumerate(schedule['actions']):
            writer.writerow([i, schedule['time_s'][i], schedule['phase'][i], *action, *schedule['fractions'][i]])
    return path


def load_schedule(path, controller, motion_path, control_dt=None, frame_count=None,
                  prepared_sha256=None, controller_config=None):
    with np.load(path, allow_pickle=False) as archive:
        schedule = dict(archive)
    return validate_schedule(schedule, controller, motion_path, control_dt, frame_count,
                             prepared_sha256, controller_config)


def validate_schedule(schedule, controller, motion_path, control_dt=None, frame_count=None,
                      prepared_sha256=None, controller_config=None):
    metadata = json.loads(str(schedule['metadata_json']))
    if metadata.get('controller') != controller:
        raise ValueError('Schedule schema/controller mismatch')
    if (metadata.get('schema_version') != 2
            or metadata.get('action_transform') != contact_action_transform()):
        raise ValueError('Schedule uses a different action mapping; migrate the legacy sigmoid schedule '
                         'with manual_wbc_schedule.py --migrate INPUT OUTPUT or rebuild it')
    if metadata.get('action_layout') != list(LAYOUTS[controller]):
        raise ValueError('Schedule action order mismatch')
    if metadata.get('motion_sha256') != file_digest(motion_path):
        raise ValueError('Schedule belongs to a different motion file')
    actions = schedule['actions']
    if actions.ndim != 2:
        raise ValueError('Schedule actions must be a frame-by-output matrix')
    n = len(actions)
    if actions.shape != (n, len(LAYOUTS[controller])) or n < 2 or not np.isfinite(actions).all():
        raise ValueError('Schedule must contain finite raw action vectors for every frame')
    if np.max(np.abs(actions)) > np.finfo(np.float32).max:
        raise ValueError('Schedule actions must fit float32')
    fractions = schedule['fractions']
    decoded = decode_contact_values_numpy(actions)
    if fractions.shape != actions.shape or not np.allclose(
            fractions, decoded, atol=1e-6, rtol=0):
        raise ValueError('Schedule fractions must match the controller action mapping')
    fps = float(schedule['fps'].item())
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid schedule fps')
    if not np.array_equal(schedule['reference_frame'], np.arange(n)):
        raise ValueError('Schedule reference frames must cover the clip exactly once in order')
    if not np.allclose(schedule['time_s'], np.arange(n)/fps, atol=1e-8, rtol=0):
        raise ValueError('Schedule timestamps do not match reference fps')
    if control_dt is not None and not np.isclose(control_dt*fps, 1., atol=1e-8, rtol=0):
        raise ValueError('Schedule requires one reference frame per control step; fps/control dt mismatch')
    if frame_count is not None and n != frame_count:
        raise ValueError('Schedule length does not match the complete reference')
    if prepared_sha256 is not None and metadata.get('prepared_reference_sha256') != prepared_sha256:
        raise ValueError('Prepared reference differs from the schedule; rebuild it for this reference/model')
    if controller_config is not None and metadata.get('controller_config') != json.loads(json.dumps(controller_config)):
        raise ValueError('Controller configuration differs from the schedule; rebuild the schedule')
    return schedule


def migrate_schedule(source, destination):
    """Preserve legacy fractions in a new file; require a fresh physics replay."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix('.csv').exists():
        raise ValueError('Migration destination already exists; choose a new filename')
    with np.load(source, allow_pickle=False) as archive:
        schedule = dict(archive)
    metadata = json.loads(str(schedule['metadata_json']))
    controller = metadata.get('controller')
    if (controller not in LAYOUTS or metadata.get('schema_version') != 1
            or metadata.get('action_layout') != list(LAYOUTS[controller])
            or metadata.get('action_transform', {'type': 'sigmoid'}) != {'type': 'sigmoid'}):
        raise ValueError('Migration requires a legacy sigmoid FORCE or AUTH schedule')
    actions, fractions = schedule['actions'], schedule['fractions']
    if actions.ndim != 2 or actions.shape[1] != len(LAYOUTS[controller]) or fractions.shape != actions.shape:
        raise ValueError('Legacy schedule action/fraction dimensions do not match the controller')
    if not np.isfinite(actions).all() or not np.allclose(
            fractions, np.exp(-np.logaddexp(0., -actions.astype(np.float64))), atol=1e-6, rtol=0):
        raise ValueError('Legacy schedule fractions must match sigmoid(raw actions)')
    schedule['actions'] = encode_contact_values_numpy(fractions).astype(np.float32)
    previous_status = metadata.get('status')
    previous_validation = metadata.pop('validation', None)
    metadata.update(schema_version=2, action_transform=contact_action_transform(),
                    status='mapping_migrated_requires_replay',
                    migration={'source_schedule': str(source.resolve()), 'source_sha256': file_digest(source),
                               'preserved': 'mapped fractions; raw outputs re-encoded',
                               'previous_status': previous_status, 'previous_validation': previous_validation})
    schedule['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    validate_schedule(schedule, controller, metadata['motion_file'])
    return save_schedule(destination, schedule)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate', nargs=2, required=True, metavar=('INPUT', 'OUTPUT'),
                        help='Re-encode a legacy schedule while preserving its mapped fractions')
    args = parser.parse_args()
    try:
        print(migrate_schedule(*args.migrate))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
