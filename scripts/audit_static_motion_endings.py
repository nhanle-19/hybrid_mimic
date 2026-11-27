"""Check the saved static-ending artifacts, including every mirrored trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np

from dsms_t1_model import ROOT, build_model, load_reference, sha
from prepare_static_motion_endings import EndingBuilder


def main(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    editor = EndingBuilder(build_model(args.output.parent/'audit_model'))
    manifest = json.loads((args.directory/'manifest.json').read_text())
    by_name = {r['file']: r for r in manifest['motions']}
    records = []
    for row in manifest['motions']:
        path = args.directory/row['file']
        assert sha(path) == row['sha256']
        z, states, fps = load_reference(path, editor.model)
        q = states[:, :editor.model.nq]
        source = by_name[row['mirrored_from']] if row.get('mirrored_from') else row
        checks = editor.validate_tail(q, source['ending']['edited_from_frame'], fps)
        assert checks['passed'], (path.name, checks)
        for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
            assert np.max(np.abs(z[key][-50:]-z[key][-1:])) < 1e-12, (path.name, key)
        for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
            assert np.max(np.abs(z[key][-50:])) < 1e-9, (path.name, key)
        lower, upper = editor.model.jnt_range[1:].T
        joint_excess = float(np.maximum(lower-q[:, 7:], q[:, 7:]-upper).max())
        assert joint_excess < 1e-6, (path.name, joint_excess)
        max_speed = float((np.abs(states[:, editor.model.nq+6:])/editor.speeds).max())
        assert max_speed < 1.01, (path.name, max_speed)
        max_penetration = 0.
        for pose in q:
            assert editor.geometry(pose)[2][..., 2].min() >= -2e-6
            for contact in editor.data.contact:
                if 0 not in [int(editor.model.geom_bodyid[g]) for g in contact.geom]:
                    max_penetration = max(max_penetration, -float(contact.dist))
        assert max_penetration < .003, (path.name, max_penetration)
        records.append(dict(file=path.name, sha256=sha(path), frames=len(q),
                            full_clip_max_self_penetration_m=max_penetration,
                            full_clip_max_joint_speed_fraction=max_speed, ending=checks))
        print(f'Audited {path.name}', flush=True)
    # The reference comparison baseline must remain exactly as it was.
    old_manifest = ROOT/'retargeted_motion/training_list/manifest.json'
    old = json.loads(old_manifest.read_text())
    config = json.loads((ROOT/manifest['selection_config']).read_text())
    assert sha(old_manifest) == config['source_library_manifest_sha256']
    assert all(sha(old_manifest.parent/r['file']) == r['sha256'] for r in old['motions'])
    result = dict(passed=True, clips=len(records), frames=sum(r['frames'] for r in records),
                  old_library_hashes_unchanged=True, manifest_sha256=sha(args.directory/'manifest.json'),
                  generator_hash_matches=sha(ROOT/manifest['generator']) == manifest['generator_sha256'],
                  complete_static_hold_seconds=1., full_clip_collision_threshold_m=.003,
                  maximum_static_torque_fraction=max(r['ending']['final_support']['peak_torque_fraction'] for r in records),
                  all_final_planted_settling_frames_force_feasible=True, motions=records,
                  limitation='No learned-policy rollout or full-motion dynamic-feasibility certificate. The inverse-dynamics checks apply to the final planted settling phase and hold.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'motions'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'retargeted_motion/training_static_end')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/static_end_build_20261125/endpoint_validation.json')
    main(parser.parse_args())
