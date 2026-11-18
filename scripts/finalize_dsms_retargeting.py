"""Audit every DSMS attempt and collect only full-length candidates passing the screen."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from retarget_force2_dsms import (
    INPUT, REPO, ADAPTER_SHA256, MODEL_ADAPTER_SHA256, MPCTrackingConfig,
    atomic_json, actuator_limits, replay_metrics, sha,
)
from src.dynamics import Dynamics


def audit(folder):
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    batch = json.loads((folder / 'batch.json').read_text())
    expected = {Path(r['file']).stem: r for r in manifest['motions']}
    assert set(batch['motions']) == set(expected)
    assert batch['source_manifest_sha256'] == sha(INPUT / 'manifest.json')
    files = {p.parent.name: p for p in folder.glob('clips/*/assessment.json')}
    assert set(files) == set(expected), f'Incomplete attempts: {len(files)}/{len(expected)}'
    accepted = folder / 'accepted_candidates'
    accepted.mkdir(exist_ok=True)
    rows, selected = [], []
    for name, path in sorted(files.items()):
        r = json.loads(path.read_text())
        assert sha(INPUT / (name + '.npz')) == expected[name]['sha256']
        if r.get('error'):
            rows.append(dict(motion=name, complete=False, accepted=False, failure=r['error']))
            continue
        assert r['source_sha256'] == expected[name]['sha256']
        assert r['adapter_sha256'] == ADAPTER_SHA256 and r['model_adapter_sha256'] == MODEL_ADAPTER_SHA256
        assert r['model_sha256'] == sha(folder / 'model/t1_force2.xml')
        z = dict(np.load(path.parent / 'rollout.npz', allow_pickle=False))
        assert len(z['state']) == len(z['input']) + 1
        assert all(np.isfinite(z[k]).all() for k in ('state', 'input', 'reference', 'time'))
        np.testing.assert_allclose(np.diff(z['time']), r['settings']['sim_dt'], atol=1e-12)
        cfg = MPCTrackingConfig(**r['config'])
        dyn = Dynamics(cfg.build_dynamics_config(str(REPO)))
        metrics = replay_metrics(dyn.model, z['state'], z['input'], z['reference'], actuator_limits())
        for key, value in metrics.items():
            np.testing.assert_allclose(value, r['metrics'][key], atol=1e-8, rtol=1e-8)
        assert metrics['replay_max_state_error'] < 1e-7
        candidate = dict(np.load(path.parent / 'candidate.npz', allow_pickle=False))
        assert all(np.isfinite(candidate[k]).all() for k in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'))
        stride = round(.02 / r['settings']['sim_dt'])
        assert len(candidate['joint_pos']) == len(z['state'][::stride])
        expected_pass = r['complete'] and r['trial_seconds'] is None and all(metrics[k] <= v for k, v in r['screen_thresholds'].items())
        assert expected_pass == r['accepted']
        if r['complete']:
            assert len(candidate['joint_pos']) == expected[name]['frames']
        if r['accepted']:
            dest = accepted / (name + '.npz')
            shutil.copy2(path.parent / 'candidate.npz', dest)
            selected.append(dict(file=dest.name, frames=len(candidate['joint_pos']), fps=50., sha256=sha(dest),
                                 source_sha256=r['source_sha256'], assessment=str(path.relative_to(folder))))
        rows.append(dict(motion=name, complete=r['complete'], accepted=r['accepted'],
                         simulated_seconds=r['simulated_seconds'], root_rmse_m=metrics['root_rmse_m'],
                         joint_rmse_deg=metrics['joint_rmse_deg'], max_penetration_m=metrics['max_contact_penetration_m'],
                         speed_limit_fraction=metrics['max_speed_limit_fraction'],
                         failure=r['failure'] or ','.join(r['screen_failures']), wall_seconds=r['wall_seconds']))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (folder / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(accepted / 'manifest.json', dict(motions=selected, total_frames=sum(r['frames'] for r in selected),
                                               note='Only complete MuJoCo candidates passing the declared screen; not installed or validated in PhysX.'))
    result = dict(attempted=len(rows), expected=len(expected), completed=sum(r['complete'] for r in rows),
                  accepted=len(selected), errors=sum(bool(json.loads(p.read_text()).get('error')) for p in files.values()),
                  exact_independent_replays_verified=True, active_inputs_unchanged=True,
                  candidate_manifest_sha256=sha(accepted / 'manifest.json'), passed=True)
    atomic_json(folder / 'verification.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    audit(args.folder.resolve())
