"""Run contact evaluation on every motion in a manifest, then build a report."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def newest_checkpoint(root):
    # Download/copy mtimes do not indicate training order.
    runs = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob('model_*.pt')))
    if not runs:
        raise ValueError(f'No checkpoints in {root}')
    return max(runs[-1].glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--motions', type=Path, default=ROOT/'retargeted_motion/training_list')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--posture_kp', type=float, help='Evaluation-only posture position gain.')
    parser.add_argument('--posture_kd', type=float, help='Evaluation-only posture velocity gain.')
    parser.add_argument('--manual_labels_root', type=Path, help='Replay <root>/<motion>/annotations.npz instead of the policy.')
    args = parser.parse_args()
    overrides = {key: value for key, value in (
        ('posture_position_gain', args.posture_kp), ('posture_velocity_gain', args.posture_kd)) if value is not None}
    if any(not math.isfinite(value) or value <= 0 for value in overrides.values()):
        parser.error('Posture gains must be finite and positive')
    checkpoint = (args.checkpoint or newest_checkpoint(ROOT/'logs/rsl_rl/t1_wbc_force')).resolve()
    motions = args.motions.resolve()
    manifest = json.loads((motions/'manifest.json').read_text())
    listed = {m['file'] for m in manifest['motions']}
    if listed != {p.name for p in motions.glob('*.npz')}:
        raise ValueError('Manifest must cover every motion NPZ exactly')
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for motion in manifest['motions']:
        path = motions/motion['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != motion['sha256']:
            raise ValueError(f'Motion hash differs from manifest: {path}')
        output = args.output.resolve()/path.stem
        output.mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT/'scripts/rsl_rl/eval_force_contacts.py'),
               '--checkpoint', str(checkpoint), '--motion_file', str(path),
               '--output', str(output), '--device', args.device, '--headless']
        for flag in ('posture_kp', 'posture_kd'):
            if getattr(args, flag) is not None:
                cmd.extend(['--'+flag, str(getattr(args, flag))])
        manual_hash = None
        if args.manual_labels_root:
            labels = args.manual_labels_root.resolve()/path.stem/'annotations.npz'
            manual_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
            cmd.extend(['--manual_labels', str(labels)])
        entry = dict(motion=path.name, command=cmd)
        if not args.skip_existing and (output/'diagnostics.npz').exists():
            raise FileExistsError(f'{output}: choose a new output or explicitly use --skip_existing')
        if args.skip_existing and (output/'diagnostics.npz').exists():
            meta = json.loads((output/'run.json').read_text())
            if meta['checkpoint_sha256'] != checkpoint_hash or meta['motion_sha256'] != digest:
                raise ValueError('Existing results belong to a different checkpoint or reference')
            if meta.get('controller_overrides', {}) != overrides:
                raise ValueError('Existing results use different controller overrides')
            if meta.get('manual_labels_sha256') != manual_hash:
                raise ValueError('Existing results use a different policy or manual label schedule')
            if meta.get('qp_contact_model') != 'sole_corners':
                raise ValueError('Existing results use a different QP contact model')
            if meta.get('force_cap_scope') != 'per_sole_vertex':
                raise ValueError('Existing results use a different QP force-cap layout')
            entry['existing'] = True
        else:
            print(f'[SUITE] Starting {path.name}', flush=True)
            started = time.monotonic()
            with (output/'eval.log').open('w') as log:
                status = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4'))
            entry.update(exit_code=status.returncode, elapsed_s=time.monotonic()-started)
        entry['diagnostics_present'] = (output/'diagnostics.npz').exists()
        runs.append(entry)
        (args.output/'suite.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash, controller_overrides=overrides,
            policy_source='manual_labels' if args.manual_labels_root else 'learned_policy',
            qp_contact_model='sole_corners',
            force_cap_scope='per_sole_vertex', policy_action_dim=8,
            manifest=manifest, runs=runs), indent=2)+'\n')
        print('[SUITE] '+json.dumps(entry), flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/analyze_force_contacts.py'), str(args.output.resolve())], check=True)
    if not all(r['diagnostics_present'] for r in runs):
        raise RuntimeError('Some motions failed to produce diagnostics; see their eval.log files')


if __name__ == '__main__':
    main()
