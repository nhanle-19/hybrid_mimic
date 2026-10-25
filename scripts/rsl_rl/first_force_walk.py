"""Prepare or launch a small FORCE PPO experiment on the validated walk reference.

Run with --stage check first. On a CUDA training host, run --stage baselines,
then --stage smoke, then --stage train. Add --dry-run to print any command.
Manual labels are used only for baseline evaluation, never by PPO training.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT/'eval_data/force/walk_in_place'
MOTION = ROOT/'retargeted_motion/hdm_bk_walk_in_place_settled_t1_training.npz'


def load_case():
    profile = json.loads((CASE/'controller_profile.json').read_text())
    with np.load(CASE/'walk_force_labels.npz', allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        fractions = archive['fractions'].copy()
        actions = archive['actions'].copy()
        if 'controller_profile_json' in archive.files:
            profiles = json.loads(str(archive['controller_profile_json']))
            if len(profiles) != len(actions) or any(item != profile for item in profiles):
                raise ValueError('The manual replay uses changing controller settings')
    if profile != metadata['controller_config']:
        raise ValueError('Saved controller profile differs from the validated label table')
    if hashlib.sha256(MOTION.read_bytes()).hexdigest() != metadata['motion_sha256']:
        raise ValueError('Reference differs from the validated walk')
    with np.load(MOTION, allow_pickle=False) as archive:
        frames, fps = len(archive['joint_pos']), float(archive['fps'].item())
    if fractions.shape != (frames, 2) or not np.isclose(fps, 50):
        raise ValueError('Expected two labels per frame at the 50 Hz control rate')
    decoded = np.clip((1/(1+np.exp(-actions.astype(np.float64)))-.1)/.8, 0, 1)
    np.testing.assert_allclose(decoded, fractions, atol=1e-6, rtol=0)
    return profile, fractions, frames, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'baselines', 'smoke', 'train'), default='check')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=256, help='Training environments; smoke uses 32.')
    parser.add_argument('--iterations', type=int, default=1000, help='Training iterations; smoke uses 2.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/force_walk_first_run')
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        parser.error('Environment and iteration counts must be positive')
    profile, fractions, frames, fps = load_case()
    if args.stage == 'check':
        print(json.dumps(dict(motion=str(MOTION), frames=frames, duration_s=frames/fps,
            reference_labels=str(CASE/'walk_force_labels.npz'),
            labels_used_for_training=False, initialization='fresh random policy',
            reward=dict(motion_completion='100 points per full executed clip',
                        swing_force_weight=-.5, action_change_weight=-.05,
                        swing_clearance_m=[.001, .005]), controller=profile,
            planned_training=dict(environments=args.num_envs, iterations=args.iterations,
                                  steps_per_environment=24, seed=args.seed),
            validation='The saved reference case was validated on CPU. Run --stage baselines for matched GPU verification.',
            success_checks=['Full deterministic rollout from frame zero, without resets or QP failures.',
                'All five intended foot lifts occur; report tracking error separately from survival.',
                'Deterministic decoded NN outputs vary with motion phase, without exploration noise.',
                'Compare with constant outputs under the same controller settings.']), indent=2))
        return
    if not args.device.startswith('cuda'):
        parser.error('This first training test targets CUDA; use the original runner for CPU evaluation')
    base = [sys.executable, '-u']
    output = args.output_root.resolve()/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{args.stage}'
    if args.stage == 'baselines':
        jobs = [dict(name='manual', schedule=str(CASE/'walk_force_labels.npz'),
                     gate=False, save_trace=True)]
        for name, caps in [('constant_mean', fractions.mean(0).tolist()),
                           ('constant_500', [.5, .5]), ('constant_1000', [1., 1.])]:
            jobs.append(dict(name=name, scheme='constant', fractions=caps,
                             profile=profile, gate=False, save_trace=True))
        command = base+[str(CASE/'runner.py'), '--controller', 'force',
            '--motion_file', str(MOTION), '--plan', str(output/'plan.json'),
            '--output', str(output), '--device', args.device, '--headless']
        if args.dry_run:
            print(json.dumps(dict(baseline_plan=jobs), indent=2))
    else:
        smoke = args.stage == 'smoke'
        command = base+[str(ROOT/'scripts/rsl_rl/train.py'),
            '--task', 'Tracking-WBC-FORCE-T1-v0', '--motion_file', str(MOTION),
            '--num_envs', str(32 if smoke else args.num_envs),
            '--max_iterations', str(2 if smoke else args.iterations),
            '--seed', str(args.seed), '--device', args.device, '--headless',
            '--logger', 'tensorboard', '--run_name', f'force_walk_first_{args.stage}',
            'agent.resume=false', 'agent.save_interval=100',
            'env.actions.wbc_force.friction=0.6']
        # Controller scalars are floats. JSON profiles can encode whole-valued
        # gains/weights as integers, which Isaac Lab's strict config loader
        # rejects. Normalize only the CLI representation of the saved profile.
        command += [f'env.actions.wbc_force.controller.{key}='
                    f'{json.dumps(float(value) if isinstance(value, (int, float)) else value, separators=(",", ":"))}'
                    for key, value in profile.items()]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. No simulator or training run was launched.')
    # Keep all new artifacts separate from the validated manual case.
    output.mkdir(parents=True, exist_ok=False)
    (output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    if args.stage == 'baselines':
        (output/'plan.json').write_text(json.dumps(jobs, indent=2)+'\n')
    env = dict(os.environ, PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    result = subprocess.run(command, cwd=ROOT, env=env)
    if args.stage == 'baselines' and result.returncode == 0:
        trials = json.loads((output/'trials.json').read_text())
        manual = next(item for item in trials if item['name'] == 'manual')
        comparison = dict(manual_completed=manual['complete'],
            manual_qp_failures=manual['qp_failures'],
            trials=[{key: item[key] for key in ('name', 'complete', 'steps', 'qp_failures')}
                    for item in trials],
            constant_completed=any(item['complete'] for item in trials if item['name'] != 'manual'))
        (output/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
        print(json.dumps(comparison, indent=2))
        if not manual['complete'] or manual['qp_failures']:
            raise SystemExit('The GPU manual baseline failed; resolve this before the training test.')
        if comparison['constant_completed']:
            print('A constant baseline completed. Completion alone is not sufficient: '
                  'also compare swing-cap penalties and deterministic policy variation.')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
