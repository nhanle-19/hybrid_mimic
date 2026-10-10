"""Run the toes-up standing fixture with explicitly supplied manual NN outputs."""
import argparse
import os
from pathlib import Path
import sys

from manual_wbc_inputs import load_manual_inputs

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('force', 'authority'), required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'configs/toy_wbc_inputs.json')
    parser.add_argument('--motion_file', type=Path, default=ROOT / 'eval_data/toy_standing/reference.npz')
    parser.add_argument('--output_dir', type=Path, help='Default: eval_data/toy_standing/<controller>')
    parser.add_argument('--steps', type=int, default=500, help='50 Hz control steps; default is 10 seconds.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--qp_backend', choices=('batched', 'osqp'), default='batched')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    try:
        load_manual_inputs(args.inputs, args.controller)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not args.motion_file.is_file():
        parser.error('Reference missing; first run scripts/toy_standing_reference.py --headless --device cpu')
    output = (args.output_dir or ROOT / 'eval_data/toy_standing' / args.controller).resolve()
    command = [sys.executable, str(ROOT / 'scripts/rsl_rl/eval_wbc.py'),
               '--task', f'Toy-Standing-WBC-{args.controller.upper()}-T1-v0',
               '--qp_only', '--manual_inputs', str(args.inputs.resolve()),
               '--motion_file', str(args.motion_file.resolve()), '--num_envs', '1',
               '--steps', str(args.steps), '--device', args.device, '--qp_backend', args.qp_backend,
               '--fail_on_reset', '--output', str(output / 'diagnostics.npz'),
               '--video_dir', str(output / 'videos')]
    if args.video:
        command.append('--video')
    if args.headless:
        command.append('--headless')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
