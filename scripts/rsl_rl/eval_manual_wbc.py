"""Play a saved manual action schedule through one complete motion attempt."""
import argparse
import os
from pathlib import Path
import sys

from manual_wbc_schedule import load_schedule

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('force', 'authority'), required=True)
    parser.add_argument('--schedule', type=Path, help='Default: eval_data/manual_schedules/kick_<controller>_deadzone010.npz')
    parser.add_argument('--motion_file', type=Path, default=ROOT/'retargeted_motion/g18_push_kick_right_t1_training.npz')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--qp_backend', choices=('batched', 'osqp'), default='batched')
    parser.add_argument('--steps', type=int, help='Optional shorter diagnostic run; default is the entire clip')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    schedule = (args.schedule or ROOT/f'eval_data/manual_schedules/kick_{args.controller}_deadzone010.npz').resolve()
    try:
        data = load_schedule(schedule, args.controller, args.motion_file)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if args.steps is not None and not 1 <= args.steps <= len(data['actions']):
        parser.error('--steps must be between one and the schedule length')
    output = (args.output_dir or ROOT/f'eval_data/manual_schedules/{args.controller}_deadzone010_playback').resolve()
    command = [sys.executable, str(ROOT/'scripts/rsl_rl/eval_wbc.py'),
               '--task', f'Tracking-WBC-{args.controller.upper()}-T1-Eval-v0',
               '--qp_only', '--manual_schedule', str(schedule),
               '--motion_file', str(args.motion_file.resolve()), '--num_envs', '1',
               '--device', args.device, '--qp_backend', args.qp_backend,
               '--output', str(output/'diagnostics.npz'), '--video_dir', str(output/'videos')]
    if args.steps is not None:
        command += ['--steps', str(args.steps)]
    if args.video:
        command.append('--video')
    if args.headless:
        command.append('--headless')
    print(f'[INFO] Manual {args.controller}: {len(data["actions"])} scheduled frames from {schedule}', flush=True)
    print(f'[INFO] Results: {output}; stop at first tracking/fall/QP failure.', flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
