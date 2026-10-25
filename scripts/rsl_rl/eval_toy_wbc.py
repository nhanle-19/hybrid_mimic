"""Run the toes-up standing fixture with explicitly supplied manual NN outputs."""
import argparse
import math
import os
from pathlib import Path
import sys

from manual_wbc_inputs import load_manual_inputs
from manual_wbc_schedule import decode_contact_values_numpy, load_schedule

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('force', 'authority'), required=True)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--inputs', type=Path, help='Default: toy_sway_wbc_inputs.json for tracking, toy_wbc_inputs.json for pose hold.')
    input_group.add_argument('--manual_schedule', type=Path, help='Per-reference-frame raw outputs; requires --track_motion.')
    input_group.add_argument('--build_manual_schedule', type=Path, help='Build and run a reference-derived schedule; requires --track_motion.')
    parser.add_argument('--motion_file', type=Path, help='Default: reference.npz in the selected toy directory.')
    parser.add_argument('--track_motion', action='store_true', help='Follow the sway clip once instead of holding frame zero.')
    parser.add_argument('--force_max', type=float, nargs=2, metavar=('LEFT_N', 'RIGHT_N'),
                        help='FORCE per-foot maxima; defaults to 1000 N each for sway, task defaults for pose hold.')
    parser.add_argument('--output_dir', type=Path,
                        help='Default: force or authority_cost_only under the selected toy directory.')
    parser.add_argument('--steps', type=int, help='50 Hz steps; default: full clip for tracking, 500 for pose hold.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--qp_backend', choices=('batched', 'osqp'), default='batched')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    scheduled = args.manual_schedule is not None or args.build_manual_schedule is not None
    if scheduled and not args.track_motion:
        parser.error('Manual schedules require --track_motion')
    if args.force_max is not None and (args.controller != 'force' or
            not all(math.isfinite(x) and x > 0 for x in args.force_max)):
        parser.error('--force_max requires FORCE and two positive finite maxima')
    if args.track_motion and args.controller == 'force' and args.force_max is None:
        args.force_max = [1000., 1000.]
    toy_dir = ROOT / 'eval_data' / ('toy_standing_sway' if args.track_motion else 'toy_standing')
    if not scheduled:
        args.inputs = args.inputs or ROOT / 'configs' / ('toy_sway_wbc_inputs.json' if args.track_motion else 'toy_wbc_inputs.json')
    args.motion_file = args.motion_file or toy_dir / 'reference.npz'
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    if not scheduled:
        try:
            values = load_manual_inputs(args.inputs, args.controller)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if not args.motion_file.is_file():
        parser.error('Reference missing; first run scripts/toy_standing_reference.py --headless --device cpu'
                     + (' --sway' if args.track_motion else ''))
    if args.track_motion:
        import numpy as np
        with np.load(args.motion_file, allow_pickle=False) as archive:
            frames = len(archive['joint_pos'])
            if not np.isclose(float(archive['fps'].item()), 50.):
                parser.error('Toy tracking requires a 50 FPS reference')
        args.steps = frames if args.steps is None else args.steps
        if not frames or args.steps > frames:
            parser.error('Toy tracking needs a nonempty clip and --steps no greater than its frame count')
        if args.manual_schedule is not None:
            try:
                load_schedule(args.manual_schedule, args.controller, args.motion_file, .02, frames)
            except (ValueError, OSError, KeyError) as error:
                parser.error(str(error))
    elif args.steps is None:
        args.steps = 500
    output_name = 'authority_cost_only' if args.controller == 'authority' else 'force'
    if scheduled:
        output_name += '_scheduled'
    output = (args.output_dir or toy_dir / output_name).resolve()
    mode = 'Single-pass tracking' if args.track_motion else 'Pose hold: frame zero'
    print(f'[INFO] {mode} of {args.motion_file.resolve()}, '
          f'{args.steps} control steps ({args.steps / 50:g} s), stop on first failure.', flush=True)
    if scheduled:
        print(f'[INFO] Per-frame manual outputs: {args.manual_schedule or args.build_manual_schedule}', flush=True)
    else:
        print(f'[INFO] Constant raw NN outputs: {values}', flush=True)
        fractions = decode_contact_values_numpy(values).tolist()
    if not scheduled and args.controller == 'authority':
        print(f'[INFO] Cost authority: left xyz={fractions[:3]}, right xyz={fractions[3:]}. '
              'No rotational contact cost or authority-dependent wrench bounds.', flush=True)
        print(f'[INFO] Effective xyz contact weights: {[100*v*v for v in fractions]} (base weights 100).', flush=True)
    elif not scheduled:
        maxima = args.force_max or [600., 600.]
        print(f'[INFO] Force fractions: {fractions}; normal-force caps: '
              f'{[maximum*v for maximum, v in zip(maxima, fractions)]} N (independent per foot).', flush=True)
    print(f'[INFO] Toy evaluation results: {output}', flush=True)
    command = [sys.executable, str(ROOT / 'scripts/rsl_rl/eval_wbc.py'),
               '--task', f'Toy-Standing-WBC-{args.controller.upper()}-T1-v0',
               '--qp_only',
               '--motion_file', str(args.motion_file.resolve()), '--num_envs', '1',
               '--steps', str(args.steps), '--device', args.device, '--qp_backend', args.qp_backend,
               '--fail_on_reset', '--output', str(output / 'diagnostics.npz'),
               '--video_dir', str(output / 'videos')]
    if scheduled:
        option = '--manual_schedule' if args.manual_schedule else '--build_manual_schedule'
        command.extend([option, str((args.manual_schedule or args.build_manual_schedule).resolve())])
    else:
        command.extend(['--manual_inputs', str(args.inputs.resolve())])
    if args.track_motion:
        command.append('--track_motion')
    if args.force_max is not None:
        command.extend(['--force_max', *map(str, args.force_max)])
    if args.video:
        command.append('--video')
    if args.headless:
        command.append('--headless')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
