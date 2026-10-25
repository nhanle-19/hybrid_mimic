"""Tune an open-loop FORCE action table using short first-failure physics trials.

Only the two force-cap logits change. This does not train a neural network,
change the controller, or relax physical/tracking constraints. Batched results
must be replayed in a single environment before accepting a schedule.
"""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--schedule', type=Path, required=True)
parser.add_argument('--motion_file', required=True)
parser.add_argument('--output_dir', type=Path, required=True)
parser.add_argument('--candidates', type=int, default=32)
parser.add_argument('--iterations', type=int, default=6)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--skip_initial_grid', action='store_true', help='Refine an existing tuned table without repeating constant-cap trials')
parser.add_argument('--knot_stride', type=int, help='Optional finer control table knot spacing, in frames')
parser.add_argument('--local_search', action='store_true', help='Refine short time windows around the best action table')
parser.add_argument('--serial', action='store_true', help='Evaluate candidates sequentially in one environment')
parser.add_argument('--repeat_only', action='store_true', help='Replay identical tables to check batched repeatability')
parser.add_argument('--cap_space_search', action='store_true', help='Explore one-foot force ceilings directly within short windows')
parser.add_argument('--start_frame', type=int, default=0, help='Preserve the prefix before this frame during local search')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.candidates < 4 or args.iterations < 1:
    parser.error('Use at least four candidates and one iteration')
if args.knot_stride is not None and args.knot_stride < 1:
    parser.error('--knot_stride must be positive')
if args.start_frame < 0:
    parser.error('--start_frame must be nonnegative')
launcher = AppLauncher(args, fast_shutdown=False)
app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg
import whole_body_tracking.tasks
from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.motion import WBCACCSinglePassMotion
from whole_body_tracking.tasks.tracking.config.t1_wbc_acc.rewards import qp_failed
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TerminationsCfg
from manual_wbc_schedule import (load_schedule, save_schedule, reference_digest,
                                 decode_contact_values_numpy, encode_contact_values_numpy)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    cfg = parse_env_cfg('Tracking-WBC-FORCE-T1-Eval-v0', device=args.device, num_envs=1 if args.serial else args.candidates)
    cfg.seed = args.seed
    cfg.commands.motion.motion_file = args.motion_file
    cfg.commands.motion.class_type = WBCACCSinglePassMotion
    cfg.terminations = TerminationsCfg()
    cfg.terminations.ee_body_pos.params['body_names'] = list(cfg.commands.motion.eef_names)
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed, params={'action_name': 'wbc_force'})
    cfg.actions.wbc_force.record_diagnostics = False
    cfg.actions.wbc_force.allow_cpu_batched_evaluation = args.device == 'cpu'
    schedule = load_schedule(args.schedule, 'force', args.motion_file,
                             cfg.sim.dt*cfg.decimation, controller_config=cfg.actions.wbc_force.controller.to_dict())
    n, count = len(schedule['actions']), args.candidates
    cfg.episode_length_s = n*cfg.sim.dt*cfg.decimation+1.
    env = gym.make('Tracking-WBC-FORCE-T1-Eval-v0', cfg=cfg)
    try:
        env.reset()
        motion = env.unwrapped.command_manager.get_term('motion')
        array = lambda t: t.detach().cpu().numpy()
        digest = reference_digest(array(motion.motion.joint_pos), array(motion.motion._body_pos_w), array(motion.motion._body_quat_w))
        load_schedule(args.schedule, 'force', args.motion_file, env.unwrapped.step_dt, motion.motion.time_step_total,
                      digest, cfg.actions.wbc_force.controller.to_dict())
        ee_indices = [motion.cfg.body_names.index(name) for name in cfg.commands.motion.eef_names]
        knots = np.unique(np.clip([0, 5, 10, 16, 23, 32, 48, 70, 100, n-1], 0, n-1))
        if args.knot_stride is not None:
            knots = np.unique(np.r_[np.arange(0, n, args.knot_stride), n-1])
        best = schedule['actions'].copy()
        mean = best[knots].astype(np.float64)
        spread = np.full_like(mean, 1.4)
        best_score, best_steps = -np.inf, 0
        rng = np.random.default_rng(args.seed)
        records = []
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for iteration in range(args.iterations):
            sampled = np.clip(rng.normal(mean, spread, size=(count, *mean.shape)), -3.66, 4.6)
            tables = np.stack([np.stack([np.interp(np.arange(n), knots, row[:, foot])
                                        for foot in range(2)], -1) for row in sampled]).astype(np.float32)
            tables[0] = best
            sampled[0] = best[knots]
            if args.local_search:
                # Preserve unobserved frames and most of the successful prefix;
                # explore short windows instead of perturbing the whole clip.
                horizon = min(n, max(best_steps, 30)+5)
                sigma = max(.15, .8*(1.-iteration/max(args.iterations, 1)))
                for i in range(1, count):
                    tables[i] = best
                    start = int(rng.integers(min(args.start_frame, horizon-1), horizon))
                    stop = min(horizon, start+int(rng.integers(1, 12)))
                    offset = rng.normal(0., sigma, size=(1, 2))
                    tables[i, start:stop] = np.clip(tables[i, start:stop]+offset, -3.66, 4.6)
                    if args.cap_space_search:
                        tables[i] = best
                        foot = int(rng.integers(0, 2))
                        # Explicit endpoint proposals complement interior cap exploration.
                        choice = int(rng.integers(0, 5))
                        fraction = (0. if choice == 0 else 1. if choice == 1 else rng.uniform(0., 1.))
                        tables[i, start:stop, foot] = encode_contact_values_numpy(fraction)
                    sampled[i] = tables[i, knots]
            if iteration == 0 and not args.skip_initial_grid:
                grid = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]
                grid += [(left, right) for left in (.2, .35, .5, .7, .98) for right in (.05, .2, .4, .7, .98)]
                for i, fractions in enumerate(grid[:count-1], 1):
                    tables[i] = encode_contact_values_numpy(fractions)
                    sampled[i] = tables[i, knots]
            if args.repeat_only:
                tables[:] = best
                sampled[:] = best[knots]
            survived = np.zeros(count, dtype=int)
            costs = np.zeros(count)
            complete = np.zeros(count, dtype=bool)
            reasons = [{} for _ in range(count)]
            batch_size = 1 if args.serial else count
            for first in range(0, count, batch_size):
                env.reset()
                alive = torch.ones(batch_size, device=env.unwrapped.device, dtype=torch.bool)
                values = torch.as_tensor(tables[first:first+batch_size], device=env.unwrapped.device)
                ids = torch.arange(batch_size, device=env.unwrapped.device)
                for step in range(n):
                    current = motion.time_steps.clamp(0, n-1)
                    if not bool((current[alive] == step).all()):
                        raise RuntimeError('Reference/frame alignment lost during tuning')
                    error = (motion.body_pos_w[:, ee_indices, 2]-motion.robot_body_pos_w[:, ee_indices, 2]).abs().amax(-1)
                    costs[first:first+batch_size] += array(torch.where(alive, error, 0.))
                    _, _, terminated, timed_out, _ = env.step(values[ids, current])
                    ended = (terminated | timed_out) & alive
                    for i in torch.where(ended)[0].tolist():
                        survived[first+i] = step+1
                        reasons[first+i] = {key: int(env.unwrapped.termination_manager.get_term(key)[i])
                                           for key in env.unwrapped.termination_manager.active_terms}
                    alive &= ~ended
                    if not bool(alive.any()):
                        break
                complete[first:first+batch_size] = array(alive)
            survived[complete] = n
            scores = survived + n*complete - .1*costs/np.maximum(survived, 1)
            order = np.argsort(scores)[::-1]
            winner = int(order[0])
            improved = scores[winner] > best_score
            if improved:
                best_score, best_steps = float(scores[winner]), int(survived[winner])
                best = tables[winner].copy()
                result = {k: v.copy() for k, v in schedule.items()}
                result['actions'] = best
                result['fractions'] = decode_contact_values_numpy(best)
                meta = json.loads(str(result['metadata_json']))
                meta.update(design='force_cap_table_tuning', status='requires_single_environment_replay',
                            tuning={'iteration': iteration, 'candidate': winner, 'steps': best_steps,
                                    'seed': args.seed, 'num_envs': batch_size})
                meta.pop('validation', None)
                result['metadata_json'] = np.asarray(json.dumps(meta, sort_keys=True))
                save_schedule(args.output_dir/'best_force.npz', result)
            elites = sampled[order[:max(2, count//5)]]
            mean = .3*mean+.7*elites.mean(0)
            spread = np.maximum(.35, .3*spread+.7*elites.std(0))
            row = {'iteration': iteration, 'best_steps': int(survived.max()), 'steps': survived.tolist(),
                   'scores': scores.tolist(), 'termination_reasons': reasons}
            records.append(row)
            (args.output_dir/'trials.json').write_text(json.dumps(records, indent=2)+'\n')
            np.savez_compressed(args.output_dir/f'iteration_{iteration}.npz', actions=tables, steps=survived, scores=scores)
            print(f'[TUNE] iteration={iteration} best_steps={int(survived.max())}/{n} global_best={best_steps}; '
                  f'winner={winner} caps_first={600*decode_contact_values_numpy(tables[winner, 0])}', flush=True)
            if complete.any():
                print('[TUNE] A candidate survived the clip; fresh single-environment replay is required.', flush=True)
                break
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
