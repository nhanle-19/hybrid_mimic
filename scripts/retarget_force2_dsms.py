"""Refine FORCE2's complete motion set using the cloned Shooting for Contact repo.

Run with .venvs/dsms/bin/python. Results are continuous, actuated MuJoCo
rollouts, never stitched optimizer states. Originals and mirrors are solved
independently. Quality screening happens after generation, without shortening
clips. This checkout uses CPU MuJoCo/IPOPT; it has no GPU optimizer backend.
Active training references are never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import mujoco
import numpy as np

from dsms_t1_model import ROOT, actuator_limits, build_model, load_reference, sha, validate_references

REPO = ROOT / 'shooting-for-contact'
os.environ['TRAJOPT_ROOT_DIR'] = str(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples/g1_tracking_mpc'))
from config import MPCTrackingConfig
from g1_tracking_mpc import G1TrackingMPC

DEFAULT_OUTPUT = ROOT / 'outputs/force2_dsms_full70'
INPUT = ROOT / 'retargeted_motion/training_list'
ADAPTER_SHA256 = sha(__file__)
MODEL_ADAPTER_SHA256 = sha(ROOT / 'scripts/dsms_t1_model.py')
SCREEN_THRESHOLDS = dict(replay_max_state_error=1e-7, max_torque_limit_fraction=1.000001,
                        max_speed_limit_fraction=1.01, max_joint_limit_violation_rad=.01,
                        max_contact_penetration_m=.005, torso_head_floor_contact_samples=0,
                        root_rmse_m=.10, max_root_error_m=.25, joint_rmse_deg=15.)


class T1TrackingMPC(G1TrackingMPC):
    """Use the upstream robot-independent tracking objective and solver unchanged."""

    def pd_warmstart(self, H, x_init):
        # A bounded seed only; these gains do not actuate the optimized rollout.
        x = x_init.copy()
        states, controls = [x.copy()], []
        kp = np.minimum(80., self.gear * 2.)
        kd = 2. * np.sqrt(kp * .05)
        for k in range(H):
            ref = self._ref(k * self.dt)
            tau = kp * (ref[self.act_qadr] - x[self.act_qadr])
            tau += kd * (ref[self.dyn.nq + self.act_vadr] - x[self.dyn.nq + self.act_vadr])
            u = np.clip(tau / self.gear, self.dyn.u_lb, self.dyn.u_ub)
            x = self.dyn.dynamics(x, u)
            states.append(x.copy())
            controls.append(u)
        return np.array(states), np.array(controls)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def export_motion(path, model, states, template, fps):
    """Export executed poses/velocities without floor lifting, clamping or smoothing."""
    d = mujoco.MjData(model)
    names = template['body_names'].tolist()
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in template['joint_names'].tolist()]
    positions, quaternions, linear, angular = [], [], [], []
    for x in states:
        d.qpos[:] = x[:model.nq]
        d.qvel[:] = x[model.nq:]
        mujoco.mj_forward(model, d)
        velocity = np.empty((len(bids), 6))
        for j, bid in enumerate(bids):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, d, jp, jr, bid)
            velocity[j, :3], velocity[j, 3:] = jr @ d.qvel, jp @ d.qvel
        positions.append(d.xpos[bids].copy())
        quaternions.append(d.xquat[bids].copy())
        angular.append(velocity[:, :3].copy())
        linear.append(velocity[:, 3:].copy())
    np.savez_compressed(path, fps=np.array([fps]), joint_names=template['joint_names'], body_names=template['body_names'],
                        joint_pos=states[:, model.jnt_qposadr[joint_ids]], joint_vel=states[:, model.nq + model.jnt_dofadr[joint_ids]],
                        body_pos_w=np.array(positions), body_quat_w=np.array(quaternions),
                        body_lin_vel_w=np.array(linear), body_ang_vel_w=np.array(angular),
                        preparation_note=np.array('Continuous DSMS MuJoCo rollout; no post-hoc pose corrections. See per-clip assessment.'))


def physical_context(model, motors):
    joints = model.actuator_trnid[:, 0]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    return dict(qadr=model.jnt_qposadr[joints], vadr=model.jnt_dofadr[joints], ranges=model.jnt_range[joints],
                speeds=np.array([motors[n]['speed'] for n in names]), efforts=np.array([motors[n]['effort'] for n in names]))


def physical_sample(model, d, ctx):
    penetration = 0.
    forbidden = 0
    for c in d.contact:
        penetration = max(penetration, -float(c.dist))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in c.geom]
        if 'world' in names and any(n in ('Trunk', 'H1', 'H2') for n in names) and c.dist < 0:
            forbidden += 1
    q = d.qpos[ctx['qadr']]
    return dict(max_contact_penetration_m=penetration, torso_head_floor_contact_samples=forbidden,
                max_torque_limit_fraction=float(np.max(np.abs(d.qfrc_actuator[ctx['vadr']]) / ctx['efforts'])),
                max_speed_limit_fraction=float(np.max(np.abs(d.qvel[ctx['vadr']]) / ctx['speeds'])),
                max_joint_limit_violation_rad=max(0., float(np.maximum(ctx['ranges'][:, 0] - q, q - ctx['ranges'][:, 1]).max())))


def replay_metrics(model, states, controls, reference, motors):
    """Independent continuous replay of all controls; inspect every physics step."""
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = states[0, :model.nq], states[0, model.nq:]
    replay_error = penetration = torque_fraction = speed_fraction = joint_excess = 0.
    forbidden_contacts = 0
    joints = model.actuator_trnid[:, 0]
    limits = np.array([motors[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]['speed'] for j in joints])
    efforts = np.array([motors[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]['effort'] for j in joints])
    for i, u in enumerate(controls):
        d.ctrl[:] = u
        mujoco.mj_step(model, d)
        replay_error = max(replay_error, float(np.abs(np.r_[d.qpos, d.qvel] - states[i + 1]).max()))
        torque_fraction = max(torque_fraction, float(np.max(np.abs(d.qfrc_actuator[model.jnt_dofadr[joints]]) / efforts)))
        speed_fraction = max(speed_fraction, float(np.max(np.abs(d.qvel[model.jnt_dofadr[joints]]) / limits)))
        qj = d.qpos[model.jnt_qposadr[joints]]
        joint_excess = max(joint_excess, float(np.maximum(model.jnt_range[joints, 0] - qj, qj - model.jnt_range[joints, 1]).max()))
        for c in d.contact:
            penetration = max(penetration, -float(c.dist))
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in c.geom]
            if 'world' in names and any(n in ('Trunk', 'H1', 'H2') for n in names) and c.dist < 0:
                forbidden_contacts += 1
    e = states[:, :3] - reference[:len(states), :3]
    jerr = states[:, 7:model.nq] - reference[:len(states), 7:model.nq]
    return dict(replay_max_state_error=replay_error, max_contact_penetration_m=penetration,
                max_torque_limit_fraction=torque_fraction, max_speed_limit_fraction=speed_fraction,
                max_joint_limit_violation_rad=joint_excess, torso_head_floor_contact_samples=forbidden_contacts,
                root_rmse_m=float(np.sqrt(np.mean(np.sum(e * e, axis=1)))),
                max_root_error_m=float(np.linalg.norm(e, axis=1).max()),
                joint_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(jerr * jerr)))))


def optimize_rollout(mpc, initial_state, initial_iterations, iterations, report):
    """Generate the requested duration; assess motion quality only afterward.

    Finite, bounded controls are executed even when IPOPT reports that it has
    not converged. All solver statuses remain in the result. Only unusable
    controls, nonfinite simulation, or an exception can interrupt generation.
    """
    x = initial_state.copy()
    states, controls, solves = [x.copy()], [], []
    failure = None
    count = mpc.n_steps
    while len(controls) < count:
        k = len(controls)
        mpc.t_now = k * mpc.dt
        mpc.problem.add_option('max_iter', initial_iterations if k == 0 else iterations)
        tick = time.perf_counter()
        _, planned, U, info = mpc.solve_step(x)
        if not np.isfinite(U).all():
            failure = 'nonfinite_optimizer_control'
            break
        if len(U) == 0:
            failure = 'empty_optimizer_control'
            break
        status = int(info['status'])
        defect = float(np.abs(mpc.constraints(mpc.pack(planned, mpc.spline.fit(U)))).max())
        solves.append(dict(step=k, status=status, seconds=time.perf_counter() - tick,
                           objective=float(info['obj_val']), max_constraint_residual=defect))
        for u in U[:min(mpc.steps_per_solve, count - k)]:
            if np.max(np.maximum(mpc.dyn.u_lb - u, u - mpc.dyn.u_ub)) > 1e-6:
                failure = 'control_bound_violation'
                break
            u = np.clip(u, mpc.dyn.u_lb, mpc.dyn.u_ub)
            x = mpc.dyn.dynamics(x, u)
            if not np.isfinite(x).all():
                failure = 'nonfinite_simulation'
                break
            controls.append(u.copy())
            states.append(x.copy())
        report(len(controls), x, solves, failure)
        if failure:
            break
    return np.array(states), np.array(controls).reshape(-1, mpc.dyn.nu), solves, failure


def run_clip(name, output, settings, trial_seconds=None):
    output = Path(output)
    folder = output / 'clips' / name
    folder.mkdir(parents=True, exist_ok=True)
    input_path = INPUT / (name + '.npz')
    input_hash = sha(input_path)
    model_path = output / 'model/t1_force2.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    template, reference, fps = load_reference(input_path, model)
    duration = (len(reference) - 1) / fps
    if trial_seconds is not None:
        duration = min(duration, trial_seconds)
    cfg = MPCTrackingConfig(name=name, traj_dir=(), model_xml=(str(model_path.resolve()),),
                            actuator_mode=settings['actuator_mode'], sim_dt=settings['sim_dt'], node_dt=settings['node_dt'],
                            mpc_dt=settings['mpc_dt'], H=settings['horizon'], M=settings['horizon'],
                            n_threads=settings['threads'], max_iter_initial=settings['initial_iterations'],
                            max_iter=settings['iterations'], fd_centered=False,
                            track_state=True, track_body=True, w_joint_pos=.5,
                            term_scale=settings['terminal_weight'], friction=[.6, 0., 0.], condim=3,
                            solref=(.02, 1.), solimp=(0., .95, .001, .5, 2.),
                            ee_anchor='Trunk',
                            ee_track=((1, 'left_foot_link', 10., 1., 0., 0., False),
                                      (1, 'right_foot_link', 10., 1., 0., 0., False)))
    # The upstream tracking subclass replaces self.cfg after base construction;
    # keep-best options must live on that final config too.
    cfg.keep_best_sol = True
    cfg.keep_best_sol_rho = 1e3
    lower, upper = np.full(model.nq + model.nv, -np.inf), np.full(model.nq + model.nv, np.inf)
    motors = actuator_limits()
    for j in range(1, model.njnt):
        lower[model.jnt_qposadr[j]], upper[model.jnt_qposadr[j]] = model.jnt_range[j]
        name_j = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        speed = motors[name_j]['speed']
        lower[model.nq + model.jnt_dofadr[j]], upper[model.nq + model.jnt_dofadr[j]] = -speed, speed
    mc = cfg.build_mpc_config(duration, lower, upper)
    mc.verbose = False
    mc.keep_best_sol = True
    mpc = T1TrackingMPC(cfg.build_dynamics_config(str(REPO)), mc, cfg, reference, 1 / fps)
    start = time.perf_counter()
    x = reference[0].copy()
    mpc.set_warm_start(*mpc.warm_start(mpc.N_sim, x))

    def report(steps, state, solves, failure):
        last = solves[-1]
        if len(solves) % 10 == 0 or failure or steps == mpc.n_steps:
            print(f'{name}: {steps * mpc.dt:.2f}/{duration:.2f}s, wall={time.perf_counter() - start:.1f}s, '
                  f'root={np.linalg.norm(state[:3] - mpc._ref(steps * mpc.dt)[:3]):.3f}m, '
                  f'status={last["status"]}, residual={last["max_constraint_residual"]:.3g}, stop={failure}', flush=True)
        atomic_json(folder / 'progress.json', dict(simulated_seconds=steps * mpc.dt, duration_seconds=duration,
                                                 wall_seconds=time.perf_counter() - start, failure=failure))

    X, U, solves, failure = optimize_rollout(mpc, x, cfg.max_iter_initial, cfg.max_iter, report)
    metrics = replay_metrics(mpc.dyn.model, X, U, mpc.X_ref, actuator_limits())
    complete = failure is None and len(U) == mpc.n_steps
    # Explicit screening thresholds, not proof of optimality or hardware feasibility.
    screen_failures = [key for key, threshold in SCREEN_THRESHOLDS.items() if metrics[key] > threshold]
    accepted = complete and trial_seconds is None and not screen_failures
    result = dict(motion=name, complete=complete, accepted=accepted, failure=failure,
                  trial_seconds=trial_seconds, expected_frames=len(reference), simulated_seconds=len(U) * mpc.dt,
                  wall_seconds=time.perf_counter() - start, metrics=metrics, source=str(input_path), source_sha256=input_hash,
                  screen_failures=screen_failures, screen_thresholds=SCREEN_THRESHOLDS,
                  adapter_sha256=ADAPTER_SHA256, model_adapter_sha256=MODEL_ADAPTER_SHA256,
                  model_sha256=sha(model_path), config=asdict(cfg), settings=settings,
                  repo_commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                  solver_converged_count=sum(s['status'] in (0, 1) for s in solves), solver_calls=len(solves),
                  interpretation='Continuous actuated MuJoCo replay with bounded joint torques. Screening success is not solver convergence or PhysX/hardware certification.')
    np.savez_compressed(folder / 'rollout.npz', time=np.arange(len(X)) * mpc.dt, state=X, input=U,
                        reference=mpc.X_ref[:len(X)], metadata_json=json.dumps(result))
    atomic_json(folder / 'solves.json', solves)
    assert sha(input_path) == input_hash, 'Source reference changed during optimization'
    atomic_json(folder / 'assessment.json', result)
    stride = round((1 / fps) / mpc.dt)
    assert abs(stride * mpc.dt - 1 / fps) < 1e-9
    export_motion(folder / 'candidate.npz', mpc.dyn.model, X[::stride], template, fps)
    if mpc.dyn._executor is not None:
        mpc.dyn._executor.shutdown()
    return result


def worker(name, output, settings, trial_seconds):
    folder = Path(output) / 'clips' / name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            return run_clip(name, output, settings, trial_seconds)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(motion=name, complete=False, accepted=False, error=repr(error))
            atomic_json(folder / 'assessment.json', result)
            return result


def summarize(output, results, expected):
    results = sorted(results, key=lambda r: r['motion'])
    report = dict(expected_clips=expected, attempted_clips=len(results), complete_clips=sum(r['complete'] for r in results),
                  accepted_clips=sum(r['accepted'] for r in results), motions=results,
                  scope='Each original and mirror independently optimized; no active training inputs replaced.')
    atomic_json(output / 'summary.json', report)
    body = '<!doctype html><meta charset="utf-8"><title>FORCE2 DSMS retargeting</title>'
    body += '<style>body{font:16px system-ui;margin:40px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
    body += '<h1>Shooting for Contact: T1 motion refinement</h1>'
    body += f'<p>{len(results)} / {expected} attempted; {report["complete_clips"]} completed; {report["accepted_clips"]} passed screening.</p>'
    body += '<p>Continuous actuated MuJoCo rollouts using FORCE2 inertias, limits and colliders, with bounded joint torques. Passing screening does not establish PhysX or hardware feasibility. Existing training inputs are unchanged.</p>'
    body += '<table><tr><th>Motion</th><th>Complete</th><th>Screen passed</th><th>Root RMSE</th><th>Joint RMSE</th><th>Reason / details</th></tr>'
    for r in results:
        m = r.get('metrics', {})
        body += f'<tr><td>{html.escape(r["motion"])}</td><td>{r["complete"]}</td><td>{r["accepted"]}</td>'
        body += (f'<td>{m["root_rmse_m"] * 100:.1f} cm</td><td>{m["joint_rmse_deg"]:.1f}°</td>' if m else '<td>—</td><td>—</td>')
        reason = r.get('error') or r.get('failure') or ', '.join(r.get('screen_failures', [])) or 'Assessment'
        body += f'<td><a href="clips/{r["motion"]}/assessment.json">{html.escape(str(reason))}</a></td></tr>'
    (output / 'index.html').write_text(body + '</table>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--motions', nargs='*')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--trial-seconds', type=float)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--initial-iterations', type=int, default=80)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--sim-dt', type=float, default=.002)
    parser.add_argument('--node-dt', type=float, default=.02)
    parser.add_argument('--mpc-dt', type=float, default=.04)
    parser.add_argument('--terminal-weight', type=float, default=100.)
    parser.add_argument('--actuator-mode', choices=['position', 'torque'], default='position')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.output / 'model')
    atomic_json(args.output / 'input_validation.json', validate_references(INPUT, model))
    if args.prepare_only:
        return
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    names = [Path(r['file']).stem for r in manifest['motions']]
    if args.motions:
        assert set(args.motions) <= set(names), 'Unknown motion'
        names = args.motions
    settings = {k: getattr(args, k) for k in ('horizon', 'initial_iterations', 'iterations', 'sim_dt', 'node_dt', 'mpc_dt', 'threads', 'terminal_weight', 'actuator_mode')}
    results, todo = [], []
    for name in names:
        path = args.output / 'clips' / name / 'assessment.json'
        if path.exists() and args.resume:
            r = json.loads(path.read_text())
            assert r.get('settings') == settings and r.get('trial_seconds') == args.trial_seconds, 'Resume settings changed'
            assert r['source_sha256'] == sha(INPUT / (name + '.npz')) and r['model_sha256'] == sha(model)
            assert r['adapter_sha256'] == ADAPTER_SHA256 and r['model_adapter_sha256'] == MODEL_ADAPTER_SHA256, 'Resume adapter changed'
            results.append(r)
        else:
            todo.append(name)
    atomic_json(args.output / 'batch.json', dict(motions=names, settings=settings, trial_seconds=args.trial_seconds,
                                               source_manifest_sha256=sha(INPUT / 'manifest.json'),
                                               backend='cpu', screening='after_generation'))
    print(f'CPU MuJoCo/IPOPT: {len(todo)} clips to generate; quality screening runs afterward.', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(worker, n, args.output, settings, args.trial_seconds): n for n in todo}
        for job in as_completed(jobs):
            r = job.result()
            results.append(r)
            print(json.dumps({k: r[k] for k in ('motion', 'complete', 'accepted')}), flush=True)
            summarize(args.output, results, len(names))
    summarize(args.output, results, len(names))


if __name__ == '__main__':
    main()
