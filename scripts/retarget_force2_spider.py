"""Run SPIDER's GPU sampling optimizer on all FORCE2 references and our T1 model.

Full source durations, no tracking/penetration/fall cutoffs, and no wall-clock
timeout. Outputs are separate from active training inputs. Uses the upstream
sampling optimizer and MuJoCo Warp backend without changing the cloned repo.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import mujoco
import numpy as np

from dsms_t1_model import ROOT, build_model, load_reference, sha

REPO = ROOT / 'spider'
INPUT = ROOT / 'retargeted_motion/training_list'
DEFAULT_OUTPUT = ROOT / 'outputs/force2_spider_full70'
sys.path.insert(0, str(REPO))


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def trial_dir(output, name):
    return Path(output) / 'dataset/processed/force2/booster_t1/humanoid' / name / '0'


def upstream_commit():
    # Transfer bundles contain the source and revision without Git history.
    if (REPO / '.git').exists():
        return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    return (REPO / 'UPSTREAM_COMMIT').read_text().strip()


def resample_reference(model, source, fps, sim_dt):
    """Keep the source timestamps and interpolate rotations on their manifold."""
    stride = round(1 / fps / sim_dt)
    if stride < 1 or not np.isclose(stride * sim_dt, 1 / fps):
        raise ValueError('sim-dt must divide the source frame interval exactly')
    q = source[:, :model.nq]
    poses = np.empty(((len(q) - 1) * stride + 1, model.nq))
    delta = np.empty(model.nv)
    for frame in range(len(q) - 1):
        mujoco.mj_differentiatePos(model, delta, 1., q[frame], q[frame + 1])
        for substep in range(stride):
            pose = q[frame].copy()
            mujoco.mj_integratePos(model, pose, delta, substep / stride)
            poses[frame * stride + substep] = pose
    poses[-1] = q[-1]
    velocity = np.empty((len(poses), model.nv))
    for i in range(len(poses)):
        a, b = max(0, i - 1), min(len(poses) - 1, i + 1)
        mujoco.mj_differentiatePos(model, velocity[i], (b - a) * sim_dt, poses[a], poses[b])
    return poses, velocity


def prepare(output, names, sim_dt):
    output.mkdir(parents=True, exist_ok=True)
    model_path = build_model(output / 'model')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    controls = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    entries = {Path(r['file']).stem: r for r in manifest['motions']}
    records = []
    for name in names:
        folder = trial_dir(output, name)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, folder.parent / 'scene.xml')
        template, source, fps = load_reference(INPUT / (name + '.npz'), model)
        qpos, qvel = resample_reference(model, source, fps, sim_dt)
        np.savez_compressed(folder / 'trajectory_kinematic.npz', qpos=qpos, qvel=qvel,
                            ctrl=qpos[:, controls], contact=np.zeros((len(qpos), 0)),
                            contact_pos=np.zeros((len(qpos), 0, 3)))
        save_json(folder.parent / 'task_info.json', dict(ref_dt=sim_dt, contact_site_ids=[]))
        record = dict(motion=name, source_sha256=sha(INPUT / (name + '.npz')),
                      frames=len(source), fps=fps, simulation_frames=len(qpos),
                      duration_seconds=(len(source) - 1) / fps)
        if record['source_sha256'] != entries[name]['sha256']:
            raise ValueError(f'Source differs from its manifest: {name}')
        records.append(record)
    save_json(output / 'prepared.json', dict(motions=records, sim_dt=sim_dt,
              model_sha256=sha(model_path), source_manifest_sha256=sha(INPUT / 'manifest.json'),
              model='Project Booster T1 USD physics, including inertias, joint geometry, colliders and torque limits',
              upstream_commit=upstream_commit()))
    print(f'Prepared {len(records)} complete motions with our {model.body_mass.sum():.6f} kg Booster T1.', flush=True)


def export_training(path, model, qpos, qvel, template, fps):
    data = mujoco.MjData(model)
    bodies = [model.body(name).id for name in template['body_names'].tolist()]
    joints = [model.joint(name).id for name in template['joint_names'].tolist()]
    positions, rotations, linear, angular = [], [], [], []
    for q, v in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[bodies].copy())
        rotations.append(data.xquat[bodies].copy())
        vl, va = [], []
        for bid in bodies:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jp, jr, bid)
            vl.append(jp @ v)
            va.append(jr @ v)
        linear.append(vl)
        angular.append(va)
    np.savez_compressed(path, fps=np.array([fps]), joint_names=template['joint_names'],
                        body_names=template['body_names'], joint_pos=qpos[:, model.jnt_qposadr[joints]],
                        joint_vel=qvel[:, model.jnt_dofadr[joints]], body_pos_w=np.array(positions),
                        body_quat_w=np.array(rotations), body_lin_vel_w=np.array(linear),
                        body_ang_vel_w=np.array(angular),
                        preparation_note=np.array('SPIDER MuJoCo Warp rollout using project T1 physics; no quality cutoffs or postprocessing.'))


def run_one(args, name):
    import torch
    import warp as wp
    from spider.config import Config, process_config
    from spider.io import load_data
    from spider.optimizers.sampling import make_rollout_fn, make_optimize_once_fn, make_optimize_fn
    from spider.simulators import mjwp as sim

    folder = args.output / 'clips' / name
    folder.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if args.device == 'auto' else args.device
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    # Position servos are bounded by the effort limits in our exported model.
    # These are SPIDER control targets, not a FORCE2 policy/QP replay.
    config = process_config(Config(
        dataset_dir=str(args.output / 'dataset'), dataset_name='force2', robot_type='booster_t1',
        embodiment_type='humanoid', task=name, device=device, sim_dt=args.sim_dt,
        ref_dt=args.sim_dt, trace_dt=.02, render_dt=.02, ctrl_dt=.1, knot_dt=.1,
        horizon=args.horizon, num_samples=args.samples, max_num_iterations=args.iterations,
        first_ctrl_noise_scale=.5, last_ctrl_noise_scale=1., final_noise_scale=.1,
        joint_noise_scale=.1, temperature=.6, joint_rew_scale=1., pos_rew_scale=3.,
        rot_rew_scale=1., vel_rew_scale=0., contact_rew_scale=0.,
        sanity_check_seconds=0., terminate_resample=False,
        base_pos_threshold=float('inf'), base_rot_threshold=float('inf'),
        object_pos_threshold=float('inf'), object_rot_threshold=float('inf'),
        show_viewer=False, viewer='', wait_on_finish=False, save_video=False,
        use_torch_compile=not args.no_compile, num_trace_uniform_samples=0, num_trace_topk_samples=0,
    ))
    config.env_params_list = [[{}] for _ in range(args.iterations)]
    ref = load_data(config, config.data_path)
    raw = np.load(config.data_path)
    count = len(raw['qpos']) - 1
    if args.smoke_test:
        count = min(count, round(.04 / args.sim_dt))
    started = time.perf_counter()
    env = sim.setup_env(config, ref)
    # setup_env performs a CPU initialization step. Restore the exact starting
    # pose before generating the trajectory, so output time zero is source zero.
    data = mujoco.MjData(env.model_cpu)
    data.qpos[:], data.qvel[:], data.ctrl[:] = raw['qpos'][0], raw['qvel'][0], raw['ctrl'][0]
    mujoco.mj_forward(env.model_cpu, data)
    sim.sync_env_mujoco(config, env, data)
    rollout = make_rollout_fn(sim.step_env, sim.save_state, sim.load_state,
                             sim.get_reward, sim.get_terminal_reward, sim.get_terminate,
                             sim.get_trace, sim.save_env_params, sim.load_env_params, sim.copy_sample_state)
    optimize = make_optimize_fn(make_optimize_once_fn(rollout))
    ctrls = ref[2][:config.horizon_steps].clone()
    poses = [sim.get_qpos(config, env)[0].detach().cpu().numpy().copy()]
    velocities = [sim.get_qvel(config, env)[0].detach().cpu().numpy().copy()]
    controls, replans = [], []
    step = 0
    while step < count:
        tick = time.perf_counter()
        ref_slice = tuple(r[step + 1:step + config.horizon_steps + 1] for r in ref)
        ctrls, info = optimize(config, env, ctrls, ref_slice)
        take = min(config.ctrl_steps, count - step)
        for u in ctrls[:take]:
            sim.step_env(config, env, u)
            poses.append(sim.get_qpos(config, env)[0].detach().cpu().numpy().copy())
            velocities.append(sim.get_qvel(config, env)[0].detach().cpu().numpy().copy())
            controls.append(u.detach().cpu().numpy().copy())
        step += take
        replans.append(dict(step=step, seconds=time.perf_counter() - tick,
                            optimizer_iterations=int(info['opt_steps'][0])))
        ctrls = torch.cat((ctrls[take:], ref[2][step + config.horizon_steps - take:step + config.horizon_steps]))
        progress = dict(motion=name, simulated_seconds=step * args.sim_dt,
                        duration_seconds=count * args.sim_dt, wall_seconds=time.perf_counter() - started)
        save_json(folder / 'progress.json', progress)
        print(f'{name}: {step * args.sim_dt:.2f}/{count * args.sim_dt:.2f}s, wall {progress["wall_seconds"]:.1f}s', flush=True)
    wp.synchronize()
    qpos, qvel, ctrl = np.array(poses), np.array(velocities), np.array(controls)
    template, _, fps = load_reference(INPUT / (name + '.npz'), env.model_cpu)
    stride = round(1 / fps / args.sim_dt)
    target = raw['qpos'][:len(qpos)]
    complete = len(qpos) == len(raw['qpos']) and not args.smoke_test
    result = dict(motion=name, complete=complete, smoke_test=args.smoke_test,
                  frames=len(qpos[::stride]), source_frames=len(template['joint_pos']),
                  simulated_seconds=step * args.sim_dt, wall_seconds=time.perf_counter() - started,
                  finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
                  root_rmse_m=float(np.sqrt(np.mean(np.sum((qpos[:, :3] - target[:, :3]) ** 2, axis=1)))),
                  source_sha256=sha(INPUT / (name + '.npz')), model_sha256=sha(config.model_path),
                  device=device, gpu=torch.cuda.get_device_name(device),
                  settings=vars(args) | {'output': str(args.output)},
                  interpretation='Executed SPIDER controls on our T1 in MuJoCo Warp; quality is not used to stop generation.')
    np.savez_compressed(folder / 'rollout.npz', qpos=qpos, qvel=qvel, ctrl=ctrl,
                        reference_qpos=target, time=np.arange(len(qpos)) * args.sim_dt)
    export_training(folder / 'candidate.npz', env.model_cpu, qpos[::stride], qvel[::stride], template, fps)
    if complete:
        (args.output / 'motions').mkdir(exist_ok=True)
        shutil.copy2(folder / 'candidate.npz', args.output / 'motions' / (name + '.npz'))
    save_json(folder / 'replans.json', replans)
    save_json(folder / 'assessment.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--motions', nargs='+')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--horizon', type=float, default=.8)
    parser.add_argument('--sim-dt', type=float, default=.002)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Verify 0.04 seconds only; never exports a full training clip.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(ROOT / '.venvs/spider-cache/torch'))
    os.environ.setdefault('WARP_CACHE_PATH', str(ROOT / '.venvs/spider-cache/warp'))
    if args.worker:
        run_one(args, args.worker)
        return
    manifest = json.loads((INPUT / 'manifest.json').read_text())
    names = [Path(r['file']).stem for r in manifest['motions']]
    if args.motions:
        if not set(args.motions) <= set(names):
            parser.error('Unknown motion name')
        names = args.motions
    prepare(args.output, names, args.sim_dt)
    if args.prepare_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('SPIDER MuJoCo Warp requires an available CUDA GPU. Inputs are prepared; run on a GPU host.')
    print(f'GPU: {torch.cuda.get_device_name()}. {len(names)} clips; {args.samples} samples per clip. No quality cutoffs.', flush=True)
    results = []
    for name in names:
        folder = args.output / 'clips' / name
        folder.mkdir(parents=True, exist_ok=True)
        assessment = folder / 'assessment.json'
        if args.resume and assessment.exists():
            previous = json.loads(assessment.read_text())
            settings_match = all(previous.get('settings', {}).get(key) == getattr(args, key)
                                 for key in ('samples', 'iterations', 'horizon', 'sim_dt', 'no_compile'))
            inputs_match = (previous.get('source_sha256') == sha(INPUT / (name + '.npz'))
                            and previous.get('model_sha256') == sha(args.output / 'model/t1_force2.xml'))
            if previous.get('complete') and settings_match and inputs_match:
                results.append(previous)
                continue
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', name,
                   '--output', str(args.output), '--device', args.device, '--samples', str(args.samples),
                   '--iterations', str(args.iterations), '--horizon', str(args.horizon), '--sim-dt', str(args.sim_dt)]
        if args.no_compile:
            command.append('--no-compile')
        if args.smoke_test:
            command.append('--smoke-test')
        print(f'Starting {name}; progress log: {folder / "run.log"}', flush=True)
        with (folder / 'run.log').open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            result = dict(motion=name, complete=False, error=f'Process exit {code}; see run.log')
            save_json(assessment, result)
        else:
            result = json.loads(assessment.read_text())
        results.append(result)
        save_json(args.output / 'summary.json', dict(expected=len(names), attempted=len(results),
                  complete=sum(r['complete'] for r in results), motions=results))
        print(f'{name}: complete={result["complete"]}; {sum(r["complete"] for r in results)}/{len(names)} full clips.', flush=True)
    if any(r.get('error') for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
