"""Build the selected contact-motion NPZs in staging, then install with a backup.

Run retarget in the GMR/rl_lab environment and assemble in hybridmimic.
Neither stage starts training. Install is explicit and follows full validation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ('fps', 'joint_names', 'body_names', 'joint_pos', 'joint_vel', 'body_pos_w',
        'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retarget(config, work):
    import torch
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

    torch.set_num_threads(4)
    for row in config['sources']:
        source = ROOT / row['source']
        if sha(source) != row['source_sha256']:
            raise ValueError(f'Source changed: {source}')
        if row.get('existing_reference'):
            continue
        dest = work / 'raw' / (row['name']+'.npz')
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                if str(z['selection_json'].item()) != json.dumps(row, sort_keys=True):
                    raise ValueError(f'Stale cache: {dest}')
            print(f'Reuse {row["name"]}', flush=True)
            continue
        with np.load(source, allow_pickle=False) as z:
            fps = float(z['mocap_frame_rate'])
            n = len(z['trans'])
            begin, end = row.get('crop_seconds', [0, n/fps])
            start, stop = round(begin*fps), min(n, round(end*fps))
            stride = round(fps/config['retarget_fps'])
            if abs(fps/stride-config['retarget_fps']) > 1e-9:
                raise ValueError('Source fps must divide evenly for exact temporal sampling')
            ids = np.arange(start, stop, stride)
            sampled = {key: z[key][ids] for key in ('trans', 'root_orient', 'pose_body')}
            sampled.update(betas=z['betas'], gender=z['gender'], mocap_frame_rate=np.asarray(fps/stride))
        sample = work / 'sampled' / (row['name']+'.npz')
        sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample, **sampled)
        print(f'Retarget {row["name"]}: {len(ids)} frames', flush=True)
        with torch.no_grad():
            data, model, output, height = load_smplx_file(str(sample), str(ROOT/'GMR/assets/body_models'))
            frames, aligned = get_smplx_data_offline_fast(data, model, output, tgt_fps=config['retarget_fps'])
        retargeter = GeneralMotionRetargeting(src_human='smplx', tgt_robot='booster_t1',
                                             actual_human_height=height, verbose=False)
        # Converge the first pose before saving to avoid an IK initialization jump.
        for _ in range(15):
            retargeter.retarget(frames[0])
        q = np.asarray([retargeter.retarget(frame) for frame in frames])
        names = [mujoco.mj_id2name(retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(retargeter.model.njnt) if retargeter.model.jnt_qposadr[j] >= 7]
        assert q.shape == (len(ids), 30) and np.isfinite(q).all()
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, q=q, joint_names=np.asarray(names), fps=aligned,
                            source_time_s=ids/fps, source_crop_frames=np.asarray([start, stop]),
                            selection_json=json.dumps(row, sort_keys=True))
        print(f'Saved {dest.name}', flush=True)
        del model, output, data, frames, retargeter


def assemble(config, work):
    import torch
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation, Slerp
    sys.path.insert(0, str(ROOT/'source/whole_body_tracking/whole_body_tracking/utils'))
    from wbc_reference import prepare_wbc_reference
    from mirror_training_motions import generate

    torch.set_num_threads(4)
    active = ROOT/'retargeted_motion/training_list'
    stage = Path(tempfile.mkdtemp(prefix='staging_build_', dir=work))
    old = json.loads((active/'manifest.json').read_text())
    template = dict(np.load(active/'01_walk.npz', allow_pickle=False))
    joint_names, body_names = template['joint_names'].tolist(), template['body_names'].tolist()
    geometry = dict(np.load(ROOT/'eval_data/reference_walk_20261016/hdm_bk_settled_env_geometry.npz', allow_pickle=False))
    order = [geometry['joint_names'].tolist().index(name) for name in joint_names]
    limits, speeds = geometry['joint_limits'][order], geometry['velocity_limits'][order]
    records, reports = [], {}
    for stem in config['retained_originals']:
        record = dict(next(r for r in old['motions'] if r['file'] == stem+'.npz'))
        if sha(active/record['file']) != record['sha256']:
            raise ValueError(f'Active source changed: {stem}')
        if stem == '09_front_kick':
            with np.load(active/record['file'], allow_pickle=False) as z:
                values = dict(z)
            root = body_names.index('Trunk')
            corrected, prep = prepare_wbc_reference(values['joint_pos'], values['body_pos_w'][:, root],
                values['body_quat_w'][:, root], joint_names, body_names, limits, config['output_fps'])
            values.update(corrected)
            np.savez_compressed(stage/record['file'], **values)
            record['unprepared_collection_source'] = record.pop('collection_source')
            record['unprepared_sha256'] = record['sha256']
            record['preparation'] = prep
            record['sha256'] = sha(stage/record['file'])
        else:
            shutil.copyfile(active/record['file'], stage/record['file'])
        records.append(record)
    for row in config['sources']:
        fps = config['output_fps']
        if row.get('existing_reference'):
            source = ROOT/row['existing_reference']
            with np.load(source, allow_pickle=False) as z:
                values = {key: z[key] for key in CORE if key in z}
                # Older converters omitted names; their native order matches the template.
                values.setdefault('joint_names', np.asarray(joint_names))
                values.setdefault('body_names', np.asarray(body_names))
            assert values['joint_names'].tolist() == joint_names and values['body_names'].tolist() == body_names
            assert float(values['fps'].item()) == fps
            root = body_names.index('Trunk')
            corrected, prep = prepare_wbc_reference(values['joint_pos'], values['body_pos_w'][:, root],
                values['body_quat_w'][:, root], joint_names, body_names, limits, fps)
            values.update(corrected)
            report = dict(method='Existing T1 reference with standard WBC limit/floor preparation',
                          source=str(source.relative_to(ROOT)), source_sha256=sha(source), timing_scale=1.,
                          preparation=prep)
        else:
            source = work/'raw'/(row['name']+'.npz')
            raw = dict(np.load(source, allow_pickle=False))
            q, source_fps = raw['q'], float(raw['fps'])
            times = np.arange(len(q))/source_fps
            # Mild smoothing before resampling; no foot leveling or contact locks.
            pos = gaussian_filter1d(q[:, :3], .03*source_fps, axis=0, mode='nearest')
            joints = gaussian_filter1d(q[:, 7:], .03*source_fps, axis=0, mode='nearest')
            quat = q[:, [4, 5, 6, 3]].copy()  # GMR root WXYZ to scipy XYZW.
            for i in range(1, len(quat)):
                if quat[i] @ quat[i-1] < 0:
                    quat[i] *= -1
            quat = gaussian_filter1d(quat, .03*source_fps, axis=0, mode='nearest')
            rot = Rotation.from_quat(quat)
            mapping = [raw['joint_names'].tolist().index(name) for name in joint_names]
            joints = joints[:, mapping]
            rate_ratio = float((np.abs(np.gradient(joints, 1/source_fps, axis=0))/speeds).max())
            timing_scale = max(1., rate_ratio/.9)
            sample_times = np.arange(0, times[-1]+1e-9, 1/fps/timing_scale)
            jp = np.column_stack([np.interp(sample_times, times, joints[:, j]) for j in range(23)])
            rp = np.column_stack([np.interp(sample_times, times, pos[:, j]) for j in range(3)])
            rq = Slerp(times, rot)(sample_times).as_quat()[:, [3, 0, 1, 2]]
            values, prep = prepare_wbc_reference(jp, rp, rq, joint_names, body_names, limits, fps)
            values.update(fps=np.asarray([float(fps)]), joint_names=np.asarray(joint_names), body_names=np.asarray(body_names))
            values['source_time_s'] = raw['source_time_s'][0]+sample_times
            values['source_crop_frames'] = raw['source_crop_frames']
            report = dict(method='GMR at 30 Hz; 30 ms Gaussian smoothing; interpolate to 50 Hz; standard WBC limit/floor preparation',
                          timing_scale=timing_scale, raw_retarget_sha256=sha(source), preparation=prep)
        values.update(source_motion=np.asarray(row['source']), source_motion_sha256=np.asarray(row['source_sha256']))
        report['max_joint_speed_limit_fraction'] = float((np.abs(values['joint_vel'])/speeds).max())
        path = stage/(row['name']+'.npz')
        np.savez_compressed(path, **values)
        record = dict(file=path.name, source=row['source'], source_sha256=row['source_sha256'],
                      sha256=sha(path), frames=len(values['joint_pos']), fps=float(fps),
                      family=row['family'], contact_focus=row['contact_focus'], preparation=report)
        if row.get('crop_seconds'):
            record['source_crop_seconds'] = row['crop_seconds']
        records.append(record)
        reports[path.name] = report
        print(f'Prepared {path.name}: {len(values["joint_pos"])} frames', flush=True)
    manifest = dict(motions=sorted(records, key=lambda r:r['file']),
                    total_frames=sum(r['frames'] for r in records),
                    sampling='equal probability per episode reset',
                    purpose='Contact-focused references selected for FORCE2; every original has a sagittal mirror, including walking.',
                    note='Reference motions only; no manual force labels. New motions are kinematically prepared, not demonstrated policy successes.',
                    selection_config='configs/contact_training_sources.json',
                    selection_config_sha256=sha(ROOT/'configs/contact_training_sources.json'),
                    removed=['04_narrow_sway.npz', '04_narrow_sway_mirrored.npz'])
    (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = generate(stage)
    (work/'preparation.json').write_text(json.dumps(reports, indent=2)+'\n')
    if (work/'staging').exists():
        (work/'staging').rename(work/f'staging_previous_{time.time_ns()}')
    stage.rename(work/'staging')
    print(f'Staged {len(manifest["motions"])} clips / {manifest["total_frames"]} frames', flush=True)


def install(work):
    active, stage, backup = ROOT/'retargeted_motion/training_list', work/'staging', work/'previous_training_list'
    validation = json.loads((work/'validation.json').read_text())
    if not validation['passed'] or validation['manifest_sha256'] != sha(stage/'manifest.json'):
        raise ValueError('Staged selection needs current passing validation')
    manifest = json.loads((stage/'manifest.json').read_text())
    for row in manifest['motions']:
        if sha(stage/row['file']) != row['sha256']:
            raise ValueError(f'Staged file changed: {row["file"]}')
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    # Same-filesystem renames preserve the complete previous folder for rollback.
    active.rename(backup)
    try:
        stage.rename(active)
    except BaseException:
        backup.rename(active)
        raise
    print(f'Installed {len(manifest["motions"])} clips in {active}; backup: {backup}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['retarget', 'assemble', 'install'])
    parser.add_argument('--work-dir', type=Path, default=ROOT/'outputs/contact_training_build_20261115')
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/contact_training_sources.json').read_text())
    if args.stage == 'install':
        install(args.work_dir)
    else:
        globals()[args.stage](config, args.work_dir)


if __name__ == '__main__':
    main()
