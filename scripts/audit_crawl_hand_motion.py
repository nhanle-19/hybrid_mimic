"""Compare source hand markers, raw GMR hand motion, and geometric repair.

Kinematics cannot identify friction or prove a hand carries a pushing force.
The same material point is used for both robot trajectories; human markers
are differently sized anatomical proxies, not identical robot contact points.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d

from crawl_force_schedule import ROOT, digest


def main():
    source_path = ROOT/'ACCAD/Female1General_c3d/A11_-_crawl_forward_stageii.npz'
    raw_path = ROOT/'outputs/crawl_feasibility_20261022/female1_forward_raw.npz'
    repaired_path = ROOT/'retargeted_motion/a11_crawl_forward_female1_t1.npz'
    geometry_path = ROOT/'eval_data/reference_crawl_forward/contact_geometry.npz'
    output = ROOT/'eval_data/reference_crawl_forward/hand_motion'
    output.mkdir(parents=True, exist_ok=True)
    source = np.load(source_path, allow_pickle=True)
    raw, repaired = np.load(raw_path), np.load(repaired_path)
    geometry = np.load(geometry_path, allow_pickle=False)
    fps, source_fps = float(raw['fps']), float(source['mocap_frame_rate'])
    time = np.arange(len(raw['root_pos']))/fps
    source_time = np.arange(len(source['markers']))/source_fps
    # GMR samples evenly across the complete original frame range.
    aligned_source_time = np.linspace(source_time[0], source_time[-1], len(time))
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    data = mj.MjData(model)
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    rows, saved = {}, dict(time_s=time, aligned_source_time_s=aligned_source_time)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    for side, name in enumerate(('left_hand', 'right_hand')):
        region = geometry['surface_names'].tolist().index(name)
        members = geometry['point_surface_index'] == region
        local = geometry['point_position_body'][members]
        bodies = geometry['point_body_index'][members]
        assert len(np.unique(bodies)) == 1
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, str(geometry['body_names'][bodies[0]]))
        centroid = local.mean(0)  # A fixed material point; avoids lowest-vertex switches.
        positions, gaps = {}, {}
        for tag, motion in (('raw_gmr', raw), ('repaired', repaired)):
            order = [motion['joint_names'].tolist().index(n) for n in joint_names] if 'joint_names' in motion else list(range(23))
            pos, gap = [], []
            for i in range(len(time)):
                data.qpos[:] = np.r_[motion['root_pos'][i], motion['root_rot'][i, [3, 0, 1, 2]], motion['dof_pos'][i, order]]
                mj.mj_forward(model, data)
                rotation, translation = data.xmat[body].reshape(3, 3), data.xpos[body]
                pos.append(translation+rotation@centroid)
                gap.append((local@rotation.T+translation)[:, 2].min())
            positions[tag], gaps[tag] = np.asarray(pos), np.asarray(gap)
        marker_name = ('LFIN', 'RFIN')[side]
        marker = source['markers'][:, source['labels'].tolist().index(marker_name)].copy()
        valid = np.isfinite(marker).all(-1) & (np.linalg.norm(marker, axis=-1) > 0)
        for dim in range(3):
            marker[:, dim] = np.interp(source_time, source_time[valid], marker[valid, dim])
        marker = gaussian_filter1d(marker, 2., axis=0)  # 16.7 ms source smoothing.
        positions['source_marker'] = np.stack([np.interp(aligned_source_time, source_time, marker[:, d]) for d in range(3)], -1)
        speeds = {tag: np.linalg.norm(np.gradient(pos[:, :2], time, axis=0), axis=-1) for tag, pos in positions.items()}
        prefix = time < 6.
        slow = (speeds['source_marker'] < .03) & prefix
        ids = np.flatnonzero(slow)
        groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
        windows = []
        for g in groups:
            if len(g) < 2 or time[g[-1]]-time[g[0]] < .2:
                continue
            window = dict(start_s=float(time[g[0]]), end_s=float(time[g[-1]]))
            for tag, pos in positions.items():
                xy = pos[g, :2]
                window[tag+'_xy_excursion_m'] = float(np.linalg.norm(xy-xy[0], axis=-1).max())
            windows.append(window)
        rows[name] = dict(source_marker=marker_name, source_missing_frames=int((~valid).sum()),
            median_planar_speed_first6s_m_s={tag: float(np.median(v[prefix])) for tag, v in speeds.items()},
            source_slow_windows=windows,
            grounded_frames_within_1mm={tag: int(((g >= -.001) & (g <= .001)).sum()) for tag, g in gaps.items()},
            horizontal_repair_displacement_median_m=float(np.median(np.linalg.norm(positions['repaired'][:, :2]-positions['raw_gmr'][:, :2], axis=-1))))
        for tag, pos in positions.items():
            saved[name+'_'+tag+'_position_w'] = pos
            saved[name+'_'+tag+'_speed_m_s'] = speeds[tag]
            axes[0, side].plot(time[prefix], speeds[tag][prefix]*100, label=tag)
        for tag, gap in gaps.items():
            saved[name+'_'+tag+'_clearance_m'] = gap
            axes[1, side].plot(time[prefix], gap[prefix]*1000, label=tag)
        for tag, pos in positions.items():
            axes[2, side].plot(time[prefix], np.linalg.norm(pos[prefix, :2]-pos[0, :2], axis=-1)*100, label=tag)
        axes[3, side].plot(time[prefix], (positions['source_marker'][prefix, 2]-np.median(positions['source_marker'][:30, 2]))*1000)
        axes[0, side].set_title(name)
        for ax in axes[:, side]:
            ax.set_xlim(0, 6)
            ax.grid(alpha=.25)
        axes[0, side].legend(fontsize=8)
        axes[3, side].set_xlabel('Reference time (s)')
    axes[0, 0].set_ylabel('Planar speed (cm/s)')
    axes[1, 0].set_ylabel('Robot surface clearance (mm)')
    axes[2, 0].set_ylabel('XY displacement from initial point (cm)')
    axes[3, 0].set_ylabel('Source marker height change (mm)')
    fig.suptitle('Hand motion audit: human markers vs fixed robot material points\nDifferent anatomical points and body scales; stationary markers do not prove support force')
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(output/'kinematics.npz', **saved)
    report = dict(source=str(source_path.relative_to(ROOT)), frames=len(time), region_results=rows,
        source_alignment='GMR linspace over all original frames, displayed at saved robot FPS.',
        method='Human finger markers smoothed at sigma=2 source frames; robot hand mesh centroid fixed in its body. Velocities use centered finite differences.',
        limitations=['Different source/robot anatomical points and scales; compare raw GMR vs repaired robot directly.',
            'Marker height is not palm-floor clearance; stationary marker intervals do not prove load or a push.',
            'Friction coefficients cannot be inferred from kinematics alone.'],
        inputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in (source_path, raw_path, repaired_path, geometry_path)})
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
