"""Plot measured forces and render recorded PhysX states beside the reference."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    time = data['time_s']
    names = data['region_names'].tolist()
    measured_names = data['measured_region_names'].tolist()
    for region, ax in enumerate(axes.ravel()):
        measured = data['measured_forces'][:, measured_names.index(names[region]), 2]
        predicted = data['wrenches'][:, region, 2]
        ax.plot(time, predicted, label='QP prediction', linewidth=1.1)
        ax.plot(time+float(data['physics_dt']), measured, label='PhysX measurement', linewidth=.9, alpha=.8)
        high = max(10., float(max(measured.max(), predicted.max()))*1.05)
        ax.fill_between(time, 0, high, where=(data['activation'][:, region] <= 0) if 'activation' in data else ~data['contact_labels'][:, region], color='gray', alpha=.12, step='post')
        ax.set(title=names[region].replace('_', ' '), ylabel='Normal force (N)', ylim=(-2, high))
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel('Simulation time (s)')
    axes[-1, 1].set_xlabel('Simulation time (s)')
    fig.suptitle('Eight-region FORCE in PhysX | gray = '+('manual NN output decoded OFF' if 'activation' in data else 'reference label OFF'))
    fig.tight_layout()
    fig.savefig(output/'forces.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time, data['root_error']*1000)
    report = json.loads((output/'validation.json').read_text())
    axes[0].axhline(1000*report.get('root_error_limit_m', .15), color='r', linestyle='--', label='Stop threshold')
    axes[0].set_ylabel('Root error (mm)')
    axes[0].legend()
    axes[1].plot(time, np.rad2deg(data['joint_rmse']))
    axes[1].set_ylabel('Joint RMSE (deg)')
    axes[2].plot(time, data['measured_forces'][..., 2].sum(-1), label='All measured ground force')
    columns = [i for i, name in enumerate(measured_names) if name not in names]
    axes[2].plot(time, data['measured_forces'][:, columns, 2].sum(-1), label='Regions outside the 8-region QP')
    axes[2].set(ylabel='Normal force (N)', xlabel='Simulation time (s)')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'tracking.png', dpi=150)
    plt.close(fig)


def render(data, output):
    report = json.loads((output/'validation.json').read_text())
    motion_path = Path(report.get('motion_file', ROOT/'retargeted_motion/a11_crawl_forward_female1_t1.npz'))
    if hashlib.sha256(motion_path.read_bytes()).hexdigest() != report['motion_sha256']:
        raise ValueError('The current reference differs from the one used in this rollout')
    motion = np.load(motion_path)
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    sim_data = mj.MjData(model)
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    physics_order = [data['joint_names'].tolist().index(name) for name in names]
    reference_order = [motion['joint_names'].tolist().index(name) for name in names]
    actual = data['trajectory_q']
    reference = np.c_[motion['root_pos'], motion['root_rot'], motion['dof_pos'][:, reference_order]]
    width, height = 720, 480
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.55, 130, -14
    fps = float(motion['fps'])
    count = int(np.floor(data['trajectory_time_s'][-1]*fps))+1
    with mj.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
        str(output/'video.mp4'), fps=fps, codec='libx264', quality=8, macro_block_size=2,
        ffmpeg_params=['-movflags', '+faststart']) as writer:
        for frame in range(count):
            time_s = frame/fps
            index = min(np.searchsorted(data['trajectory_time_s'], time_s), len(actual)-1)
            current = actual[index]
            reference_frame = int(np.floor(time_s*fps*float(data.get('reference_speed', 1.))+1e-9))
            if report.get('hold_frame') is not None:
                reference_frame = report['hold_frame']
            ref = reference[min(reference_frame, len(reference)-1)]
            panels = []
            for label, q in [('Reference', ref), ('PhysX FORCE: recorded actual motion',
                    np.r_[current[:7], current[7:][physics_order]])]:
                sim_data.qpos[:] = np.r_[q[:3], q[[6, 3, 4, 5]], q[7:]]
                mj.mj_forward(model, sim_data)
                camera.lookat[:] = ref[:3]
                camera.lookat[2] = .2
                renderer.update_scene(sim_data, camera, scene_option=option)
                image = renderer.render().copy()
                cv2.rectangle(image, (0, 0), (width, 42), (250, 250, 250), -1)
                cv2.putText(image, label, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, .62, (25, 35, 45), 1, cv2.LINE_AA)
                cv2.putText(image, f'{time_s:.3f} s', (15, height-18), cv2.FONT_HERSHEY_SIMPLEX, .62, (25, 35, 45), 1, cv2.LINE_AA)
                panels.append(image)
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame == count-1:
                cv2.imwrite(str(output/'preview.png'), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    with imageio.get_reader(str(output/'video.mp4')) as reader:
        assert sum(1 for _ in reader) == count
    (output/'video_manifest.json').write_text(json.dumps(dict(file='video.mp4', frames=count, fps=fps,
        source='Recorded PhysX states rendered beside the reference; no physics performed by renderer.'), indent=2)+'\n')
    print('Rendered', count, 'recorded physics frames to', output/'video.mp4', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    args = parser.parse_args()
    with np.load(args.diagnostics, allow_pickle=False) as archive:
        data = dict(archive)
    plot(data, args.diagnostics.parent)
    render(data, args.diagnostics.parent)
