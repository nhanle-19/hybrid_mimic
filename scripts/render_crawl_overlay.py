"""Overlay the recorded crawl on a translucent orange reference in world space.

Example (use an environment with MuJoCo, OpenCV and imageio installed)::

    MUJOCO_GL=osmesa python scripts/render_crawl_overlay.py \
        eval_data/reference_crawl_forward/force/diagnostics.npz \
        --output eval_data/reference_crawl_forward/force/overlay

MuJoCo draws saved poses only; this script never advances physics.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RGBA = (1.0, 0.34, 0.025, 0.32)
ACTUAL_RGBA = (0.28, 0.34, 0.40, 1.0)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_pose(model, state, q):
    # Archives store xyzw; MuJoCo's free joint takes wxyz.
    state.qpos[:] = np.r_[q[:3], q[[6, 3, 4, 5]], q[7:]]
    mj.mj_forward(model, state)


def tint(geom, rgba):
    geom.matid = -1
    geom.rgba[:] = rgba
    geom.specular = 0.18
    geom.shininess = 0.25
    geom.reflectance = 0.0


def render(diagnostics, output, preview_only=False):
    with np.load(diagnostics, allow_pickle=False) as archive:
        data = dict(archive)
    report_path = diagnostics.with_name('validation.json')
    report = json.loads(report_path.read_text())
    motion_path = Path(report['motion_file'])
    if sha256(motion_path) != report['motion_sha256']:
        raise ValueError('Reference does not match the recorded rollout')
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in
                  ('root_pos', 'root_rot', 'dof_pos', 'joint_names', 'fps')}
    model_path = ROOT/'GMR/assets/booster_t1/T1_serial.xml'
    model = mj.MjModel.from_xml_path(str(model_path))
    actual_state, reference_state = mj.MjData(model), mj.MjData(model)
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
             for j in range(1, model.njnt)]
    actual_order = [data['joint_names'].tolist().index(name) for name in names]
    reference_order = [motion['joint_names'].tolist().index(name) for name in names]
    actual = np.c_[data['trajectory_q'][:, :7],
                   data['trajectory_q'][:, 7:][:, actual_order]]
    reference = np.c_[motion['root_pos'], motion['root_rot'],
                      motion['dof_pos'][:, reference_order]]
    recorded_times = data['trajectory_time_s']
    if (len(actual) != len(recorded_times) or len(actual) < 2
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()
            or not np.isfinite(recorded_times).all()
            or not np.all(np.diff(recorded_times) > 0)):
        raise ValueError('Invalid recorded poses or timestamps')
    fps = float(motion['fps'])
    count = int(np.floor(recorded_times[-1]*fps + 1e-8)) + 1
    times = np.arange(count)/fps
    # Choose the nearest saved physics state; do not interpolate or re-align poses.
    right = np.minimum(np.searchsorted(recorded_times, times), len(actual)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(abs(recorded_times[left]-times) <= abs(recorded_times[right]-times), left, right)
    offsets = abs(recorded_times[indices]-times)
    if np.max(offsets) > float(data['physics_dt'])/2 + 1e-8:
        raise ValueError('Video timing is not covered by the saved physics states')
    reference_indices = np.minimum(
        np.floor(times*fps*float(data.get('reference_speed', 1.0)) + 1e-8).astype(int),
        len(reference)-1)
    if report.get('hold_frame') is not None:
        reference_indices[:] = report['hold_frame']
    width, height = 1280, 720
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = 0.3
    model.vis.headlight.diffuse[:] = 0.65
    model.vis.headlight.specular[:] = 0.3
    option, perturb = mj.MjvOption(), mj.MjvPerturb()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 130, -24
    selected = sorted(set(np.linspace(0, count-1, 6).astype(int).tolist()))
    output.mkdir(parents=True, exist_ok=True)
    video = output/'video.mp4'
    samples = []
    writer = None
    try:
        if not preview_only:
            writer = imageio.get_writer(str(video), fps=fps, codec='libx264',
                quality=8, macro_block_size=2,
                input_params=['-r', str(Fraction(fps).limit_denominator(1000000))],
                ffmpeg_params=['-threads', '2', '-movflags', '+faststart'])
        with mj.Renderer(model, height=height, width=width) as renderer:
            for frame in selected if preview_only else range(count):
                q, ref = actual[indices[frame]], reference[reference_indices[frame]]
                set_pose(model, actual_state, q)
                set_pose(model, reference_state, ref)
                camera.lookat[:] = (q[:3]+ref[:3])/2
                camera.lookat[2] = 0.18
                renderer.update_scene(actual_state, camera, scene_option=option)
                scene = renderer.scene
                for geom in scene.geoms[:scene.ngeom]:
                    if geom.category == mj.mjtCatBit.mjCAT_DYNAMIC:
                        tint(geom, ACTUAL_RGBA)
                actual_pixels = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                actual_mask = ((segmentation[..., 1] == mj.mjtObj.mjOBJ_GEOM)
                               & (segmentation[..., 0] >= 0))
                actual_mask &= model.geom_bodyid[np.maximum(segmentation[..., 0], 0)] > 0
                first_reference_geom = scene.ngeom
                mj.mjv_addGeoms(model, reference_state, option, perturb,
                               mj.mjtCatBit.mjCAT_DYNAMIC, scene)
                for geom in scene.geoms[first_reference_geom:scene.ngeom]:
                    tint(geom, REFERENCE_RGBA)
                pixels = renderer.render().copy()
                # Put the solid actual silhouette on top of the ghost. This also
                # avoids flickering from coincident meshes when tracking is exact.
                pixels[actual_mask] = actual_pixels[actual_mask]
                pixels[:90] = (249, 250, 252)
                pixels[-52:] = (249, 250, 252)
                cv2.putText(pixels, 'CRAWL  /  REFERENCE + ACTUAL', (28, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, .78, (28, 36, 45), 2, cv2.LINE_AA)
                cv2.circle(pixels, (37, 66), 7, (245, 111, 20), -1, cv2.LINE_AA)
                cv2.putText(pixels, 'Reference: translucent orange', (55, 73),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (160, 74, 16), 1, cv2.LINE_AA)
                cv2.circle(pixels, (512, 66), 7, (71, 87, 102), -1, cv2.LINE_AA)
                cv2.putText(pixels, 'Actual: solid / recorded PhysX motion', (530, 73),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, (50, 64, 78), 1, cv2.LINE_AA)
                cv2.putText(pixels, f'{times[frame]:05.2f} s   |   frame {frame+1}/{count}',
                            (28, height-20), cv2.FONT_HERSHEY_SIMPLEX, .58, (45, 55, 65), 1, cv2.LINE_AA)
                error = np.linalg.norm(q[:3]-ref[:3])
                cv2.putText(pixels, f'Root separation: {error*100:.1f} cm',
                            (width-310, height-20), cv2.FONT_HERSHEY_SIMPLEX, .58,
                            (45, 55, 65), 1, cv2.LINE_AA)
                if writer is not None:
                    writer.append_data(pixels)
                if frame in selected:
                    samples.append(cv2.resize(pixels, (640, 360), interpolation=cv2.INTER_AREA))
                    if frame == selected[len(selected)//2]:
                        cv2.imwrite(str(output/'preview.png'), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                if frame % 100 == 0:
                    print(f'Rendered {frame+1}/{count}', flush=True)
    finally:
        if writer is not None:
            writer.close()
    sheet = np.concatenate([np.concatenate(samples[i:i+2], axis=1)
                            for i in range(0, len(samples), 2)], axis=0)
    cv2.imwrite(str(output/'review_frames.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    if preview_only:
        print(f'Preview: {output / "review_frames.png"}', flush=True)
        return
    manifest = dict(file=video.name, frames=count,
        fps=fps, duration_s=count/fps, final_recorded_time_s=float(recorded_times[-1]),
        resolution=[width, height], reference_rgba=REFERENCE_RGBA, actual_rgba=ACTUAL_RGBA,
        actual_source=str(diagnostics.resolve()), actual_source_sha256=sha256(diagnostics),
        reference_source=str(motion_path.resolve()), reference_sha256=sha256(motion_path),
        validation_source=str(report_path.resolve()), validation_sha256=sha256(report_path),
        model_source=str(model_path), model_xml_sha256=sha256(model_path),
        renderer_source_sha256=sha256(Path(__file__)), output_sha256=sha256(video),
        full_motion_completed=bool(report['full_motion_completed']),
        max_recorded_time_offset_s=float(offsets.max()),
        actual_sample_indices=indices.tolist(), reference_frame_indices=reference_indices.tolist(),
        reviewed_frame_indices=selected, simulation_rerun=False, pose_interpolation=False,
        camera=dict(distance=camera.distance, azimuth=camera.azimuth,
                    elevation=camera.elevation, follows='midpoint of actual and reference roots'),
        method='Shared 3D scene with a translucent orange reference; the segmented solid actual silhouette is composited on top.',
        alignment='Original world positions and rotations; no per-frame root alignment or spatial offset.')
    finalize_video(output, manifest)


def finalize_video(output, manifest):
    """Decode the whole export and write its provenance and playback page."""
    video = output/manifest['file']
    width, height = manifest['resolution']
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, pixels = cap.read()
        if not ok:
            break
        if pixels.shape != (height, width, 3):
            raise ValueError('Unexpected decoded video dimensions')
        decoded += 1
    cap.release()
    if decoded != manifest['frames'] or not np.isclose(output_fps, manifest['fps'], rtol=1e-8):
        raise ValueError('Decoded video count or timing does not match the recording')
    manifest.update(decoded_frames=decoded, fps=output_fps,
                    duration_s=decoded/output_fps, output_sha256=sha256(video))
    (output/'video_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'preview.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Crawl motion overlay</title><style>
body{max-width:1280px;margin:32px auto;padding:0 20px;font:17px system-ui;background:#f9fafc;color:#23303e}
video{width:100%;border-radius:8px}a{color:#a74d0e}</style>
<h1>Crawl motion overlay</h1><p>Translucent orange: reference. Solid slate: recorded actual motion.</p>
<video controls loop preload="metadata" poster="preview.png" src="video.mp4"></video>
<p>Both motions retain their original world positions and timing.</p>
<p><a href="video.mp4">Download video</a> &middot; <a href="review_frames.png">Review frames</a></p></html>\n''')
    print(f'Verified {decoded} frames at {output_fps:.6f} fps: {video}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    render(args.diagnostics, args.output, args.preview_only)
