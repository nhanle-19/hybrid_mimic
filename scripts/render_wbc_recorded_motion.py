"""Render saved actual/reference WBC states, including explicitly partial runs.

MuJoCo only draws poses recorded by PhysX; no dynamics are advanced here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(diagnostics, planned_schedule, output):
    data = dict(np.load(diagnostics, allow_pickle=False))
    planned = dict(np.load(planned_schedule, allow_pickle=False))
    actual, reference = data['recorded_actual_q'], data['recorded_reference_q']
    time = data['recorded_motion_time_s']
    frames = data['manual_schedule_frames']
    if len(time) < 2 or actual.shape != reference.shape or len(actual) != len(time):
        raise ValueError('Missing or inconsistent recorded motion states')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Recorded states must be finite')
    if bool(data['manual_actual_contact_gate']):
        raise ValueError('This renderer expects the unmodified manual-output replay')
    np.testing.assert_array_equal(data['manual_schedule_actions'], planned['actions'][frames].astype(np.float32))
    dt = float(np.median(np.diff(time)))
    np.testing.assert_allclose(np.diff(time), dt, atol=1e-8, rtol=0)
    fps = 1/dt
    count, total = len(actual), len(planned['actions'])
    complete = bool(data['complete_motion'])
    if complete and count != total:
        raise ValueError('Complete-run flag disagrees with recorded frame count')
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    sim_data = mj.MjData(model)
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    recorded_names = data['joint_names'].tolist()
    order = [recorded_names.index(name) for name in names]
    width, height = 640, 480
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.1, 140, -17
    output.mkdir(parents=True, exist_ok=True)
    video = output/'partial_cpu_video.mp4' if not complete else output/'video.mp4'
    preview_name = 'preview.png' if complete else 'partial_cpu_preview.png'
    review_name = 'review_frames.png' if complete else 'partial_cpu_review_frames.png'
    manifest_name = 'video_manifest.json' if complete else 'partial_cpu_video_manifest.json'
    selected = set(np.linspace(0, count-1, 4).astype(int).tolist())
    previews = []
    with mj.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
        str(video), fps=fps, codec='libx264', quality=8, macro_block_size=2,
        ffmpeg_params=['-movflags', '+faststart']) as writer:
        for index in range(count):
            panels = []
            for title, q in [('Short re-anchor reference', reference[index]),
                             ('Corrected FORCE: recorded CPU motion', actual[index])]:
                sim_data.qpos[:] = np.r_[q[:3], q[[6,3,4,5]], q[7:][order]]
                mj.mj_forward(model, sim_data)
                camera.lookat[:] = [reference[index,0], reference[index,1], .45]
                renderer.update_scene(sim_data, camera, scene_option=option)
                pixels = renderer.render().copy()
                cv2.rectangle(pixels, (0,0), (width,42), (250,250,250), -1)
                cv2.putText(pixels, title, (12,28), cv2.FONT_HERSHEY_SIMPLEX, .54, (20,30,40), 1, cv2.LINE_AA)
                cv2.rectangle(pixels, (0,height-64), (width,height), (250,250,250), -1)
                cv2.putText(pixels, f't = {time[index]:.2f} s | frame {frames[index]+1}/{total}',
                    (12,height-39), cv2.FONT_HERSHEY_SIMPLEX, .52, (20,30,40), 1, cv2.LINE_AA)
                status = 'Complete recording' if complete else f'PARTIAL {count}/{total}: stopped before re-anchor landing'
                cv2.putText(pixels, status, (12,height-14), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (170,35,25) if not complete else (20,30,40), 1, cv2.LINE_AA)
                panels.append(pixels)
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if index in selected:
                previews.append(cv2.resize(combined, (960,360)))
            if index == count-1:
                cv2.imwrite(str(output/preview_name), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/review_name), cv2.cvtColor(np.concatenate(previews, axis=0), cv2.COLOR_RGB2BGR))
    with imageio.get_reader(str(video)) as reader:
        decoded = sum(1 for _ in reader)
    assert decoded == count
    manifest = dict(file=video.name, frames=count, requested_frames=total, fps=fps,
        duration_s=count/fps, final_recorded_time_s=float(time[-1]), full_motion_completed=complete,
        failure_reason=str(data['failure_reason']), source=str(diagnostics.resolve()),
        source_sha256=hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        planned_schedule=str(planned_schedule.resolve()), manual_outputs_match=True,
        actual_and_reference_source=str(data['recorded_motion_source']),
        render_method='MuJoCo visualization of saved PhysX states; no physics or trajectory interpolation.',
        scope='Partial CPU replay does not reach the corrected landing.' if not complete else 'Complete recorded rollout.')
    (output/manifest_name).write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--planned-schedule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.diagnostics, args.planned_schedule, args.output)
