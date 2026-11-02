"""Render a sequential learned-policy evaluation with the orange reference overlay.

Uses recorded PhysX poses in their original world frame; never reruns dynamics.
Example: MUJOCO_GL=egl python scripts/render_force_suite_overlay.py \
    outputs/force_full_contact_eval_20261101 --output outputs/force_full_contact_eval_20261101/overlay
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RGBA = (1., .34, .025, .32)
ACTUAL_RGBA = (.28, .34, .40, 1.)
WIDTH, HEIGHT = 1280, 720
BACKGROUND, TEXT, ORANGE = (249, 250, 252), (35, 48, 62), (224, 93, 14)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def font(size, bold=False):
    name = 'FreeSansBold.ttf' if bold else 'FreeSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/freefont/'+name, size)


FONTS = {size: font(size) for size in (20, 22, 24)}
FONTS[32], FONTS[52] = font(32, True), font(52, True)


def tint(geom, rgba):
    geom.matid = -1
    geom.rgba[:] = rgba
    geom.specular, geom.shininess, geom.reflectance = .18, .25, 0.


def set_pose(model, state, q):
    state.qpos[:] = np.r_[q[:3], q[[6, 3, 4, 5]], q[7:]]
    mj.mj_forward(model, state)


def load_clips(directory, model):
    sources = sorted(directory.glob('*/diagnostics.npz'))
    if not sources:
        raise ValueError('No evaluation rollouts found')
    manifest = json.loads((ROOT/'retargeted_motion/training_list/manifest.json').read_text())
    expected = {Path(m['file']).stem for m in manifest['motions']}
    if {p.parent.name for p in sources} != expected:
        raise ValueError('Evaluation must cover every active training-list motion')
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    clips = []
    for path in sources:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata_json']))
            order = [archive['joint_names'].tolist().index(name) for name in names]
            actual, reference = archive['audit_actual_q'], archive['audit_reference_q']
            if actual.shape != reference.shape or len(actual) != meta['attempted_frames']:
                raise ValueError(f'{path}: inconsistent pose counts')
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                raise ValueError(f'{path}: nonfinite pose')
            np.testing.assert_array_equal(archive['audit_frame'], np.arange(len(actual)))
            clips.append(dict(name=path.parent.name, title=path.parent.name[3:].replace('_', ' ').title(),
                source=path, metadata=meta,
                checkpoint_label=Path(meta['checkpoint']).stem.removeprefix('model_'),
                actual=np.c_[actual[:, :7], actual[:, 7:][:, order]],
                reference=np.c_[reference[:, :7], reference[:, 7:][:, order]]))
    checkpoints = {c['metadata']['checkpoint_sha256'] for c in clips}
    if len(checkpoints) != 1 or sha256(clips[0]['metadata']['checkpoint']) not in checkpoints:
        raise ValueError('Rollouts must come from the same unchanged checkpoint')
    fps = 1./clips[0]['metadata']['policy_dt_s']
    for clip in clips:
        meta = clip['metadata']
        if not np.isclose(1./meta['policy_dt_s'], fps) or not meta['deterministic']:
            raise ValueError('Expected deterministic rollouts with a shared frame rate')
        if sha256(meta['motion_file']) != meta['motion_sha256']:
            raise ValueError('Reference file changed since evaluation')
    return clips, fps


def overlay_frame(renderer, model, states, option, perturb, camera, clip, index, task, total):
    actual, reference = clip['actual'][index], clip['reference'][index]
    set_pose(model, states[0], actual)
    set_pose(model, states[1], reference)
    camera.lookat[:] = (actual[:3]+reference[:3])/2
    camera.lookat[2] = .58
    renderer.update_scene(states[0], camera, scene_option=option)
    scene = renderer.scene
    for geom in scene.geoms[:scene.ngeom]:
        if geom.category == mj.mjtCatBit.mjCAT_DYNAMIC:
            tint(geom, ACTUAL_RGBA)
    actual_pixels = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    mask = (segmentation[..., 1] == mj.mjtObj.mjOBJ_GEOM) & (segmentation[..., 0] >= 0)
    mask &= model.geom_bodyid[np.maximum(segmentation[..., 0], 0)] > 0
    first = scene.ngeom
    mj.mjv_addGeoms(model, states[1], option, perturb, mj.mjtCatBit.mjCAT_DYNAMIC, scene)
    for geom in scene.geoms[first:scene.ngeom]:
        tint(geom, REFERENCE_RGBA)
    pixels = renderer.render().copy()
    # Match render_crawl_overlay.py: preserve the solid actual silhouette where
    # meshes coincide. No spatial offset or per-frame pose alignment is applied.
    pixels[mask] = actual_pixels[mask]
    panel = Image.fromarray(pixels)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, WIDTH, 90), fill=BACKGROUND)
    draw.text((28, 14), f'{task:02d} / {total:02d}   {clip["title"].upper()}', font=FONTS[32], fill=TEXT)
    draw.text((WIDTH-260, 23), f'CHECKPOINT {clip["checkpoint_label"]}', font=FONTS[20], fill=TEXT)
    draw.ellipse((29, 62, 43, 76), fill=ORANGE)
    draw.text((55, 55), 'Reference: translucent orange', font=FONTS[22], fill=(160, 74, 16))
    draw.ellipse((479, 62, 493, 76), fill=(71, 87, 102))
    draw.text((505, 55), 'Learned policy: solid slate', font=FONTS[22], fill=TEXT)
    meta = clip['metadata']
    draw.rectangle((0, HEIGHT-58, WIDTH, HEIGHT), fill=BACKGROUND)
    draw.text((28, HEIGHT-41), f'{index*meta["policy_dt_s"]:05.2f} s  /  {meta["expected_frames"]*meta["policy_dt_s"]:.2f} s',
              font=FONTS[22], fill=TEXT)
    draw.text((435, HEIGHT-41), f'Frame {index+1} / {meta["expected_frames"]}   |   1x speed', font=FONTS[22], fill=TEXT)
    error = np.linalg.norm(actual[:3]-reference[:3])
    draw.text((WIDTH-290, HEIGHT-41), f'Root separation: {error*100:.1f} cm', font=FONTS[22], fill=TEXT)
    draw.rectangle((0, HEIGHT-61, int(WIDTH*(index+1)/meta['expected_frames']), HEIGHT-59), fill=ORANGE)
    return panel


def task_card(clip, index, total):
    panel = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((70, 230, 78, 450), fill=ORANGE)
    draw.text((110, 225), f'TASK {index:02d} / {total:02d}', font=FONTS[24], fill=ORANGE)
    draw.text((110, 275), clip['title'], font=FONTS[52], fill=TEXT)
    draw.text((110, 355), f'Learned checkpoint {clip["checkpoint_label"]}  |  Reference overlaid in orange', font=FONTS[24], fill=TEXT)
    draw.text((110, 403), 'Real-time playback  |  Independent evaluation from frame zero', font=FONTS[22], fill=TEXT)
    return panel


def stop_card(last, clip):
    panel = last.copy()
    draw = ImageDraw.Draw(panel)
    meta = clip['metadata']
    reason = 'Torso-height limit' if meta['terminations'].get('anchor_pos') else 'Evaluation termination'
    draw.rectangle((115, 260, WIDTH-115, 420), fill=BACKGROUND, outline=(180, 48, 35), width=3)
    draw.text((145, 282), f'ROLLOUT STOPPED AT {len(clip["actual"])*meta["policy_dt_s"]:.2f} s', font=FONTS[32], fill=(165, 44, 34))
    draw.text((145, 334), f'{reason}  |  Reference length: {meta["expected_frames"]*meta["policy_dt_s"]:.2f} s', font=FONTS[24], fill=TEXT)
    draw.text((145, 375), 'Last recorded pose held here; remaining reference was not executed.', font=FONTS[22], fill=TEXT)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT/'GMR/assets/booster_t1/T1_serial.xml'
    model = mj.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.headlight.ambient[:] = .3
    model.vis.headlight.diffuse[:] = .65
    model.vis.headlight.specular[:] = .3
    clips, fps = load_clips(args.directory, model)
    states = mj.MjData(model), mj.MjData(model)
    option, perturb, camera = mj.MjvOption(), mj.MjvPerturb(), mj.MjvCamera()
    option.geomgroup[3] = 0
    camera.distance, camera.azimuth, camera.elevation = 2.35, 130, -20
    video = args.output/'all_tasks_overlay.mp4'
    temporary = args.output/'all_tasks_overlay.render.mp4'
    if not args.preview_only and video.exists():
        raise FileExistsError(video)
    writer, frames, chapters, previews = None, 0, [], []
    try:
        if not args.preview_only:
            writer = imageio_ffmpeg.write_frames(str(temporary), (WIDTH, HEIGHT), fps=fps,
                codec='libx264', quality=8, macro_block_size=2, pix_fmt_out='yuv420p',
                output_params=['-threads', '2', '-movflags', '+faststart'])
            writer.send(None)
        with mj.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            for task, clip in enumerate(clips, 1):
                start = frames
                card_frames = int(round(fps))
                if writer:
                    card = np.asarray(task_card(clip, task, len(clips)))
                    for _ in range(card_frames):
                        writer.send(card)
                    frames += card_frames
                motion_start = frames
                # Inspect the largest root separation and each motion midpoint.
                review = sorted({len(clip['actual'])//2,
                    int(np.linalg.norm(clip['actual'][:, :3]-clip['reference'][:, :3], axis=-1).argmax())})
                for index in review if args.preview_only else range(len(clip['actual'])):
                    panel = overlay_frame(renderer, model, states, option, perturb, camera, clip, index, task, len(clips))
                    if writer:
                        writer.send(np.asarray(panel))
                        frames += 1
                    if index in review:
                        panel.save(args.output/f'{clip["name"]}_{index:04d}.jpg', quality=92)
                    if index == review[-1]:
                        previews.append(panel.resize((640, 360), Image.Resampling.LANCZOS))
                        if task == len(clips):
                            panel.save(args.output/'preview.png')
                    if writer and (index+1) % 100 == 0:
                        print(f'{clip["name"]}: {index+1}/{len(clip["actual"])} frames', flush=True)
                stop_frames = 0
                if writer and not clip['metadata']['completed']:
                    card = np.asarray(stop_card(panel, clip))
                    stop_frames = int(round(2*fps))
                    for _ in range(stop_frames):
                        writer.send(card)
                    frames += stop_frames
                chapters.append(dict(motion=clip['name'], title=clip['title'], start_frame=start,
                    motion_start_frame=motion_start, end_frame_exclusive=frames,
                    recorded_frames=len(clip['actual']), reference_frames=clip['metadata']['expected_frames'],
                    title_frames=card_frames, stop_notice_frames=stop_frames,
                    completed=clip['metadata']['completed'], failure_reason=clip['metadata']['failure_reason'],
                    source=str(clip['source'].resolve()), source_sha256=sha256(clip['source'])))
                print(f'{"Previewed" if args.preview_only else "Rendered"} {clip["name"]}', flush=True)
    finally:
        if writer:
            writer.close()
    sheet = Image.new('RGB', (1920, 360*((len(previews)+2)//3)), BACKGROUND)
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i % 3)*640, (i//3)*360))
    sheet.save(args.output/'review_frames.jpg', quality=92)
    if args.preview_only:
        return
    chapter_file = args.output/'chapters.ffmetadata'
    chapter_file.write_text(';FFMETADATA1\n'+''.join(
        f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(c["start_frame"]/fps*1000)}\n'
        f'END={round(c["end_frame_exclusive"]/fps*1000)}\ntitle={c["title"]}\n' for c in chapters))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(temporary),
        '-i', str(chapter_file), '-map_metadata', '1', '-map_chapters', '1', '-c', 'copy',
        '-movflags', '+faststart', str(video)], check=True)
    decoded, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    if decoded != frames or abs(duration-frames/fps) > .05:
        raise ValueError(f'Export validation failed: {decoded} frames, {duration} seconds')
    temporary.unlink()
    manifest = dict(file=video.name, checkpoint=clips[0]['metadata']['checkpoint'],
        checkpoint_sha256=clips[0]['metadata']['checkpoint_sha256'],
        frames=frames, decoded_frames=decoded, fps=fps, duration_s=frames/fps,
        resolution=[WIDTH, HEIGHT], reference_rgba=REFERENCE_RGBA, actual_rgba=ACTUAL_RGBA,
        recorded_motion_frames=sum(len(c['actual']) for c in clips), chapters=chapters,
        simulation_rerun=False, pose_interpolation=False, playback_speed=1.,
        source='Saved deterministic PhysX pre-step poses and their prepared reference poses.',
        alignment='Original shared world frame; no root alignment or spatial offset.',
        failure_handling='Stops at the last saved pre-step pose, then shows an explicit held-pose notice; no invented terminal or remaining motion.',
        camera=dict(distance=camera.distance, azimuth=camera.azimuth, elevation=camera.elevation),
        model_xml_sha256=sha256(model_path), renderer_sha256=sha256(Path(__file__)), video_sha256=sha256(video))
    (args.output/'video_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    links = ''.join(f'<button onclick="const v=document.querySelector(\'video\');v.currentTime={c["start_frame"]/fps};v.play()">'
                    f'{i:02d}. {html.escape(c["title"])}</button>' for i, c in enumerate(chapters, 1))
    stopped = ' '.join(f'{html.escape(c["title"])} stops at {len(c["actual"])*c["metadata"]["policy_dt_s"]:.2f} s '
        f'of its {c["metadata"]["expected_frames"]*c["metadata"]["policy_dt_s"]:.2f} s reference.'
        for c in clips if not c['metadata']['completed'])
    (args.output/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Learned checkpoint: all tasks</title>'
        '<style>body{max-width:1280px;margin:30px auto;padding:0 20px;font:17px system-ui;background:#f9fafc;color:#23303e}'
        'video{width:100%}button{padding:10px;margin:4px;border:1px solid #ccd2d8;border-radius:6px;cursor:pointer;background:white}</style>'
        f'<h1>Learned checkpoint {html.escape(clips[0]["checkpoint_label"])}: all {len(clips)} tasks</h1>'
        '<p>Reference: translucent orange. Learned policy: solid slate. Playback at 1x speed.</p>'
        '<video controls preload="metadata" poster="preview.png" src="all_tasks_overlay.mp4"></video><p>'+links+'</p>'
        f'<p>{stopped} {sum(c["metadata"]["completed"] for c in clips)}/{len(clips)} clips complete. Each task is an independent rollout from frame zero.</p>'
        '<p>These recordings use the existing checkpoint, trained before the new force-mismatch penalty.</p>'
        '<p><a href="all_tasks_overlay.mp4">Download video</a> &middot; <a href="video_manifest.json">Recording details</a></p>\n')
    print(f'Verified {decoded} frames, {frames/fps:.2f} s: {video}', flush=True)


if __name__ == '__main__':
    main()
