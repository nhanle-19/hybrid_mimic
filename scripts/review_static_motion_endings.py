"""Render the revised references and summarize their endpoint checks (no policy rollout)."""
import argparse
import html
import json
from pathlib import Path
import subprocess

from contextlib import contextmanager

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dsms_t1_model import ROOT, load_reference, sha


@contextmanager
def video_writer(path, fps=25):
    writer = imageio_ffmpeg.write_frames(str(path), (800, 600), fps=fps, codec='libx264', quality=7,
                                        macro_block_size=2, output_params=['-movflags', '+faststart'])
    writer.send(None)
    try:
        yield writer
    finally:
        writer.close()


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.motions/'manifest.json').read_text())
    originals = [r for r in manifest['motions'] if not r.get('mirrored_from')]
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    state = mj.MjData(model)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 16)
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.azimuth, camera.elevation = 140, -16
    previews, rows, evidence = [], [], []
    with mj.Renderer(model, height=600, width=800) as renderer:
        for index, row in enumerate(originals):
            path = args.motions/row['file']
            assert sha(path) == row['sha256']
            z, x, fps = load_reference(path, model)
            assert fps == 50.
            q = x[:, :model.nq]
            n = len(q)
            start = max(0, n-round(4*fps))
            selected = q[start:]
            camera.lookat[:] = [*selected[:, :2].mean(0), .58]
            extent = np.linalg.norm(selected[:, :2]-camera.lookat[:2], axis=1).max()
            camera.distance = max(2.25, 2.1*(extent+.65))

            def frame(i):
                state.qpos[:] = q[i]
                mj.mj_forward(model, state)
                renderer.update_scene(state, camera, scene_option=option)
                canvas = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 800, 60), fill=(246, 248, 251))
                draw.text((16, 7), f'{index+1:02d}/{len(originals)}  {path.stem}', font=font, fill=(25, 42, 60))
                draw.text((16, 34), 'Edited reference only — no learned-policy simulation', font=small, fill=(55, 70, 90))
                phase = 'STATIC HOLD' if i >= n-50 else ('SETTLING' if i >= row['ending']['edited_from_frame'] else 'SOURCE MOTION')
                draw.rectangle((0, 561, 800, 600), fill=(246, 248, 251))
                draw.text((16, 570), f'{i/fps:.2f} / {(n-1)/fps:.2f} s   |   {phase}', font=font, fill=(25, 80, 65))
                return np.asarray(canvas)

            render = not args.only or path.stem in args.only
            if render:
                with video_writer(args.output/(path.stem+'_ending.mp4')) as writer:
                    for i in range(start, n, 2):
                        writer.send(frame(i))
                last = Image.fromarray(frame(n-1))
                last.save(args.output/(path.stem+'_final.png'))
            else:
                last = Image.open(args.output/(path.stem+'_final.png'))
            previews.append(last.resize((400, 300)))
            if render and path.stem in ('20_side_step_left', '21_side_step_right', '26_walk_to_crouch', '34_dance_mm'):
                camera.lookat[:2] = q[:, :2].mean(0)
                extent = np.linalg.norm(q[:, :2]-camera.lookat[:2], axis=1).max()
                camera.distance = max(2.25, 2.1*(extent+.65))
                with video_writer(args.output/(path.stem+'_full.mp4')) as writer:
                    for i in range(0, n, 2):
                        writer.send(frame(i))
            root = z['body_names'].tolist().index('Trunk')
            final_speed = float(np.abs(z['body_lin_vel_w'][-50:]).max())
            final_joint_speed = float(np.abs(z['joint_vel'][-50:]).max())
            support = row['endpoint_validation']['final_support']
            retained = row['ending']['source_end_frame']/fps
            original = dict(np.load(ROOT/row['source'], allow_pickle=False))
            original_seconds = (len(original['joint_pos'])-1)/fps
            values = [row['file'], f'{retained:.2f} / {original_seconds:.2f}', f'{n/fps:.2f}',
                      'Replacement' if row.get('replacement_for') else 'Original source',
                      f'{support["peak_torque_fraction"]*100:.1f}%',
                      f'<a href="{path.stem}_ending.mp4">Ending</a>']
            rows.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>')
            evidence.append(dict(file=path.name, sha256=sha(path), frames=n,
                                 retained_source_s=retained, source_duration_s=original_seconds,
                                 final_second_body_linear_velocity_max=final_speed,
                                 final_second_joint_velocity_max=final_joint_speed,
                                 final_root_height_m=float(z['body_pos_w'][-1, root, 2])))
            if render:
                print(f'Rendered {path.stem}', flush=True)
    concatenate = args.output/'video_concat.txt'
    concatenate.write_text(''.join(f"file '{(args.output/(Path(r['file']).stem+'_ending.mp4')).resolve()}'\n" for r in originals))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(concatenate), '-c', 'copy', '-movflags', '+faststart',
                    str(args.output/'all_endings.mp4')], check=True)
    sheet = Image.new('RGB', (1600, 300*((len(previews)+3)//4)), 'white')
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i%4)*400, (i//4)*300))
    sheet.save(args.output/'final_poses.png')
    (args.output/'review.json').write_text(json.dumps(evidence, indent=2)+'\n')
    excluded = ''.join(f'<li>{html.escape(r["file"])} and mirror: {html.escape(r["reason"])}</li>' for r in manifest['excluded'])
    replacements = ''.join(f'<li>{html.escape(r["file"])}: {html.escape(r["source"])}</li>' for r in manifest['replacements'])
    page = f'''<!doctype html><html><meta charset="utf-8"><title>Deliberate static motion endings</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 22px;color:#203345;line-height:1.6}}video,img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f5}}</style>
<h1>{len(originals)*2} references with deliberate static endings</h1>
<p>{len(originals)} originals plus mirrors, at 50 Hz. Every clip ends with a one-second unchanged pose and zero saved velocities. The previous training library is preserved.</p>
<p>Moving endings use a monotone decelerating phase, a balanced final pose, and foot trajectories that finish landing before the final planted settling segment. Joint and body velocities are recomputed. Existing supported static endings are preserved and extended by one second.</p>
<p>The front kick's active portion is slowed to half speed because its original reference exceeded a joint velocity limit. Its final hold stays one second.</p>
<p>Accepted clips pass full-clip sole penetration and self-collision checks (3 mm maximum self-collider penetration tolerance). Modified endpoints have grounded feet, COM support margin and bounded static contact forces. Every reference frame of the final planted settling segment satisfies inverse dynamics under friction and actuator bounds. Earlier reference motion and landing are not certified dynamically feasible. These videos show references, not learned-policy rollouts.</p>
<p><a href="../final_endpoint_validation.json">Endpoint, speed and collision validation</a> · <a href="../final_validation.json">Independent FK, loader and mirror validation</a> · <a href="video_validation.json">Video frame counts</a></p>
<p><b>Training library:</b> retargeted_motion/training_static_end/ · <b>Reproduction:</b> configs/static_end_training.json</p>
<h2>Ending montage</h2><video controls preload="metadata" src="all_endings.mp4"></video>
<h2>Full revised clips</h2><p><a href="20_side_step_left_full.mp4">Left side step</a> · <a href="21_side_step_right_full.mp4">Replacement right side step</a> · <a href="26_walk_to_crouch_full.mp4">Replacement walk to crouch</a> · <a href="34_dance_mm_full.mp4">Dance MM</a></p>
<img src="motion_progression.png" alt="Left side step, right side step, walk to crouch and dance at successive phases">
<h2>Replacements</h2><ul>{replacements}</ul><h2>Excluded pending a better source</h2><ul>{excluded}</ul>
<h2>Selected source endpoints</h2><p>Source times refer to the prepared robot reference; upstream human-mocap times, including prior crops, are retained in provenance. Final duration includes deceleration and the static hold. Static torque fraction is a model feasibility quantity, not measured motor torque.</p>
<table><tr><th>File</th><th>Retained / available source s</th><th>Final s</th><th>Source</th><th>Static torque bound used</th><th>Video</th></tr>{''.join(rows)}</table>
<h2>Final poses</h2><img src="final_poses.png"><p><a href="review.json">Per-motion review data</a></p></html>'''
    (args.output/'report.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions', type=Path, default=ROOT/'retargeted_motion/training_static_end')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/static_end_build_20261125/review')
    parser.add_argument('--only', nargs='*', help='Refresh selected renders and rebuild the report/montage using existing other videos')
    main(parser.parse_args())
