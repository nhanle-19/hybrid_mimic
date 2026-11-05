"""Render reference and recorded PhysX poses together; no dynamics are replayed."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def render(directory):
    data = dict(np.load(directory/'diagnostics.npz', allow_pickle=False))
    meta = json.loads(str(data['metadata_json']))
    actual, reference = data['audit_actual_q'], data['audit_reference_q']
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError('Invalid recorded poses')
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    state = mj.MjData(model)
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    order = [data['joint_names'].tolist().index(name) for name in names]
    width, height = 480, 400
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.1, 140, -17
    video = directory/'comparison.mp4'
    writer = imageio_ffmpeg.write_frames(str(video), (width*2, height), fps=1/meta['policy_dt_s'],
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=7,
        macro_block_size=2, output_params=['-threads', '2', '-movflags', '+faststart'])
    writer.send(None)
    previews = []
    selected = set(np.linspace(0, len(actual)-1, 4).astype(int).tolist())
    try:
        with mj.Renderer(model, height=height, width=width) as renderer:
            for index, (ref, act) in enumerate(zip(reference, actual)):
                panels = []
                actual_title = 'ACTUAL - manual force caps' if meta.get('policy_source') == 'manual_labels' else 'ACTUAL - learned contact policy'
                for title, q in [('REFERENCE', ref), (actual_title, act)]:
                    state.qpos[:] = np.r_[q[:3], q[[6, 3, 4, 5]], q[7:][order]]
                    mj.mj_forward(model, state)
                    camera.lookat[:] = [ref[0], ref[1], .45]
                    renderer.update_scene(state, camera, scene_option=option)
                    panel = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, width, 30), fill='white')
                    draw.text((10, 8), title, fill='black')
                    draw.rectangle((0, height-45, width, height), fill='white')
                    draw.text((10, height-37), f'{directory.name} | t={index*meta["policy_dt_s"]:.2f}s | {index+1}/{meta["expected_frames"]}', fill='black')
                    status = 'Full clip recorded' if meta['completed'] else f'PARTIAL: stops at {len(actual)*meta["policy_dt_s"]:.2f}s'
                    draw.text((10, height-20), status, fill='black' if meta['completed'] else 'red')
                    panels.append(np.asarray(panel))
                combined = np.concatenate(panels, axis=1)
                writer.send(combined)
                if index in selected:
                    previews.append(combined)
    finally:
        writer.close()
    Image.fromarray(np.concatenate(previews)).save(directory/'review_frames.jpg', quality=90)
    (directory/'video.json').write_text(json.dumps(dict(frames=len(actual), fps=1/meta['policy_dt_s'],
        policy_source=meta.get('policy_source', 'learned_policy'),
        source='Recorded pre-step PhysX states, visualized with MuJoCo; no dynamics or interpolation.',
        full_clip=meta['completed'], failed_terminal_state_included=False,
        diagnostics=str((directory/'diagnostics.npz').resolve())), indent=2)+'\n')
    print(f'Rendered {video}: {len(actual)} frames', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()
    for source in sorted(args.directory.glob('*/diagnostics.npz')):
        if not (args.skip_existing and (source.parent/'video.json').exists()):
            render(source.parent)
