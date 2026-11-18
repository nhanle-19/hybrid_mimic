"""Render original references against continuous DSMS simulator rollouts."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from render_force_suite_overlay import ROOT, FONTS, tint, REFERENCE_RGBA, ACTUAL_RGBA


def render(folder, names):
    model = mj.MjModel.from_xml_path(str(ROOT / 'GMR/assets/booster_t1/T1_serial.xml'))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    states = mj.MjData(model), mj.MjData(model)
    camera, option, perturb = mj.MjvCamera(), mj.MjvOption(), mj.MjvPerturb()
    option.geomgroup[3] = 0
    camera.azimuth, camera.elevation = 130, -20
    results = []
    with mj.Renderer(model, 720, 1280) as renderer:
        for name in names:
            clip = folder / 'clips' / name
            z = np.load(clip / 'rollout.npz', allow_pickle=False)
            meta = json.loads(str(z['metadata_json']))
            sim_dt = meta['settings']['sim_dt']
            stride = round(.02 / sim_dt)
            actual = z['state'][::stride, :30]
            reference = z['reference'][::stride, :30]
            # The exported USD tree and the GMR display model share this order;
            # assert rather than silently displaying reordered joints.
            physics = mj.MjModel.from_xml_path(str(folder / 'model/t1_force2.xml'))
            for j in range(1, model.njnt):
                assert mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) == mj.mj_id2name(physics, mj.mjtObj.mjOBJ_JOINT, j)
            camera.distance = max(2.4, 1.65 * np.linalg.norm(actual[:, :2] - reference[:, :2], axis=1).max() + 1.3)
            path = clip / 'comparison.mp4'
            writer = imageio_ffmpeg.write_frames(str(path), (1280, 720), fps=50, codec='libx264',
                                                pix_fmt_in='rgb24', pix_fmt_out='yuv420p', output_params=['-crf', '20'])
            writer.send(None)
            for i, (a, r) in enumerate(zip(actual, reference)):
                for state, q in zip(states, (a, r)):
                    state.qpos[:] = q
                    mj.mj_forward(model, state)
                camera.lookat[:] = (a[:3] + r[:3]) / 2
                camera.lookat[2] = .55
                renderer.update_scene(states[0], camera, scene_option=option)
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.category == mj.mjtCatBit.mjCAT_DYNAMIC:
                        tint(geom, ACTUAL_RGBA)
                first = renderer.scene.ngeom
                mj.mjv_addGeoms(model, states[1], option, perturb, mj.mjtCatBit.mjCAT_DYNAMIC, renderer.scene)
                for geom in renderer.scene.geoms[first:renderer.scene.ngeom]:
                    tint(geom, REFERENCE_RGBA)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 95), fill='#f9fafc')
                draw.text((25, 15), name.replace('_', ' '), font=FONTS[32], fill='#23303e')
                draw.text((25, 58), 'Original: orange    |    DSMS simulation: slate    |    1x speed', font=FONTS[22], fill='#23303e')
                draw.rectangle((0, 660, 1280, 720), fill='#f9fafc')
                text = f'{i / 50:.2f}s   |   root change: {np.linalg.norm(a[:3] - r[:3]) * 100:.1f} cm'
                text += '   |   Screening: ' + ('PASS' if meta['accepted'] else 'FAIL')
                draw.text((25, 676), text, font=FONTS[22], fill='#23303e')
                writer.send(np.asarray(frame))
            if not meta['complete']:
                draw.rectangle((160, 270, 1120, 405), fill='#f9fafc', outline='#a52c22', width=3)
                draw.text((185, 292), 'Simulation stopped: ' + str(meta['failure']), font=FONTS[24], fill='#a52c22')
                draw.text((185, 345), 'Last simulated pose held; no continuation was generated.', font=FONTS[22], fill='#23303e')
                for _ in range(100):
                    writer.send(np.asarray(frame))
            writer.close()
            frame.save(clip / 'preview.jpg')
            count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
            assert count == len(actual) + (0 if meta['complete'] else 100)
            results.append(dict(motion=name, file=str(path.relative_to(folder)), frames=count, duration_seconds=duration))
    (folder / 'videos.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('motions', nargs='+')
    args = parser.parse_args()
    print(json.dumps(render(args.folder, args.motions), indent=2))
