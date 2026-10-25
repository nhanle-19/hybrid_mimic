"""Render two unmarked views for visual review of crawl contact intervals."""
import argparse
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 12, 24, 36, 48, 60, 70, 82, 95, 104, 115, 127])
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/crawl_contact_review/review_prefix')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    motion = np.load(ROOT/'retargeted_motion/a11_crawl_forward_female1_t1.npz')
    model = mj.MjModel.from_xml_path(str(ROOT/'GMR/assets/booster_t1/T1_serial.xml'))
    data = mj.MjData(model)
    qpos = np.c_[motion['root_pos'], motion['root_rot'][:, [3, 0, 1, 2]], motion['dof_pos']]
    width, height = 640, 360
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.distance, camera.elevation = 1.4, -8
    with mj.Renderer(model, height=height, width=width) as renderer:
        for side, azimuth in [('left', 90), ('right', 270)]:
            camera.azimuth = azimuth
            tiles = []
            for frame in args.frames:
                data.qpos[:] = qpos[frame]
                mj.mj_forward(model, data)
                camera.lookat[:] = qpos[frame, :3]
                camera.lookat[2] = .17
                renderer.update_scene(data, camera, scene_option=option)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 0), (width, 32), (250, 250, 250), -1)
                cv2.putText(image, f'Frame {frame} | {frame/float(motion["fps"]):.3f}s | {side}', (12, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (25, 25, 25), 1, cv2.LINE_AA)
                tiles.append(image)
            while len(tiles) % 3:
                tiles.append(np.full_like(tiles[0], 255))
            sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1) for i in range(0, len(tiles), 3)])
            cv2.imwrite(str(args.output.with_name(args.output.name+'_'+side+'.png')), sheet)
    print('Rendered', len(args.frames), 'frames from both sides:', args.output)


if __name__ == '__main__':
    main()
