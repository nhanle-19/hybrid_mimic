"""Compose existing FORCE recordings with reference left and actual right.

This changes video presentation only; it never runs or modifies a simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)


def compose(actual, output, reference=None, reference_offset_frames=0, swap_walk=False):
    if output.resolve() == actual.resolve():
        raise ValueError('Render to a separate output before replacing the source')
    cap, count, fps = inspect(actual)
    refcap = None
    if swap_walk:
        width, height = 1280, 720
    else:
        if reference is None:
            raise ValueError('A reference recording is required')
        refcap, refcount, reffps = inspect(reference)
        if not np.isclose(fps, reffps) or reference_offset_frames < 0:
            raise ValueError('Videos must have the same FPS and a nonnegative offset')
        if refcount < count + reference_offset_frames:
            raise ValueError('Reference does not cover every actual frame')
        refcap.set(cv2.CAP_PROP_POS_FRAMES, reference_offset_frames)
        width, height = 1280, 440
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (width, height), fps=fps,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=8,
        macro_block_size=2, output_params=['-threads', '2', '-movflags', '+faststart'])
    writer.send(None)
    samples = []
    sample_indices = sorted(set([0, count//4, count//2, 3*count//4, count-1]))
    try:
        for i in range(count):
            ok, actual_frame = cap.read()
            if not ok:
                raise ValueError(f'Actual recording stopped at {i}/{count}')
            if swap_walk:
                if actual_frame.shape != (720, 1280, 3):
                    raise ValueError('Expected the original 1280x720 walk comparison')
                frame = actual_frame.copy()
                frame[80:, :640] = actual_frame[80:, 640:]
                frame[80:, 640:] = actual_frame[80:, :640]
                frame[:45] = 245
            else:
                ok, reference_frame = refcap.read()
                if not ok:
                    raise ValueError(f'Reference recording stopped at {i}/{count}')
                frame = np.full((height, width, 3), 245, dtype=np.uint8)
                frame[48:408, :640] = cv2.resize(reference_frame, (640, 360), interpolation=cv2.INTER_AREA)
                frame[48:408, 640:] = cv2.resize(actual_frame, (640, 360), interpolation=cv2.INTER_AREA)
                label = f't = {(i+reference_offset_frames)/fps:.2f} s | recorded frame {i+1}/{count}'
                cv2.putText(frame, label, (16, 431), cv2.FONT_HERSHEY_SIMPLEX,
                            .51, (45, 45, 45), 1, cv2.LINE_AA)
            for label, x in [('REFERENCE', 16), ('ACTUAL - FORCE', 656)]:
                cv2.putText(frame, label, (x, 31), cv2.FONT_HERSHEY_SIMPLEX,
                            .68, (30, 30, 30), 2, cv2.LINE_AA)
            writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if i in sample_indices:
                samples.append(cv2.resize(frame, (960, round(height*.75))))
            if i == count//2:
                cv2.imwrite(str(output.with_name('preview.png')), frame)
    finally:
        writer.close()
        cap.release()
        if refcap is not None:
            refcap.release()
    cv2.imwrite(str(output.with_name('review_frames.png')), np.concatenate(samples))
    cap, reported_count, output_fps = inspect(output)
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (height, width, 3)
        decoded += 1
    cap.release()
    if decoded != count or reported_count != count or not np.isclose(output_fps, fps):
        raise ValueError('Output frame count or timing changed')
    manifest = dict(file=output.name, frames=count, fps=output_fps,
        duration_s=count/fps, resolution=[width, height], decoded_frames=decoded,
        layout=dict(left='reference', right='actual_force'),
        actual_source=str(actual), actual_source_sha256=sha256(actual),
        output_sha256=sha256(output), reviewed_frame_indices=sample_indices,
        simulation_rerun=False, frame_interpolation=False,
        method='Swap only the two pose panels; preserve original timing and terminal hold.' if swap_walk
               else 'Pair existing kinematic reference and recorded physics video; preserve every actual frame.')
    if reference is not None:
        manifest.update(reference_source=str(reference), reference_sha256=sha256(reference),
            reference_offset_frames=reference_offset_frames,
            alignment='RecordVideo captures after env.step; reference starts one control frame later.'
                      if reference_offset_frames == 1 else 'Source clip starts aligned.')
    output.with_name('comparison_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), frames=decoded, fps=output_fps)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-offset-frames', type=int, default=0)
    parser.add_argument('--swap-walk', action='store_true')
    args = parser.parse_args()
    compose(args.actual, args.output, args.reference, args.reference_offset_frames, args.swap_walk)
