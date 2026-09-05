"""Retarget SMPL-X motion to a portable robot-motion NPZ without a viewer."""

import argparse
from pathlib import Path

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="booster_t1")
    parser.add_argument("--body-models", type=Path, default=root / "GMR/assets/body_models")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    data, model, output, height = load_smplx_file(str(args.input), str(args.body_models))
    frames, fps = get_smplx_data_offline_fast(data, model, output, tgt_fps=args.fps)
    retargeter = GeneralMotionRetargeting(
        actual_human_height=height, src_human="smplx", tgt_robot=args.robot
    )
    qpos = np.asarray([retargeter.retarget(frame).copy() for frame in frames])
    if qpos.ndim != 2 or not np.isfinite(qpos).all():
        raise ValueError("Retargeting produced empty or non-finite motion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fps=np.asarray(fps),
        root_pos=qpos[:, :3],
        root_rot=qpos[:, [4, 5, 6, 3]],  # xyzw, matching scripts/csv_to_npz.py
        dof_pos=qpos[:, 7:],
    )
    print(f"Saved {len(qpos)} frames at {fps:.6f} fps to {args.output}")


if __name__ == "__main__":
    main()
