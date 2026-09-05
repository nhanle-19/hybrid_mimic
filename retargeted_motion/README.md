# Retargeted motions

This directory holds versioned robot motion NPZ files. The source ACCAD dataset
and SMPL-X body models remain local and ignored by Git.

Generate the G18 right push kick from the repository root in the GMR environment:

```bash
python scripts/retarget_smplx.py \
  --input ACCAD/Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz \
  --robot booster_t1 \
  --output retargeted_motion/g18_push_kick_right_t1.npz
```

The exporter requires GMR and SMPL-X body models. Reading the exported arrays
requires only NumPy, using `np.load(path, allow_pickle=False)`.

Files contain `fps`, `root_pos` (world XYZ), `root_rot` (quaternion XYZW), and
`dof_pos` (GMR robot joint order). G18 uses the 23-joint `booster_t1` model.
These are inputs to `scripts/csv_to_npz.py`; training requires that additional
Isaac Lab conversion to produce joint velocities and body transforms.
