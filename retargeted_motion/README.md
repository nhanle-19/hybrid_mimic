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

## Crouch QP test

Run from the repository root. The selected source is the Male2 general crouch
(`A7-_Crouch_stageii.npz`); the dataset also contains Male1 and Female1 variants.
The local `rl_lab` environment has the GMR dependencies.

```bash
conda activate rl_lab
python scripts/retarget_smplx.py \
  --input ACCAD/Male2General_c3d/A7-_Crouch_stageii.npz \
  --robot booster_t1 \
  --output retargeted_motion/a7_crouch_t1.npz

conda activate hybridmimic
python scripts/csv_to_npz.py \
  --input_file retargeted_motion/a7_crouch_t1.npz \
  --input_fps 30 --output_fps 50 --output_name a7_crouch_t1 \
  --headless --local_only

python scripts/rsl_rl/eval_wbc.py \
  --task Tracking-WBC-ACC-T1-Eval-v0 \
  --motion_file retargeted_motion/a7_crouch_t1_training.npz \
  --qp_only \
  --num_envs 1 --steps 218 --headless --video \
  --video_dir eval_data/wbc_acc/crouch/videos \
  --output eval_data/wbc_acc/crouch/diagnostics.npz

python scripts/plot_wbc.py \
  --input eval_data/wbc_acc/crouch/diagnostics.npz \
  --output_dir eval_data/wbc_acc/crouch/plots
```

The raw retargeted clip has 132 frames. With the conversion settings above,
the training-format file has 219 frames; 218 evaluation steps follow the
existing evaluation convention of stopping before the clip wraps. `reference`
uses automatic height-based contact selection, not manually verified labels.
The QP run and converted motion still need inspection before claiming that
the crouch is balanced or that both feet remain in support throughout.
