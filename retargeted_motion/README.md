# Retargeted motions

This directory holds versioned robot motion NPZ files. The source ACCAD dataset
and SMPL-X body models remain local and ignored by Git.

`training_list/` contains the 68 selected training references. The separate
`training_spider/` set contains the matching 68 NPZs copied unchanged from
`outputs/force2_spider_light/motions/`, with the static-standing pair excluded.
Both use the training array format at 50 Hz, totaling 23,210 frames. The SPIDER
set's `manifest.json` records hashes, source references, optimizer settings and
per-clip assessments. File/loader validation does not establish dynamic feasibility.
Mirrored-reference clips were optimized separately, so their SPIDER outputs need
not be exact reflections. Select either directory as the motion source.

## References with deliberate static endings

`training_static_end/` is the revised library for tasks that should finish in a
stationary pose: **32 originals and 32 mirrors at 50 Hz**. Every file finishes
with a one-second static hold. The original `training_list/` remains unchanged
for checkpoint comparisons; `training_spider/` has not received these edits.

The preparation slows the selected source ending, adjusts the final pose to
supported foot placements, lands and locks the feet, and settles before holding.
Joint/body poses and velocities are rebuilt consistently. Six already stationary,
supported endings are preserved and extended. Crops and exact source hashes are
recorded in `configs/static_end_training.json` and the library manifest.
The front kick's active portion is additionally slowed to half speed because
the inherited reference exceeded a joint velocity limit; its final hold remains
one second.

Three sources were replaced:

| File | Replacement source |
| --- | --- |
| `19_walk_to_stand.npz` | ACCAD Female1 walk-to-stand take 2 |
| `21_side_step_right.npz` | ACCAD Male2 walking side-step right |
| `26_walk_to_crouch.npz` | ACCAD Male2 walk-to-crouch |

The prepared replacement inputs are preserved in `static_end_sources/` for
reproducibility. That directory is not the final training library.

`18_stand_to_walk` and `27_crouch_to_walk`, including their mirrors, are excluded.
The former has early foot collisions; the latter starts with hand/hip/shank
intersections. Endpoint edits did not produce acceptable versions preserving the
actions, and no validated replacement was found in the local source collection.

Validation checks all saved originals and mirrors: loader/schema, source hashes,
independent forward kinematics, joint limits/speeds, sole penetration, body
self-collision, and stationary terminal poses/velocities. Contact-force and
actuator-bound feasibility is checked throughout the final planted settling
phase and at the static pose. Earlier motion and landing remain kinematic
references; these checks do not establish successful policy tracking or complete
dynamic feasibility.

For multimotion FORCE training, select `retargeted_motion/training_static_end/`.
For flat training, select one file, such as
`retargeted_motion/training_static_end/20_side_step_left.npz`. The revised left
side step is 6.44 seconds; the replacement right side step is 4.78 seconds.
Some other clips exceed ten seconds, so an episode timeout must allow their
complete duration to reach the static ending. Clip completion should terminate
the episode; these files do not change the environment's boundary handling.

Rebuild into a separate directory using the existing MuJoCo/SciPy environment:

```bash
.venvs/dsms/bin/python scripts/prepare_static_motion_endings.py \
  --selection configs/static_end_training.json \
  --output outputs/static_end_rebuild --work outputs/static_end_rebuild_checks
.venvs/dsms/bin/python scripts/audit_static_motion_endings.py \
  --directory outputs/static_end_rebuild \
  --output outputs/static_end_rebuild_checks/endpoint_validation.json
```

`scripts/validate_contact_training_list.py` provides the independent Torch FK and
loader checks in the `hybridmimic` environment.
`scripts/review_static_motion_endings.py` renders the reference endings and four
full clips. The initial review is in
`outputs/static_end_build_20261125/review/report.html`; videos show references,
not policy rollouts. No policy was retrained as part of this preparation.

Generate the G18 right push kick from the repository root in the GMR environment:

```bash
python scripts/retarget_smplx.py \
  --input ACCAD/Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz \
  --robot booster_t1 \
  --output retargeted_motion/g18_push_kick_right_t1.npz
```

The exporter requires GMR and SMPL-X body models. Reading the exported arrays
requires only NumPy, using `np.load(path, allow_pickle=False)`.

Raw retarget exports contain `fps`, `root_pos` (world XYZ), `root_rot` (quaternion XYZW), and
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
