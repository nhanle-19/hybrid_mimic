# Manual per-frame FORCE and AUTH schedules

The kick experiment supplies raw NN outputs from a separate NPZ while keeping
the original motion NPZ as the reference. FORCE has two channels: left/right
normal-force-limit logits. Cost-only AUTH has six channels: left xyz, right xyz.
The reference has 142 frames at 50 Hz; a complete attempt is 142 control steps
(2.84 seconds), starting at frame zero. No learned checkpoint is used.

Both FORCE and AUTH now use `c = clip((sigmoid(z)-0.10)/0.80, 0, 1)`.
Raw values -3 and +3 give exact 0 and 1; zero still gives 0.5. FORCE uses `c`
for force ceilings; AUTH uses `c` only for squared translational cost weights.
The schedule loader requires version 2 metadata recording this mapping, and
rejects older tables until they are migrated. The selected migrated files are
`kick_force_deadzone010.npz` and `kick_authority_deadzone010.npz`.
Migration preserves the old mapped fractions and marks the new table as needing
replay. It does not validate new labels or establish motion completion.

Newly generated FORCE tables now taper the decoded weights with a quintic
envelope before reference lift-off, hold exact zero during reference swing,
and restore capacity gradually after reference touchdown. The ramp spans about
80 ms. This happens when preparing the fixed table; playback does not replace
the NN outputs. Previously saved schedules are unchanged.

Scheduled playback through `scripts/rsl_rl/eval_wbc.py` preserves the supplied
output table exactly. OFF intervals and their smooth transitions belong in
the decoded input weights. There is no runtime manual-output replacement.
Two-foot FORCE playback uses the caps alone to determine modeled support,
without an actual or reference contact mask. FORCE zero removes modeled forces;
AUTH retains its geometry gate, and AUTH zero removes only its contact-cost weights.

Mapping validation passed 61 targeted tests (three CUDA tests skipped), plus
the 273-case AUTH QP audit on CPU and OSQP. A five-step FORCE CPU playback used
the migrated table with matching recorded caps, no resets, and no QP failures.
Isaac Sim then exited with code 139 during shutdown after saving diagnostics.
This validates the short playback path, not the full motion or GPU behavior.
Details are in `eval_data/manual_schedules/deadzone010_validation.json`.

The kick experiment's FORCE run is still incomplete. AUTH kick work is paused
at the user's request; its early trial artifacts are retained for reference.
The kick schedules began as
hand-designed experiments, followed by FORCE-only table tuning; they are not
successful solutions.
The following are historical plain-sigmoid CPU PhysX trials, which stopped on end-effector height tracking error before completing
the kick, with no QP failures:

| Schedule | Steps before termination | Simulated time |
| --- | ---: | ---: |
| FORCE: reference load estimate | 17 | 0.34 s |
| FORCE: constant 98% caps, comparison | 21 | 0.42 s |
| FORCE: cap support at 55% | 16 | 0.32 s |
| FORCE: reference schedule with 35% recovery floor | 23 | 0.46 s |
| FORCE: tuned per-step table, selected and replayed singly | 35 | 0.70 s |
| AUTH: early speed-sensitive xyz trial (paused) | 23 | 0.46 s |
| AUTH: early stronger support xyz trial (paused) | 23 | 0.46 s |

Only the prefix up to failure was executed. These trials do not establish
whether an NN can learn a successful solution. CPU validation does not replace
GPU validation. In this session Isaac Sim also exited with code 139 during
shutdown after saving the failed rollouts; the tracking terminations above
occurred earlier inside the control loop.

## Run the saved schedules

From the repository root:

```bash
conda activate hybridmimic
export PYTHONNOUSERSITE=1
python scripts/rsl_rl/eval_manual_wbc.py --controller force --device cuda:0 --headless --video
```

The selected files are `eval_data/manual_schedules/kick_force_deadzone010.npz` and
`kick_authority_deadzone010.npz`. Adjacent CSVs show every raw action and mapped fraction;
PNGs show the schedules and the executed prefix. Numerical results for all
trials are in `eval_data/manual_schedules/validation.json`.

New playback results go to `eval_data/manual_schedules/force_deadzone010_playback/` and
`authority_deadzone010_playback/`, with `diagnostics.npz` and a `videos/` directory.
Use `--output_dir` for a different destination. For CPU evaluation use
`--device cpu --headless` without video. `--steps` optionally runs a shorter
prefix; that is not counted as completing the motion.

The motion command advances without resetting the robot at clip end. Playback
checks actual reference-frame indices against the schedule, restores tracking
and fall terminations, extends the timeout past the clip, and stops on the first
failure. Applied actions, frame numbers, pre-step hand/foot height errors, and
completion status are saved in diagnostics.

To migrate another legacy table without changing its force caps or AUTH costs:

```bash
python scripts/rsl_rl/manual_wbc_schedule.py --migrate \
  eval_data/manual_schedules/kick_force.npz \
  eval_data/manual_schedules/kick_force_migrated_copy.npz
```

The source stays unchanged and an existing destination is rejected. Historical
validation stays in the migration metadata, separate from validation of the new
table. Start playback from frame zero; no simulator state is resumed at 0.70 s.

## How the manual values were assigned

Schedule generation uses the same prepared reference that the controller uses:
joint limits, ground-penetration correction, and reconstructed kinematics are
applied first. Foot sole clearance comes from the collision vertices. A smooth
near-ground confidence fades between 3 and 43 mm; this is a planning heuristic,
not an assertion of physical contact. For two-foot FORCE this geometry assists
offline labeling only; it does not mask the runtime QP. AUTH retains its geometry
gate. `reference_contact_available` separately records reference
geometric availability; the retargeted right foot can remain slightly elevated
even during the apparent recovery phase.

FORCE estimates load sharing from the CoM projection between the feet and
smoothed vertical CoM acceleration. It adds force headroom, anticipates landing
by up to 80 ms, and smooths transitions. An initial revision retained at least
210 N of force capacity per foot so early/recovery contact was not restricted to
the original 15 N swing cap. Further FORCE-only tuning changed short action-table
windows; the selected table includes lower early support caps and is recorded
exactly in the NPZ/CSV. Unexecuted rows after frame 34 retain the reference-based
recovery-floor schedule, with a five-frame blend, instead of retaining random
search perturbations in the untested tail. These are upper bounds, not commanded forces.

AUTH applies stronger translational cost to near-ground, slowly moving feet,
relaxing it as the reference foot lifts or moves. Newly generated AUTH tables
also multiply each foot's authority by `sqrt(0.5 + estimated_load_share)` before
smoothing and clipping. This preserves the equal-load baseline and shifts cost
toward the more heavily loaded foot when the CoM moves, including sway with
stationary feet. The design is recorded as `reference_geometry_and_load_share_v2`;
previously saved tables retain their original labels. The xyz fractions are
mapped to raw logits. Rotational contact costs remain absent and force caps
stay fixed; the schedule changes neither controller gains nor physical bounds.

AUTH's contact objective penalizes zero-target foot acceleration while support
geometry is available. A large z weight can therefore resist lift-off even
though there is no hard foot-height constraint. The current generic generator
uses clearance and speed without an explicit pre-lift release phase. A stepping
experiment needs such a release in its manual labels and a physics replay with
actual lift-off; completing the planted-foot toy sway does not validate that
transition. Zero authority removes the cost while leaving physical support
forces available.

The NPZ contains `actions`, `fractions`, `reference_frame`, `time_s`, `fps`,
reference geometry/load diagnostics, phase labels, and JSON metadata. Metadata
records action order, the source-file hash, prepared-reference hash, controller
configuration, design, and validation status. Loading rejects mismatched motion,
frame rate, action layout, or controller settings. Existing twelve-output AUTH
checkpoints/actions are not used.

## Generate another reference-based baseline

Generation needs the simulator to obtain the exact prepared reference. It saves
the complete schedule before attempting playback, even when the rollout later
fails. Choose a fresh filename; existing schedule files are not overwritten.

```bash
python scripts/rsl_rl/eval_wbc.py \
  --task Tracking-WBC-FORCE-T1-Eval-v0 --qp_only \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --build_manual_schedule eval_data/manual_schedules/kick_force_new.npz \
  --device cuda:0 --headless --num_envs 1 \
  --output eval_data/manual_schedules/force_new/diagnostics.npz
```

For AUTH substitute `Tracking-WBC-AUTHORITY-T1-Eval-v0` and distinct output names.
The generator produces a reference-based candidate with exact-zero swing
weights and smooth release/restore, not the historical FORCE recovery-floor
variant. The old generator retained a 15 N swing cap; that positive floor is
now removed during reference swing. Actual timing still needs a physics replay.
To revise a schedule, update `fractions` and
`actions = encode_contact_values_numpy(fractions).astype(np.float32)` together,
using the helper exported by `manual_wbc_schedule`, keeping the frame
alignment and provenance intact, and save under a new filename. CSVs are review
exports; changing a CSV does not change the NPZ used for playback.

```bash
python scripts/plot_manual_wbc_schedule.py \
  --schedule eval_data/manual_schedules/kick_force_deadzone010.npz
```

## FORCE tuning and remaining limitation

The tuning tool changes only the two force-cap outputs. It searches piecewise
logit profiles and short per-step windows, terminating each trial at its first
failure. Batched candidate scores are provisional: the selected schedule was
replayed in a fresh single-environment run and failed at step 35. A final serial
search in one environment also plateaued at 35 steps. Candidate histories and
exact action tables are retained in the `force_tuning*`, `force_contact_refinement`,
and `force_serial_refinement` output directories. Scores include repeated trials
and are not counts of unique schedules.

```bash
python scripts/rsl_rl/tune_manual_force.py \
  --schedule eval_data/manual_schedules/kick_force_deadzone010.npz \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --output_dir eval_data/manual_schedules/force_tuning_new \
  --device cpu --headless --serial --candidates 24 --iterations 6 \
  --skip_initial_grid --local_search --cap_space_search --knot_stride 1
```

The historical best validated rollout loses all available support at frame 30, then fails
trunk-height and end-effector-height checks at step 35. Before contact loss, the
QP assigns about 195–217 N to the left foot while predicting 34–41 m/s² upward
acceleration at the foot link origin. FORCE contains neither a foot-acceleration
objective nor a stationary-contact equality. This is a warning about the
controller/contact behavior, not proof of global infeasibility: a different
earlier trajectory could behave differently. The two force ceilings cannot
create contact or independently demand a stationary support foot.

A lack of contact is expected during a jump's flight phase. However,
the prepared reference in this saved kick experiment places the left sole at
ground level around 0.60–0.70 s. Confirm the intended reference and flight timing
before assigning zero-force labels for a jump. The recorded failure is excessive
height tracking error, not the absence of contact by itself.

`force_failure_analysis.json` records the relevant measurements. Controller
objectives, gains, contact rules, and failure thresholds have not been changed;
changing those would be a separate experiment requiring a new scope decision.
