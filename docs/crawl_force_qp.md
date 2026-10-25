# Eight-channel crawl FORCE QP

The corrected manual-weight table completes all **599 frames / 19.992 seconds**
on CPU, with **zero QP failures** and no positive output at an unavailable
contact in the recorded rollout. It travels 1.204 m; peak root error is
0.382 m and peak joint RMSE is 4.00 degrees. This remains a
preliminary result under the 0.5 m root-error cutoff.

The fixed input has 2,396 samples at 119.85 Hz for 599 reference frames.
Decoded weights taper through short quintic ramps (two QP intervals,
16.7 ms) into exact-zero OFF plateaus, then restore. Both knees remain
geometrically available throughout. No runtime manual-output override is used.
Reference, pose gains and static/dynamic/QP friction **0.2** remain unchanged.

Height overshoot is 4.12 mm and peak downward speed is
0.093 m/s. The hand-force peak is 309 N at every physics
step (141 N at QP sampling). A brief incidental head touch peaks at
3.64 N; incidental arm/right-foot contacts remain. All 2,396 runtime QPs satisfy
the force/torque constraints and decoded OFF forces are exactly zero.

This revised table has one completed physical validation; the previous
table's exact-repeat evidence is historical. `force/manual_weight_fix.json`
and `force/verification.json` record the current scope and comparison.

The earlier vertical-weight and friction comparisons in `force/bounce_comparison*`,
`force/before_after*` and `force/friction_tuning.json` describe the archived table.
The current video is `force/video.mp4`. Earlier complete runs and this correction
are retained under `outputs/manual_contact_fix_20261023/`.

The case follows the other evaluation cases:

```text
eval_data/reference_crawl_forward/
  reference.npz, reference.mp4, reference_preview.png
  case.json, validation.json, reproduction.json, artifact_sha256.json
  reference_validation.json, smoothing.json, dynamic_feasibility.json/.npz
  contact_geometry.npz, contact_labels.csv, contact_schedule.json, label_audit.json
  hand_motion/                  # source / GMR / smoothed-reference comparison
  physics_asset/                # matching crawl collision geometry
  force/
    planned_labels.npz/.csv      # complete manual NN input table
    labels.npz/.csv              # policy-sample execution with executed_mask
    diagnostics.npz              # every QP update and actual physics state
    validation.json, verification.json, contact_audit.json
    controller_profile.json, replay_command.json
    video.mp4, preview.gif, preview.png, review_frames.png
    before_after.mp4, before_after.gif, bounce_comparison.json/.png
    friction_tuning.json, friction_tradeoffs.png
    tracking.png, forces.png, labels.png, tests.log
```

The reference has 599 frames. Manual output tables may use an integer multiple
of that rate, recorded as `sample_rate_multiple`; `reference_frame` maps every
policy sample to its reference frame. Effective contact availability and caps in
`labels.npz` describe the first applied QP update of each policy sample;
`diagnostics.npz` retains every update. The video contains 600 recorded states,
including the final state, beside the reference. MuJoCo renders saved states;
the physics execution is Isaac Sim / PhysX.

## Inputs and force bounds

The fixed channel order is left hand, right hand, left knee, right knee,
left forearm, right forearm, right foot, left upper arm.
The editable JSON is `configs/crawl_manual_nn_outputs.json`; its decoded table
exactly matches `force/planned_labels.npz`. Both formats can be replayed.
For control-rate tables, JSON interval endpoints are policy-sample indices,
and `frame_count` remains the reference length. `policy_sample` in diagnostics
records the actual input row supplied to each QP update.
These are hand-designed, data-informed NN replacements, not a learned policy
or measured force labels. Inputs were tuned after checking actual contacts.

The existing FORCE decoder is `clip((sigmoid(raw)-0.1)/0.8, 0, 1)`:
raw -3 is exactly OFF, raw +3 gives full capacity, and raw 0 gives half capacity.
With the world floor normal `n = (0, 0, 1)`, each of the eight bounds is

```text
0 <= sum_p n.T f[i,p] <= decoded_output[i] * 600 N
```

They bound force; they do not prescribe force equalities. Live contact geometry
removes unavailable points. Decoded OFF outputs, unavailable points and padding
have exactly zero predicted force. Reference annotations never multiply the
NN outputs in the QP. The friction pyramid uses mu=0.2 in both QP and PhysX.

Manual OFF phases belong in the decoded output table itself. Offline quintic
ramps reduce and restore those weights around exact-zero plateaus; there is no
runtime manual-output overwrite. Actual geometry still prevents forces at
unavailable vertices. Smooth ramps change the feasible force bounds near
touchdown, so their physical result must be replayed rather than assumed equal
to an abrupt output gate.

Each region uses up to four distinct mesh vertices, with each point's own body
Jacobian. Vertices must lie within 1 mm above the floor and 0.5 mm of the lowest
eligible patch. The live penetration allowance is 3 cm, as in the existing
physics gate; reference floor-clearance validation remains strict. Thirty
convex patches cover all 16 measured regions and preserve the reference's
floor-extreme vertices. Physics mass, inertia, transforms, torque limits and
collision material friction are checked before actuation.

The QP has 157 variables (29 accelerations and 128 force-ray coefficients),
six balance equalities and eight region normal-force caps among its
inequalities. It retains torque limits, posture/momentum objectives and
regularization. No stationary-contact equality or full Coulomb slip law is
imposed. Region wrenches are about the robot CoM.

Actual ground-filtered normal and friction forces are measured across all 16
regions. Decoded OFF outputs do not disable simulator collisions or clamp
collision impulses. The selected rollout has brief incidental arm and right-foot contacts. The
previous run also loaded the unmodeled left foot; see the diagnosis.

## Reference smoothing and contact labels

`scripts/smooth_crawl_reference.py` smooths positions, continuous-sign normalized
quaternions and joints with sigma=2.5 frames, then projects onto joint and floor
constraints. Joint jerk RMS decreases from 57.52 to 19.95; root rotation jerk
RMS decreases from 115.71 to 13.63. Maximum root translation change is 6.08 mm,
maximum rotation change 2.50 degrees, and RMS joint change 0.624 degrees.
Timing and the 599-frame length stay unchanged. Independent USD/MuJoCo geometry,
floor clearance, joint bounds, non-neighbor mesh intersections and static
support are checked again after smoothing.

The current geometric annotation intervals are inclusive, zero-based:

| Region | Reference contact frames | Reference frames ON | Positive output samples / 2396 |
| --- | --- | ---: | ---: |
| Left hand | 0–598 | 599 | 2291 |
| Right hand | 0–598 | 599 | 2307 |
| Left knee | 0–598 | 599 | 2396 |
| Right knee | 0–598 | 599 | 2396 |
| Left forearm | 68–72, 511–516 | 11 | 57 |
| Right forearm | None | 0 | 0 |
| Right foot | 496–500, 542–552 | 16 | 64 |
| Left upper arm | None | 0 | 0 |

Thus six regions touch in the reference. The manual output phases also account
for measured contacts in preceding physical trials; they need not equal the
reference annotation table. The independent label audit recomputes all 599
frames and verifies CSV/NPZ agreement and the archived smoothing baseline hash.

Necessary dynamic feasibility uses the reference's finite-difference
acceleration, unilateral forces, friction bounds and actuator torque limits.
All 599 frames pass after smoothing, versus 567 before. Requiring sliding
friction to do no positive work passes 580/599 frames, versus 484 before.
These are instantaneous necessary checks, not a full contact-dynamics proof.
The four-support repair still flattens some source recovery lifts.

## Replay and verification

Use the `hybridmimic` environment from the repository root:

```bash
PYTHONNOUSERSITE=1 python scripts/rsl_rl/eval_crawl_force.py \
  --device cpu --headless --friction 0.2 --root-error-limit 0.5 \
  --vertical-momentum-weight 10 --momentum-kv 25 \
  --record-contact-substeps \
  --manual-outputs eval_data/reference_crawl_forward/force/planned_labels.npz \
  --output outputs/crawl_force_replay \
  --kit_args '--/app/userConfigPath=/tmp/crawl_force_physics.config.json --/log/file=/tmp/crawl_force_replay.kit.log --/app/enableCrashReporter=false'

MUJOCO_GL=osmesa MPLCONFIGDIR=/tmp/crawl_mpl \
python scripts/render_crawl_force_rollout.py outputs/crawl_force_replay/diagnostics.npz
```

Defaults match the selected profile: posture gains 1600/80, posture weight 3,
CoM position gain 160, linear momentum gain 25 and vertical momentum weight 10.
Use `--vertical-momentum-weight 1` to reproduce the previous bouncing profile. The full profile is saved.
The physics timestep is about 2.086 ms; the QP updates every four steps.
The runtime scaled optimality tolerance is 1e-6 and the separate physical
constraint tolerance is 2e-5. Only joint-effort commands are applied after
initialization; actuator PD, armature and joint friction are zero.

`reproduction.json` records smoothing, annotation, force-planning and replay
commands, including the required pxr environment for geometric checks.
`python scripts/crawl_manual_nn_outputs.py` exports JSON to
`outputs/crawl_manual_export/planned_labels.npz/.csv`, leaving the selected case
intact. Trial outputs and alternative schedules belong under `outputs/`.

The reusable solver API is:

```python
geometry = CrawlContactGeometry(annotation_path, device)
solver = WBCCrawlForceQP(cfg, geometry, friction=0.2)
result = solver.solve(state, reference, raw_actions, torque_limits)
# raw_actions: float64 [batch, 8]; no reference-label argument
```

Current targeted regressions: 55 passed, 1 CUDA skip; one known pre-existing
OSQP torque-parity test remains failing (0.542 Nm mismatch before these edits).
Zero-weight and geometry checks pass. The earlier strict offline snapshot
audits remain unresolved and are historical; they do not certify this new
manual table. CPU execution is supported and was used for this complete run.
