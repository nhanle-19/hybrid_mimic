# Crawl force diagnosis

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

The following sections preserve the earlier tuning investigation. Their numerical
results and exact-repeat claims refer to the pre-correction table archived in
`outputs/manual_contact_fix_20261023/before/crawl_force/`.

## Softer hand-contact transitions

The subsequent friction sweep retains **0.2**. Values 0.15, 0.225, 0.25,
0.275, 0.30 and 0.35 also complete all 599 frames with zero failed QPs, holding
the reference, raw NN outputs and vertical-weight-10 profile fixed. Higher
friction reduces forward drift but increases joint tracking error. Full-rate
force measurements reveal short peaks missed at the original QP sampling rate:

| Friction | Peak root error | Forward progress | Peak joint RMSE | Peak combined arm load, every physics step |
| --- | ---: | ---: | ---: | ---: |
| **0.20, retained** | 38.3 cm | 1.203 m | 3.97 degrees | **412 N** |
| 0.275 | 35.7 cm | 1.232 m | 4.82 degrees | 533 N |
| 0.30 | 34.9 cm | 1.248 m | 5.44 degrees | 537 N |

The 0.275 run's apparent 194 N arm peak at QP sampling rises to 533 N when
all physics steps are measured. Its 20 ms average arm peak is 176 N versus
164 N at 0.2. The modest tracking gain does not justify changing the setting
when softer contacts are the priority. Lowering friction to 0.15 worsens peak
root error to 40.5 cm and reduces forward progress to 1.182 m.

The selected 0.2 run is unchanged. Its full-rate hand-only peak is **309 N**;
the **173 N** figure below is specifically the QP-rate sampled peak. All 9,584
physics steps are now recorded through `--record-contact-substeps`. Each of
the three full-rate reruns exactly matches its original trajectory, torques
and QP-rate force samples, proving that added measurement did not change
control. `force/friction_tuning.json` contains all seven comparisons and full-rate
checks; raw trials and analysis scripts are in
`outputs/crawl_friction_tuning_20261023/`.

The previous full-duration run rose 3.87 cm above the reference and dropped at
up to 0.348 m/s. Increasing the QP's vertical momentum weight from 1 to 10
addresses this while retaining the reference, all eight manual output channels,
friction 0.2, posture gains and the 0.5 m root-error cutoff.

| Full-motion trial | Height overshoot | Peak downward speed | Peak sampled hand force |
| --- | ---: | ---: | ---: |
| Previous profile | 3.87 cm | 0.348 m/s | 717 N |
| Hand caps reduced by 15% | 3.77 cm | 0.357 m/s | 531 N |
| Vertical weight 10, selected | **0.41 cm** | **0.093 m/s** | **173 N** |
| Vertical weight 10, momentum damping 40 | 0.39 cm | 0.088 m/s | 175 N |

All four complete 599 frames without QP failures. The selected change reduces
height-error RMS by 88%, peak downward speed by 73% and peak hand load by 76%.
The total sampled ground-force peak falls from 1,097 to 623 N. We retained the
single weight change: extra damping brings little improvement and raises the
99th percentile total ground load relative to weight 10 alone. Measured peaks
are sampled after QP updates and do not establish the peak between samples.

The root travels 1.203 m forward versus 1.216 m previously, a 1.3 cm reduction.
Peak joint RMSE falls from 9.00 to 3.97 degrees and peak root error from 0.394 to
0.383 m. This controlled comparison supports insufficient vertical tracking
priority as the main contributor to the observed bounce; lowering hand limits
alone did not remove it. The body remains physically torque-driven, with no
pose resets, external support, or altered collision restitution.

Evidence and all three candidates are in `outputs/crawl_soft_landing_20261023/`.
The preceding complete case is backed up in `before_tuning/case/`. The standard
case contains the new actual video and a synchronized previous/tuned comparison.
Old commands without a vertical-weight flag need `--vertical-momentum-weight 1`
to reproduce the old profile; current default is 10.

## Earlier reference and manual-output tuning

The old reference was jittery and imposed continuous hand/knee grounding.
Smoothing with sigma=2.5 frames and reprojection reduces joint jerk RMS by 65%
and root angular jerk RMS by 88%. Root position changes by at most 6.08 mm,
root orientation by 2.50 degrees, and joints by 0.624 degrees RMS. The cadence
and duration do not change. Independent geometry validation passes every frame.

At mu=0.2, the reference acceleration admits bounded unilateral contact forces
and actuator torques in 599/599 frames, up from 567/599. Adding non-positive
sliding-friction power passes 580/599, up from 484/599. The remaining 19 frames
are listed in `dynamic_feasibility.json`; the reference is not a fully feasible
Coulomb-contact trajectory. These checks use the actual finite-difference
reference acceleration, rather than letting a tracking QP choose another one.

Measured contacts, force-weighted material-point velocities and root drift were
reviewed before changing outputs. The smoothed all-on baseline still loses
forward progress. Simply reducing every cap performs worse. The selected
schedule reduces caps by up to 65% on forward-sliding contacts, with extra
capacity on slow or backward-moving contacts. It keeps the initial stable
stance and blends into changing caps. A second revision uses contacts observed
in the first, longer physical rollout. These are manually designed output
values, not force measurements or a trained policy. The table is fixed before
replay and is not updated using future data during execution.

All new comparisons request the full motion with the same 0.5 m root-error
cutoff and existing pose gains:

| Trial | Friction | Duration | Stop |
| --- | ---: | ---: | --- |
| Smoothed reference, initial caps | 0.2 | 7.534 s | Root error |
| Narrow load-based caps | 0.2 | 6.884 s | Root error |
| Wider load-based caps | 0.2 | 6.717 s | Root error |
| Push/release revision 1 | 0.2 | 15.478 s | Root error |
| Push/release revision 1 | 0.25 | 15.002 s | Root error |
| Push/release revision 1 | 0.3 | 14.894 s | Root error |
| Push/release revision 2 | 0.2 | **19.992 s** | **Clip completed** |

No trial has a failed live QP. Raising friction above 0.2 did not improve the
first revision. The earlier full-duration run moves 1.216 m forward versus about 1.543 m in
the reference. Its final root-position error is approximately
(-0.327, -0.010, +0.036) m. Peak joint RMSE is 9.00 degrees and peak root-angle
error is 13.39 degrees. Stronger tracking alone was not the change responsible
for the improvement; the established 1600/80 posture gains are retained.

The small hand-anchoring candidate under `outputs/crawl_smooth_tuning_20261023/anchored/`
was not selected or physically tested. The successful run uses the smoothed
reference without that additional edit.

## Contact interpretation

The current reference has six distinct touching regions: both hands, both knees,
left forearm and right foot. Right forearm and left upper arm have no geometric
reference contacts after smoothing. Eight NN channels remain available.
All reference labels were recomputed and independently checked over all 599
frames; visual inspection samples the motion rather than claiming every image
was manually inspected.

The tuned actual rollout differs from the reference. Measured normal loads
above 5 N occur on seven regions: both hands, both knees, left forearm, left
upper arm and right foot. Left upper arm has two such control updates, peaking
at 24.5 N. Left foot does not exceed 5 N in the tuned run. It is not represented
in the eight-channel QP; the previous bouncing run had 157 loaded updates on
that foot, peaking at 25.7 N. No measured torso/head/thigh/shin load exceeds 5 N.
This is an incidental-contact limitation, not a claim that all physical contact
has been planned correctly.

The selected output table also enables short left-forearm, right-foot and
left-upper-arm phases observed in preceding trials. Right forearm remains OFF.
The actual geometry gate and decoded zero outputs produce exactly zero
predicted force when disabled. They cannot suppress a physical collision.
The current right-hand force peaks at 173 N, versus 717 N previously.
The maximum nominal predicted cap remains 600 N; collision impulses are not
clamped by NN values.

The QP bounds friction but does not enforce kinetic friction opposing slip,
stationary contact acceleration, or full normal-contact complementarity.
Predicted and measured forces therefore differ. Force redistribution helps
this particular motion but does not repair the contact dynamics formulation.

The source audit still finds a lost planted-hand phase: at 2.870–3.237 s the
human right finger marker moves only 2.19 mm in XY, raw GMR's fixed hand point
moves 53.77 mm, and the smoothed robot reference moves 73.89 mm. Marker and robot
points differ in scale and anatomy, so this cannot identify force or friction.
The four-support repair also flattens source hand recovery lifts. Smoothing
improves regularity; it does not reconstruct those human support transitions.

## Verification scope and numerical limitation

The independent replay exactly matches the selected trial's complete physics
trajectory, manual outputs, joint torques, predicted forces and measured forces.
Both videos decode completely: 599 reference frames and 600 recorded-state
comparison frames. `force/review_frames.png` samples 2, 6, 10, 14, 18 and 20 s.
The current geometric label audit and executed force-gate audit pass.
Targeted tests: 69 passed, 6 CUDA skips; the control tests were also rerun after
isolating the synthetic fixture's action array (11 passed).

The existing offline checks below use their default snapshot controller profile,
not the new vertical-weight-10 runtime profile. The old strict offline batched snapshot check at scaled tolerance 1e-12 fails
on some tuned-output snapshots. The reported inverse-dynamics error in that
failed check comes from deliberately zeroed fallback outputs, not the solver's
pre-fallback physical residual (below 7e-13). Sampled convergence stalls around
1e-9 to 1e-8; accepting the runtime tolerance of 1e-6 gives successful sampled
solves but does not meet the old 0.002 Nm cross-solver torque-parity criterion.
This limitation is retained in `outputs/crawl_smooth_tuning_20261023/selected_offline/`.
Independent OSQP snapshot audits also remain incomplete: 11 snapshots fail
at tolerance 1e-12 (`selected_osqp/`), and 4 fail at 1e-9
(`selected_osqp_1e9/`, frames 117, 291, 296, 484). Accepted outputs meet the
physical checks, but these are not all-frame solver passes. The live run uses
its established 1e-6 tolerance and has zero failed updates. No solver tolerance
or acceptance criterion was changed to obtain the completed physical run.

After the audit CLI and current metadata changes, the focused contact/control/
manual-output/mapping/FORCE tests were rerun: 42 passed, 1 CUDA skip. This is a
subset of the earlier 69-test run, not 42 additional distinct tests.

## Evidence and earlier work

The selected case is `eval_data/reference_crawl_forward/`. Its `validation.json`
links geometry, dynamic feasibility, replay, gates and numerical audits;
`reproduction.json` records inputs and commands. All raw new trials and before/
after contact evidence are in `outputs/crawl_smooth_tuning_20261023/`.

Earlier investigations are historical, using the unsmoothed reference:

- At the 0.15 m root cutoff, baseline friction 0.6 stopped at 3.463 s with
  right-knee lift; increasing body tracking or changing hand caps did not fix it.
- Stronger joint gains reduced angular error but did not restore knee support.
  Disabling separating above-floor predicted support removed that particular
  force inconsistency without completing the motion.
- Friction 0.1 and 0.02 improved joint angles while increasing body drift.
- With the preliminary 0.5 m cutoff, friction 0.6 / 0.2 / 0.1 reached
  5.857 / 7.618 / 8.068 seconds. These are superseded by the current mu=0.2 run.

Those trials are in `outputs/crawl_force_archive_20261023/` and
`outputs/crawl_preliminary_20261023/`. The earlier detailed diagnosis is retained
in `outputs/crawl_smooth_tuning_20261023/before_smoothing/docs/`.
