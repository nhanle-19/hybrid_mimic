# WBC-FORCE: learned normal-force inequalities

WBC-FORCE is a separate task derived from [WBC-ACC](wbc_acc_controller.md).
It shares the robot dynamics, physical foot colliders, physics, observations and PPO
hyperparameters, with fixed unit exploration noise, its own six-term reward and clip-completion boundary. The existing WBC-ACC
policy retains its fourteen-output interface and acceleration objectives.

## Policy and QP

The NN outputs eight values, ordered left sole corners 0–3 then right sole
corners 0–3. Each set uses the stored `sole_vertices` order: `(x-, y-)`,
`(x-, y+)`, `(x+, y-)`, `(x+, y+)` in the foot-link frame.

```
mean_i = actor_output_i
action_i ~ Normal(mean_i, 1.0)
c_i = clip((sigmoid(action_i) - 0.10) / 0.90, 0, 1)
F_max = robot_mass * (-gravity_z) * point_force_max_body_weights
F_limit_i = c_i * F_max
0 <= Fz_i = sum(four_friction_rays_at_corner_i) <= F_limit_i
```

The Gaussian mean is the unrestricted linear actor output in training,
deterministic playback, and ONNX/JIT export. Each action's standard deviation is
fixed at 1.0 in a checkpointed buffer that PPO cannot optimize. The configured
Gaussian entropy bonus is therefore constant and contributes no policy gradient.
Raw actions at or below approximately -2.197225 give exactly zero capacity;
capacity approaches one asymptotically at the positive end. Zero raw output
gives 4/9 capacity (137.83860 N per corner). PPO stores and scores raw Gaussian
actions, and rewards and diagnostics use the same /0.90 mapping as the QP.

Manual `--manual_point_caps` accepts eight fractions, including exact 0 and 1.
`--manual_support LEFT RIGHT` applies each fraction uniformly to that foot's
four corners. It no longer specifies a total-foot cap.
The encoder uses -3 for zero and +40 for floating-point saturation to one in
both float32 and float64. Interior fractions `c` encode as
`logit(0.10 + 0.90*c)`. Start a fresh training run for `FixedStdActorCritic`;
archived bounded-mean checkpoints require the retained `BoundedMeanActorCritic`
class for playback. Checkpoints with non-unit noise are rejected by the new
policy. Eight-corner manual annotations must declare the /0.90 mapping and
encode their fractions accordingly. The separate legacy region/crawl and AUTH
controllers retain their /0.80 mapping and existing manual schedules.

`point_force_max_body_weights` defaults to `1.0`. The 31.61435756 kg robot model
and gravity 9.81 m/s² give **310.13684767 N per corner**. One corner can therefore
be assigned enough capacity to support one robot body weight. All four caps at
maximum allow up to 1,240.54739 N per foot; there is no additional aggregate
foot-force bound. The legacy `contact_force_max` field is used by the separate
region-cap/crawl controller and does not affect this eight-output foot policy.
Normal force is world-Z force
on the robot from the horizontal ground. The NN selects upper bounds; the QP
chooses the predicted force within those bounds. These are hard inequalities in
the QP, not force-target penalties or desired force commands.

The foot-acceleration tracking objectives are omitted entirely. No stationary
foot-acceleration equalities are added. Momentum, joint posture, pelvis
orientation, and force/acceleration regularization remain. The separate swing
task must have zero weight. The inherited contact-motion weight/gain config
fields are unused by WBC-FORCE.

Friction-cone approximation, nonnegative point forces, floating-base dynamics,
and joint torque bounds remain active. Two-foot FORCE keeps all four modeled
corners of each foot available to the QP during training and evaluation.
There is no actual- or reference-geometry contact mask: the eight learned caps
alone control whether each corner may supply predicted force. A zero cap gives
exactly zero force at that point; all four caps zero give exactly zero foot
wrench. A positive cap can permit predicted force
even for an airborne foot; learning to unload it is the policy's responsibility.

The controller uses forward kinematics and the stored sole corners to compute
force moment arms and point Jacobians. The QP chooses one three-dimensional
force per corner, represented by four nonnegative friction rays: 12 physical
force components and 16 ray weights per foot. Each corner has its own cap;
capacity cannot move between corners. Their distribution can shift the center of pressure;
the wrench moment is the sum of the corner moments `r_corner × F_corner`.
The two-foot QP has 61 variables and 86 inequality rows before eliminating
the six balance equalities.

These robot-model calculations do not classify contact or compare corners with
a floor. Ground-gap and contact-patch
height tolerances therefore do not affect two-foot FORCE force availability.
ACC and AUTHORITY retain their geometry gates; the separate manual eight-region
crawl controller retains its existing contact handling.

The QP force limit constrains the prediction; PhysX determines the physical
contact response to the applied motor torques and may produce different measured
forces. Diagnostics record both plus the learned upper bounds.

The four-corner model without a geometry gate was restored after the midpoint
experiment below, then changed from two foot-total caps to eight corner caps.
All earlier two-output and midpoint results remain historical evidence and do
not validate the current policy. New diagnostics and ONNX exports identify
`qp_contact_model=sole_corners`, `force_cap_scope=per_sole_vertex`, and eight
actions. Diagnostics store each corner's force and cap as well as the summed
foot wrench. The evaluation suite rejects results from previous cap layouts.

The 2026-11-05 fixed-midpoint manual retry used the archived caps, posture gains
1600/40, 600 N maximum per foot, seed 42, and the same four 500-frame references
as `outputs/force_manual_slide_stand_sway_20261105`. Each replay started at frame
zero with the existing torso guards. The experimental results are in
[`outputs/force_manual_midpoint_20261105`](../outputs/force_manual_midpoint_20261105/report.html).

| Manual case | Previous four-corner frames | Fixed-midpoint frames |
| --- | ---: | ---: |
| Tilted sway | 500/500 | 52/500 |
| Standing | 114/500 | 47/500 |
| Narrow sway | 56/500 | 43/500 |
| Slide | 500/500 | 54/500 |

All four midpoint replays stopped on `anchor_pos`, with zero QP failures. The
recorded predicted wrench moments equal `r_midpoint × F` to floating-point
precision. These unchanged manual schedules performed worse with the midpoint;
this retry does not test learned adaptation or establish that lower caps would
recover the motions. Isaac Sim exited with signal 11 after saving each run and
printing its shutdown completion message; diagnostics and rendered videos were
validated separately. The controller and evaluation regression suite passed
150 tests, including CUDA cases.

The replay and training measurements recorded below predate removal of the
two-foot FORCE geometry gate. They remain evidence about that historical
controller, not validation of the current ungated controller. Existing gated
checkpoints require new evaluation; train a fresh policy to assess learning
without the gate. Lower outputs for unfavorable contact shapes remain a
learning hypothesis, not an automatic shape-based cap reduction. The recorded
results and artifacts are preserved.

The flat-foot slide diagnostic also tested whether the remaining costs were
too weak. With 800 N caps on both feet and friction 0.6, eight 500-step physics
trials compared posture weights from 0.1 to 100, stronger posture feedback
gains, and diagnostic direct left-foot XY tracking costs. Every trial stayed
upright but moved the foot less than 0.004 mm against a 40 mm target. The
largest feedback gain produced predicted sideways foot acceleration up to
11.7 m/s² while the physical foot remained planted. These tests point to the
missing coupling between contact motion and sticking/sliding friction, rather
than insufficient tracking weight alone. A force inside the friction pyramid
does not by itself describe the friction reaction during sliding. Only motor
torques are applied to PhysX, which determines the actual contact response.
These diagnostic trial files were removed during the evaluation cleanup.
They used the historical geometry-gated controller described above.

A subsequent reference shifted the root 8 cm right, lowered it 2 cm, and added
a 5-degree rightward roll while retaining the same flat-foot slide path.
With this feasible support pose, a motion-phased left cap of zero and right
cap of 800 N produced controlled sliding at the original friction 0.6. The
independent 500-step replay moved the left foot 36.93 mm against the 40 mm
target, with 4.69 mm XYZ tracking RMS and no resets, QP failures, or torque
saturation. The measured left load averaged 0.17 N during sliding. See
`eval_data/left_foot_slide/force/force_validation.json` and the named
`force_slide_schedule.npz`. The superseded replay directory was removed;
the final refined replay below is retained. This starts from the shifted pose; it does not
validate a transfer from centred standing or the earlier upright reference.

Refining the posture feedback on that same reference and identical manual
force labels improved the independent replay to 0.416 mm XYZ RMS (from
4.686 mm) and 39.59 mm travel. The fixture uses position gain 1600 and velocity
gain 40, with posture weight still 0.1. All 500 steps completed without resets,
solver failures, or torque saturation; the maximum sole-height error was
0.021 mm. These gains are now the shared FORCE defaults and satisfy the
refined schedule metadata without gain overrides; see
`eval_data/left_foot_slide/force/force_refined_validation.json` and the
README replay command. This improvement follows successful unloading; it does
not reverse the earlier result for the loaded, upright sliding fixture.

## Learning

The actor receives **590 values** in two blocks:

- Reference frames at offsets `[0, 1, 3, 5, 10]`: 295 values.
- Actual-state frames at offsets `[-4, -3, -2, -1, 0]`: 295 values, oldest first.

For each frame in either block, the input order is joint positions (23), joint
velocities (23), root world position (3), root WXYZ quaternion (4), root world
linear velocity (3), and root world angular velocity (3). Frames are
concatenated in offset order; reference lookahead beyond the clip repeats its
last frame. The root is the configured anchor (`Trunk` for T1). Actual inputs
come from simulated joint states and body-link states, with the environment
origin subtracted from root position so actual and reference coordinates
match. Measured contact forces and previous actions are not included in the
actor. The robot model derives link kinematics, CoM and momentum for the QP.

Isaac Lab's per-term history buffer samples actual state once per policy step
(20 ms, not each 2 ms physics substep). Its five frames span 80 ms of history
through the current state. Reset clears only the affected environment and
fills missing history with its first new state. Recomputing observations for
inspection does not advance history. Only the actual-state term has history;
the reference lookahead window is not itself stacked across previous steps.

The critic receives both windows plus its existing privileged robot state and
previous action, for **867 inputs**. Actual-state feedback now enters both
the policy and the WBC. FORCE uses six reward terms:

```text
r_t = (100 / N) * [alive_new_frame
                  - 0.5 * mean_i(swing_weight[t,i] * c[t,i]^2)
                  - 0.5 * mean_i((c[t,i] - c[t-1,i])^2)
                  - 0.4 * mean_i(contact_weight[t,i] * indicator(c[t,i] == 0))
                  - 0.08 * mean_i(max(c[t,i] - F_used[t,i] / F_max[i], 0)^2)
                  - 0.5 * (1 - exp(-mean_j((q_actual[t,j] - q_reference[t,j])^2) / 0.2^2))]
contact_weight[t,i] = 1 - swing_weight[t,i]
```

`N` is the number of reference frames; `c` is the decoded cap fraction in
`[0, 1]`, not the raw action or measured force. The first term adds 100/N
percentage points for each newly executed frame without a failure. A full
undiscounted clip earns 100 points before penalties. Starting at frame 200
does not credit the skipped prefix. Repeated endpoint frames and internal
reference wraps earn no extra progress. TensorBoard's
`WBCForce/Episode/executed_clip_percent` reports the earned percentage of the
whole clip; `Episode/suffix_completed_percent` instead divides by the frames
available at that episode's sampled start. PPO still uses discounted returns.

The cap index `i` ranges over all eight corners. Swing weights come from each
prepared reference corner's clearance
above the ground: zero at or below 1 mm, a cubic smoothstep between 1 and 5 mm,
and one at or above 5 mm. A grounded heel can retain capacity while the elevated
toe corners are penalized for nonzero caps.
This non-contact penalty teaches the policy to remove force capacity during
swing. Reference geometry is training supervision only: it never replaces NN
outputs or masks the QP with a contact label. Actual geometry also does not
remove support from the two-foot FORCE QP.

Completion, swing-cap, and action-change terms retain their weights and contact timing. Optional
`env.rewards.swing_force.params.unload_lead_time` (seconds, default **0**) adds
a linearly discounted lookahead to this same cap penalty, allowing an unloading
lead-in before reference liftoff. It never shifts the end of swing later or
wraps the end of a clip. For a lead of `K` policy steps, the weight is
`max_{k=0..K}((1-k/(K+1))*swing_weight[min(t+k,N-1)])`.
This is a contact-timing experiment, not an actual-foot tracking reward.
The QP continues to command tracking, while the joint penalty below rewards
tracking outcomes. No clearance, foot-position, or foot-velocity reward is added.

The false-zero penalty complements the swing-cap penalty: a zero decoded cap
costs reference contact weight, with weight **-0.4** (reduced from -0.5). Reference contact
weight is one at or below 1 mm corner clearance, decreases smoothly to zero at
5 mm, and is evaluated independently for each reference corner. Actual foot height, load,
and orientation do not change this weight. This term uses current reference
contact with no unloading lead-in.

False-zero means an **exact zero after decoding**. A raw action of zero decodes
to 0.5 and is free; any positive decoded cap is also free. This term encourages
available support without prescribing a minimum stance load. A zero cap on one
fully contacting foot throughout a clip costs 20 points; on both feet it costs
40 points. Tune `env.rewards.false_zero.weight` and its `clearance_start` /
`clearance_full` parameters in metres. Because costs are normalized by the full
clip, short contact or swing intervals contribute only part of the total return.

The change penalty uses consecutive decoded actions and skips the first action
after each environment reset. It encourages smooth changes without imposing
a hard slew limit. Changes between two raw values in the same saturated region
incur no penalty. The starting weights are tunable with
`env.rewards.swing_force.weight` and `env.rewards.action_change.weight`.
All six implementations account for RewardManager's multiplication by the
control timestep, so the formula above is the final per-step reward.

The cap-gap penalty charges unused normal-force capacity at all eight corners in all
reference phases. `F_used` is the current policy step's QP-selected vertical
force at that corner, available even when diagnostic recording is disabled.
Each corner's positive gap is divided by its maximum (`mass × gravity` by
default), squared, and averaged across the eight corners. A half-maximum gap
therefore has normalized squared cost 0.25.
The weight is **-0.08** (reduced from -0.1), tunable with
`env.rewards.force_cap_gap.weight`. Persistent half-maximum gaps on both feet
cost 2 points per full clip; the maximum cost is 8 points. Using the entire
cap incurs no gap penalty, and zero capacity has no gap cost.

Negative force roundoff is clamped to zero, and force above the cap incurs
zero gap cost. A missing or failed QP solve counts its capacity as unused;
nonfinite force candidates also count as zero used force. QP failure still
terminates the episode. PhysX-measured loads remain separate diagnostics.

The joint-tracking penalty compares all actual joint positions with the current
prepared reference in simulator joint order. Its bounded cost is
`1 - exp(-mean((q_actual - q_reference)^2) / error_scale^2)`, with
`error_scale=0.2` radians and weight **-0.5**. Exact tracking costs zero, and
the maximum full-clip cost is 50 points. A persistent joint RMS error of 0.1 rad
costs 11.06 points; 0.2 rad costs 31.61 points. Averaging over joints keeps its
scale independent of joint count. Tune `env.rewards.joint_tracking.weight`
and `env.rewards.joint_tracking.params.error_scale`.

The contact-orientation reward remains removed. The penalties still charge failing
steps before reset and use the current clip's frame count in mixed-motion batches.
Actor/critic inputs are unchanged; existing checkpoints acquire these objectives
only through further training. The cap-change weight is now **-0.5** (10 times
its previous magnitude), and the joint-tracking weight is **-0.5** (twice its
previous magnitude). These stronger penalties have not been validated by a
training run.

Training prioritizes balanced, feasible motion. Limb tracking error does not
terminate an episode. The existing coarse torso guards remain:
reference-relative torso height error above 0.25 m, or a projected-gravity
vertical-component difference above 0.8. QP failures also terminate. These
guards are proxies for loss of balance, not a requirement for precise imitation.
Adaptive phase starts remain enabled for 80% of resets; the remaining 20% start
at frame zero (`env.commands.motion.start_from_beginning_probability=0.2`).
The first sampled frame is executed after reset, including frame zero, rather
than skipped by the command update. FORCE no longer randomizes the separate
episode timeout counter at startup.
Executing the last reference frame ends the training attempt as a timeout
before the command can silently reset the robot for another clip. A sampled
suffix therefore earns only its executed fraction; evaluate from frame zero
to measure full-motion completion. Scheduled manual and toy-standing FORCE
replays use the same torso/QP guards without a limb-height cutoff. These rewards do not use
the hand-designed action tables as training targets.

### Contact-learning logs

The FORCE runner writes the following to TensorBoard/W&B once per PPO rollout.
Rates use accumulated numerators and denominators, not averages of unequal reset
batches. Conditional metrics are omitted when no eligible samples occur;
terminal output shows `n/a`. Counts are provided alongside sparse metrics.
Contact metrics use sampled training actions; deterministic policy behavior
still requires a separate evaluation.

| Metric prefix (under `WBCForce/`) | Meaning |
| --- | --- |
| `FullMotion/episode_count`, `success_percent`, `completed_percent` | Ended training episodes whose first executed frame was zero. Success means surviving to the last frame, not accurate motion tracking. These use sampled training actions, not deterministic evaluation. |
| `Episode/count`, `suffix_success_percent`, `suffix_completed_percent`, `executed_clip_percent` | Separate successful short suffixes from the fraction of a whole clip actually executed. Failure wins over simultaneous clip completion. |
| `Sampling/start_count`, `start_phase_percent`, `start_frame_zero_percent`, `start_final_sixth_percent` | Actual starts observed during this rollout, including ongoing episodes. |
| `Contact/{left,right}_swing_samples`, `swing_zero_cap_percent`, `swing_cap_mean` | Exact-zero coverage and mean capacity where reference sole clearance is at least 5 mm. |
| `Contact/{left,right}_liftoff_count`, `liftoff_zero_cap_percent` | Cap at the reference's first crossing above 1 mm; starts already inside swing are excluded. |
| `Contact/{left,right}_zero_hold_ms` | Mean uninterrupted exact-zero duration within reference swing, reported when a run ends or its episode resets. |
| `Contact/{left,right}_swing_measured_fz_N` | Actual PhysX vertical load during reference swing; a predicted zero cap does not guarantee zero measured force. |
| `Caps/{left,right}_stance_zero_percent`, `absolute_change`, `mean`; `Caps/both_zero_percent` | Stance unloading, chattering, mean cap, and simultaneous zero caps. |
| `Failure/{anchor_pos,anchor_ori,qp_infeasible}_percent` | Percent of ended episodes with each failure flag; overlapping flags are not mutually exclusive. |
| `Caps/{left,right}_unused_force_N` | Mean positive difference between decoded cap and current QP-used normal force, in newtons. Failed solves count their capacity as unused. |
| `Reward/{completion,swing_cap,cap_change,false_zero,force_cap_gap,joint_tracking}_points_per_step` | Weighted reward contributions in actual PPO reward units per policy step, without maximum-episode-duration normalization. |
| `Tracking/joint_position_rmse_rad` | Mean per-environment RMS joint-position error against the current prepared reference, in radians. |
| `Policy/{left,right}_noise_std` | Per-foot raw Gaussian exploration scale. |

The terminal shows PPO losses plus full-motion/suffix success and contact
coverage. The old FORCE reset-time averages and per-axis wrench console dump
are replaced by these metrics. `Train/mean_episode_length` and `mean_reward`
remain standard RSL-RL curves; inspect them together with start-phase and
full-motion metrics. Physical wrench diagnostics remain available in evaluation.

Defaults: 1,024 environments, 500 Hz physics, 50 Hz policy/QP, and 30,000 PPO
iterations. Each QP torque is held for ten physics steps. Solver failures
terminate the affected episode and apply bounded damping until reset.

The task has its own experiment folder, `logs/rsl_rl/t1_wbc_force/`.
Both two-output FORCE and WBC-ACC checkpoints are incompatible with this
eight-output policy. This layout requires fresh FORCE training: earlier reference-only checkpoints
with 236 or 1,432 actor inputs cannot be resumed into this layout.
The ACC and AUTHORITY observation layouts retain their existing dimensions.

The shared controller defaults are posture gains 1600/40 with weight 0.1,
CoM position gain 40, linear/angular momentum gains 10/15, momentum weights
`[1, 1, 1, 10, 10, 10]`, and pelvis gains 60/12 with weight 5. Two-foot training
and eight-region crawl replay read the same FORCE configuration. Motion-specific
gain overrides are unnecessary; their different contact handling and physical
limits are separate from these gains. See the
[validation and training command](../README.md#shared-force-defaults-measured-scope).

## Commands

Train with the [default server command](../README.md#train-wbc-force), which
sets W&B authentication and CUDA visibility and uses the configured controller
and PPO defaults. Change `--motion_file` and `--run_name` for another motion.

For a short training check, use `--num_envs 32 --max_iterations 2`.

Manual standing with all eight corner caps at maximum:

```bash
python scripts/rsl_rl/eval_wbc.py --task Standing-WBC-FORCE-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --qp_only --manual_point_caps 1 1 1 1 1 1 1 1 \
  --num_envs 1 --steps 200 --device cuda:0 --headless
```

Evaluate a trained policy:

```bash
python scripts/rsl_rl/eval_wbc.py --task Tracking-WBC-FORCE-T1-Eval-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --load_run RUN_DIRECTORY --checkpoint model_29999.pt \
  --num_envs 1 --steps 500 --device cuda:0 --headless

python scripts/plot_wbc.py --input logs/rsl_rl/t1_wbc_force/RUN_DIRECTORY/diagnostics.npz
```

Learned-policy evaluation saves videos to the checkpoint run's `videos/play/`
folder and diagnostics to its `diagnostics.npz`, matching the playback layout.
QP-only evaluation uses `eval_data/wbc_force/` because it has no checkpoint.
Use `--video_dir` or `--output` to choose a different destination explicitly.

Numerical tests cover learned-cap enforcement, zero-cap support removal,
friction and torque bounds, inverse dynamics, CPU solver agreement, absence
of foot-reference objectives, the action interface, and exported policy metadata:

```bash
python -m pytest tests -q
```

These checks do not establish closed-loop tracking performance. Validate with
the simulator before a full training run.
