# WBC-FORCE: learned normal-force inequalities

WBC-FORCE is a separate task derived from [WBC-ACC](wbc_acc_controller.md).
It shares the robot dynamics, foot model, physics, observations and PPO
configuration, with its own three-term reward and clip-completion boundary. The existing WBC-ACC
policy retains its fourteen-output interface and acceleration objectives.

## Policy and QP

The NN outputs two values, ordered left foot then right foot:

```
c_i = clip((sigmoid(action_i) - 0.10) / 0.80, 0, 1)
F_limit_i = c_i * contact_force_max_i
0 <= Fz_i = sum(rho_i) <= F_limit_i
```

The actor outputs unconstrained Gaussian actions. The shared FORCE/AUTH mapping
has a 10% dead zone at each end: raw actions at or below approximately -2.197225
give exactly zero, and actions at or above +2.197225 give exactly one. The mapping
is continuous, with slope changes at those boundaries. Zero raw output still
gives 50% capacity (300 N). PPO stores and scores the raw Gaussian actions;
the mapping is applied only by the controller.

Manual `--manual_support` arguments remain fractions, including exact 0 and 1.
The encoder uses ordinary raw values -3 and +3 for those endpoints. Interior
fractions `c` encode as `logit(0.10 + 0.80*c)`. Existing plain-sigmoid checkpoints
have different action semantics and are not equivalent controllers under this
mapping; start a new training run for a clean comparison. Old manual schedules
must be migrated or rebuilt; see [the schedule guide](manual_wbc_schedule.md).

`contact_force_max` defaults to `(600, 600)` N. Normal force is world-Z force
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
There is no actual- or reference-geometry contact mask: the two learned caps
alone control whether a foot may supply predicted force. A zero cap gives
exactly zero predicted foot wrench. A positive cap can permit predicted force
even for an airborne foot; learning to unload it is the policy's responsibility.

The controller still uses forward kinematics and the stored sole corners to
compute force moment arms and Jacobians. These robot-model calculations do not
classify contact or compare corners with a floor. Ground-gap and contact-patch
height tolerances therefore do not affect two-foot FORCE force availability.
ACC and AUTHORITY retain their geometry gates; the separate manual eight-region
crawl controller retains its existing contact handling.

The QP force limit constrains the prediction; PhysX determines the physical
contact response to the applied motor torques and may produce different measured
forces. Diagnostics record both plus the learned upper bounds.

The replay and training measurements recorded below predate removal of the
two-foot FORCE geometry gate. They remain evidence about that historical
controller, not validation of the current ungated controller. Existing gated
checkpoints require new evaluation; train a fresh policy to assess learning
without the gate. The recorded results and artifacts are preserved.

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
previous action, for **861 inputs**. Actual-state feedback now enters both
the policy and the WBC. FORCE uses exactly three reward terms:

```text
r_t = (100 / N) * [alive_new_frame
                  - 0.5 * mean_i(swing_weight[t,i] * c[t,i]^2)
                  - 0.05 * mean_i((c[t,i] - c[t-1,i])^2)]
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

Swing weights come from the prepared reference's lowest foot collision vertex
above the ground: zero at or below 1 mm, a cubic smoothstep between 1 and 5 mm,
and one at or above 5 mm. A grounded toe or heel is not classified as airborne.
This non-contact penalty teaches the policy to remove force capacity during
swing. Reference geometry is training supervision only: it never replaces NN
outputs or masks the QP with a contact label. Actual geometry also does not
remove support from the two-foot FORCE QP.

The default timing and three reward weights are unchanged. Optional
`env.rewards.swing_force.params.unload_lead_time` (seconds, default **0**) adds
a linearly discounted lookahead to this same cap penalty, allowing an unloading
lead-in before reference liftoff. It never shifts the end of swing later or
wraps the end of a clip. For a lead of `K` policy steps, the weight is
`max_{k=0..K}((1-k/(K+1))*swing_weight[min(t+k,N-1)])`.
This is a contact-timing experiment, not an actual-foot tracking reward.
The QP continues to handle tracking. No clearance, foot-position, or
foot-velocity tracking reward is added.

The existing contact cost is one-sided: it penalizes a positive cap in reference
swing and has no direct preference for a positive cap in stance. Stance support
is learned through survival and the controller. Because costs are normalized by
the full clip, short swing intervals contribute only part of the total return.
On the downloaded iteration-6999 walk policy with the historical geometry gate,
deterministic replay earned 97.31/100 points: 100 progress points minus
2.13 swing-cap and 0.56 action-change
points, despite imperfect zero-cap timing. A flat return alone therefore does
not establish a good contact schedule.

The change penalty uses consecutive decoded actions and skips the first action
after each environment reset. It encourages smooth changes without imposing
a hard slew limit. Changes between two raw values in the same saturated region
incur no penalty. The starting weights are tunable with
`env.rewards.swing_force.weight` and `env.rewards.action_change.weight`.
All three implementations account for RewardManager's multiplication by the
control timestep, so the formula above is the final per-step reward.

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
| `Reward/{completion,swing_cap,cap_change}_points_per_step` | Weighted reward contributions in actual PPO reward units per policy step, without maximum-episode-duration normalization. |
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
WBC-ACC checkpoints are incompatible with this two-output policy.
This layout requires fresh FORCE training: earlier reference-only checkpoints
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

Manual standing with both maximum force caps:

```bash
python scripts/rsl_rl/eval_wbc.py --task Standing-WBC-FORCE-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --qp_only --manual_support 1 1 \
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
