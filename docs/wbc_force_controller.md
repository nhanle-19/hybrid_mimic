# WBC-FORCE: learned normal-force inequalities

WBC-FORCE is a separate task derived from [WBC-ACC](wbc_acc_controller.md).
It shares the robot dynamics, support geometry, physics, observations and PPO
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
chooses the actual force within those bounds. These are hard inequalities in
the QP, not force-target penalties or desired force commands.

The foot-acceleration tracking objectives are omitted entirely. No stationary
foot-acceleration equalities are added. Momentum, joint posture, pelvis
orientation, and force/acceleration regularization remain. The separate swing
task must have zero weight. The inherited contact-motion weight/gain config
fields are unused by WBC-FORCE.

Friction-cone approximation, nonnegative point forces, actual collider support
geometry, floating-base dynamics, and joint torque bounds remain active.
A zero force cap or unavailable support geometry gives exactly zero predicted
foot wrench. The NN cannot create support for an airborne foot. The QP force
limit constrains the prediction; PhysX may produce different measured forces,
especially at impacts. Diagnostics record both plus the learned upper bounds.

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
These diagnostic trial files were removed during the evaluation cleanup;
production defaults and the FORCE definition above remain unchanged.

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
0.021 mm. These gains are explicit evaluation overrides rather than new
production defaults. The refined schedule metadata requires them; see
`eval_data/left_foot_slide/force/force_refined_validation.json` and the
README replay command. This improvement follows successful unloading; it does
not reverse the earlier result for the loaded, upright sliding fixture.

## Learning

The actor receives only reference motion at offsets 0, 1, 5, and 10 frames;
the critic receives privileged robot state and the previous action. The WBC
supplies actual-state feedback. FORCE uses exactly three reward terms:

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
`WBCForce/completed_motion_percent` reports the earned percentage on episode
reset; the standard `Episode_Reward/*` values retain Isaac Lab's logging
normalization by maximum episode duration. PPO still uses discounted returns.

Swing weights come from the prepared reference's lowest foot collision vertex
above the ground: zero at or below 1 mm, a cubic smoothstep between 1 and 5 mm,
and one at or above 5 mm. A grounded toe or heel is not classified as airborne.
The reward penalizes a positive cap even if actual geometry already removes
that foot's support. It never replaces NN outputs or masks the QP with a
reference contact label.

The change penalty uses consecutive decoded actions and skips the first action
after each environment reset. It encourages smooth changes without imposing
a hard slew limit. Changes between two raw values in the same saturated region
incur no penalty. The starting weights are tunable with
`env.rewards.swing_force.weight` and `env.rewards.action_change.weight`.
All three implementations account for RewardManager's multiplication by the
control timestep, so the formula above is the final per-step reward.

Training retains adaptive phase starts and tracking/QP failure terminations.
Executing the last reference frame ends the training attempt as a timeout
before the command can silently reset the robot for another clip. A sampled
suffix therefore earns only its executed fraction; evaluate from frame zero
to measure full-motion completion. Existing manual and standing evaluation
tasks retain their playback termination behavior. These rewards do not use
the hand-designed action tables as training targets.

Defaults: 1,024 environments, 500 Hz physics, 50 Hz policy/QP, and 30,000 PPO
iterations. Each QP torque is held for ten physics steps. Solver failures
terminate the affected episode and apply bounded damping until reset.

The task has its own experiment folder, `logs/rsl_rl/t1_wbc_force/`.
WBC-ACC checkpoints are incompatible with this two-output policy.

## Commands

Train:

```bash
python scripts/rsl_rl/train.py --task Tracking-WBC-FORCE-T1-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --num_envs 1024 --device cuda:0 --headless \
  --logger tensorboard --run_name wbc_force_side_step
```

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

Numerical tests cover learned-cap enforcement, zero/unavailable support,
friction and torque bounds, inverse dynamics, CPU solver agreement, absence
of foot-reference objectives, the action interface, and exported policy metadata:

```bash
python -m pytest tests -q
```

These checks do not establish closed-loop tracking performance. Validate with
the simulator before a full training run.
