# WBC-FORCE: learned normal-force inequalities

WBC-FORCE is a separate task derived from [WBC-ACC](wbc_acc_controller.md).
It shares the robot dynamics, support geometry, physics, observations, PPO
configuration, tracking rewards, and termination rules. The existing WBC-ACC
policy retains its fourteen-output interface and acceleration objectives.

## Policy and QP

The NN outputs two values, ordered left foot then right foot:

```
c_i = clip(action_i, 0, 1)
F_limit_i = c_i * contact_force_max_i
0 <= Fz_i = sum(rho_i) <= F_limit_i
```

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

## Learning

The actor still receives only reference motion at offsets 0, 1, 5, and 10 frames;
the critic receives privileged robot state. The WBC supplies actual-state
feedback. PPO learns force-cap timing from the existing motion-tracking,
survival, slip, collision, joint-limit, and action-smoothness rewards. The
action-smoothness penalty is the mean squared change in the two clipped caps.

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
