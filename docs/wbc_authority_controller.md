# WBC-AUTHORITY: directional contact authority

A separate task with twelve policy logits: six for the left foot, then six for
the right. Axis order is `x, y, z, rx, ry, rz`. Sigmoid maps each logit smoothly
to an authority fraction `c` in `[0, 1]`, without clipping. Zero logits give
50% authority. Extreme logits can still numerically saturate sigmoid; this
mapping alone does not guarantee successful training.

## Coupled QP formulation

All spatial quantities use world-aligned axes at the foot link origin, ordered
linear then angular. The contact acceleration is `a = J nu_dot + Jdot nu`.
For each geometrically available foot, the QP adds the soft objective

```
0.5 * sum_j contact_authority_weights[j] * c[j]^2 * a[j]^2
```

This is a weighted version of `||diag(c) a||^2`. It targets zero actual contact
acceleration, not the moving reference's acceleration. No hard no-slip or
stationary-foot equality is added. Reducing authority weakens that axis's
penalty. An airborne foot has no contact-acceleration penalty.

The same authority controls the allowable wrench, at the same foot origin:

```
|Fx| <= 360 cx       |Mx| <= 60 crx
|Fy| <= 360 cy       |My| <= 60 cry
0 <= Fz <= 600 cz    |Mz| <= 30 crz
```

Force units are N and moment units are N m. These are configurable initial
research bounds (`contact_wrench_max`), not measured contact properties.
`contact_authority_weights` defaults to six weights of 100; linear and angular
acceleration use different units, so these weights remain tuning parameters.

The bounds intersect the existing point-force friction pyramid and support
geometry. Positive normal forces, friction, available sole vertices, centroidal
balance, and joint torque limits remain enforced. The wrench bounds do not
replace friction or permit tensile normal force. Geometry can restrict the
wrench further; a small moment cap does not by itself specify an edge pivot.
Setting all authorities to zero removes support and the soft contact task.
Exact zero directional bounds are eliminated as equalities, including mixed
batches with different equality ranks. They are not relaxed to positive floors.
Very narrow nonzero bounds remain inequalities. The equilibrated optimality
tolerance is `1e-7`; physical residual acceptance remains `2e-5`. Failed solves
still terminate episodes rather than relaxing the requested constraints.

Momentum, joint posture, and pelvis tracking remain. The optional swing-foot
objective is disabled. These caps constrain QP predictions; measured PhysX
contact forces can differ, especially on impact.

## Training interface

This task retains the existing reference-only actor observations at frame
offsets `0, 1, 5, 10`. The critic and WBC receive actual state. It learns a
reference-conditioned authority schedule; the actor cannot react to measured
tracking errors. The observation architecture is unchanged for this comparison.

The action-rate reward uses changes in the twelve sigmoid-mapped authorities.
The Gaussian logit entropy bonus is disabled for this task: increasing logit
variance can saturate sigmoid without improving exploration in authority space.
PPO still samples stochastic actions. Training logs include the fraction of
authorities near the boundaries (below .01 or above .99).
The task has its own experiment folder, `logs/rsl_rl/t1_wbc_authority/`.
Train from scratch; two-output FORCE and fourteen-output ACC checkpoints are
incompatible. Defaults are 1,024 environments, 50 Hz policy/QP, 500 Hz physics,
and 30,000 training iterations.

```bash
conda activate hybridmimic
python scripts/rsl_rl/train.py --task Tracking-WBC-AUTHORITY-T1-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --num_envs 1024 --device cuda:0 --headless \
  --logger tensorboard --run_name wbc_authority_side_step
```

Start with a smoke run using `--num_envs 32 --max_iterations 2`.

```bash
python scripts/rsl_rl/eval_wbc.py --task Tracking-WBC-AUTHORITY-T1-Eval-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --load_run RUN_DIRECTORY --checkpoint model_29999.pt \
  --num_envs 1 --steps 500 --device cuda:0 --headless --video
```

For a manual baseline, replace the checkpoint arguments with `--qp_only`.
`--manual_authority` accepts twelve fractions (left six, then right six).
If omitted, `--manual_support LEFT RIGHT` supplies one fraction repeated over
all six axes of each foot; its default is full authority on both feet. Fractions
are converted to logits internally, including exact endpoint handling for
manual zero/full support. For a fixed first-frame standing reference use
`Standing-WBC-AUTHORITY-T1-v0`. `--qp_backend osqp` selects the CPU reference
solver for comparison.

Diagnostics include `contact_authority`, effective `contact_weights`,
`wrench_limits`, predicted/measured wrenches, and solver failures.

```bash
python scripts/plot_wbc.py \
  --input logs/rsl_rl/t1_wbc_authority/RUN_DIRECTORY/diagnostics.npz
python -m pytest tests/test_wbc_authority.py -q
```

ONNX metadata describes the twelve raw logits and sigmoid transform; downstream
consumers must apply that transform once. Exported network outputs are logits.

## Required checks before training

Use the project environment with user-site packages disabled to avoid loading
another Isaac Lab installation:

```bash
conda activate hybridmimic
export PYTHONNOUSERSITE=1
python scripts/check_wbc_authority.py
python scripts/rsl_rl/eval_wbc.py \
  --task Standing-WBC-AUTHORITY-T1-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --nominal_standing --qp_only --manual_support 0.5 0.5 \
  --qp_backend osqp --num_envs 1 --steps 500 --device cpu --headless \
  --fail_on_reset --output eval_data/wbc_authority/pretraining/standing_half.npz \
  env.episode_length_s=30.0
```

Repeat with `--manual_support 1 1` for full authority. The half-authority case
is the zero-logit controller. `--nominal_standing` explicitly replaces the input
clip with a symmetric stance, checks the simulator joint limits, grounds both
sole planes, and sets velocities to zero. The file still supplies the motion
archive structure. This is a controller baseline, not validation of the supplied
motion's first frame or its tracking trajectory. Remove `--nominal_standing`
to test that actual reference, and use the tracking task to test the full motion.

`--fail_on_reset` saves partial diagnostics and exits with an error on any reset,
including timeouts. Set the episode duration longer than the rollout as above.
Standing validation uses fall and tracking-error terminations as well as QP
failure detection. Normal evaluation still supports repeated episodes, but its saved diagnostics
now contain reset counts, reset steps, and termination reasons, so completing
an evaluation loop cannot be confused with surviving an episode.

The numerical audit checks 282 fixed-authority/contact-geometry cases on both
the batched solver and OSQP, using the configured robot torque limits. It also
checks unilateral force, friction, wrench caps, inverse dynamics, and solver
agreement. A numerical pass is not a simulator stability certificate. Run the
same standing validation with `--qp_backend batched --device cuda:0` on the
training GPU before a long training run. Explicit `--device cpu` evaluation
also supports the same batched torch algorithm; training still requires CUDA.
The evaluator disables Isaac Sim fast shutdown so Python failures retain their
tracebacks and nonzero exit status.
