# WBC-AUTHORITY: cost-only translational contact authority

A separate task with six policy logits: left foot `x, y, z`, then right foot
`x, y, z`. The shared FORCE/AUTH mapping is
`c = clip((sigmoid(action) - 0.10) / 0.80, 0, 1)`.
It reaches exactly zero below approximately -2.197225 and exactly one above
+2.197225, with continuous transitions and slope changes at the boundaries.
Zero logits give 50% authority. These fractions affect only the contact cost;
they never activate/deactivate support or scale force limits.

## QP formulation

All quantities use world-aligned axes at the foot link origin. The linear
contact acceleration is `a_lin = J_lin nu_dot + Jdot_lin nu`. For each
geometrically available foot, the QP adds the soft objective

```
0.5 * sum_{j=x,y,z} contact_authority_weights[j] * c[j]^2 * a_lin[j]^2
```

The three base weights default to 100. The target is zero actual linear
contact acceleration, not reference acceleration. There is no rotational
contact-acceleration cost and no hard stationary-foot equality. An airborne
foot has no contact cost. Zero authority removes the cost without removing
contact forces. Angular behavior follows the remaining WBC objectives and
physical contact constraints; removing this penalty does not guarantee that
feet will stay flat or that the motion will succeed.

There are no authority-dependent force or moment bounds. The QP retains
nonnegative normal forces, the point-force friction pyramid, available sole
vertices, centroidal balance, joint torque limits, and fixed normal-force caps
of 600 N per foot (`contact_force_max`). Moments arise from point forces and
support geometry; there are no additional independent moment caps. Geometry
and the configured contact source gate support independently of authority.

Momentum, joint posture, and pelvis tracking remain. The optional swing-foot
objective is disabled. Constraints apply to QP predictions; measured PhysX
contact forces can differ, especially on impact. Failed solves still terminate
episodes rather than relaxing constraints.

The flat-foot sliding fixture exposes a further limitation: the QP's friction
pyramid bounds force magnitude but does not enforce the sticking/sliding law
that couples friction to contact motion. Removing an x/y acceleration cost can
therefore produce a predicted slide while PhysX keeps the loaded foot stuck.
This occurred with exact zero x/y authority in the left-foot slide experiment;
the raw labels and applied torques were verified. Eight AUTH label schedules,
including zero authority on every axis, failed to track its 40 mm lateral travel.
The left foot still carried about 155 N. These discarded trial files were
removed during the evaluation cleanup.

Kinematic joint-limit checks alone also do not establish sliding feasibility.
For this exact fixed-torso reference at friction 0.6, an inverse-dynamics check
with sliding friction found all 20 sampled outward-moving frames infeasible:
the required right hip-roll torque was 34.5–37.3 N m against its 30 N m limit,
even after relaxing the right-foot friction cone. All 20 sampled inward-moving
frames were feasible in that model. This is a result for the exact reference,
not a claim about motions with weight transfer or altered contact conditions.

Reducing simulator friction to 0.3 or 0.1 did not make the released-cost AUTH
foot track the slide. A separate eight-label search with both the QP friction
and simulator friction set to 0.3 also remained planted. At extremely low
physical friction, the feet drifted instead. These experiments change only
their own simulation instances; the normal task still uses friction 0.6.
The first physical-friction sweeps deliberately held the QP cone at 0.6;
matched-model runs were separate tests. Their discarded trial files were removed.

Further tuning used the support pose that passed FORCE sliding: root 8 cm
right, 2 cm lower, and rolled 5 degrees right, with the same foot paths and
friction 0.6. All 64 AUTH cost-label candidates at default 600/600 N caps still
remained planted. A separate AUTH experiment imposed fixed caps of 0.01/600 N
and used active authorities left `(0, 0, 1)`, right `(0, 0, 0)`. Its independent
500-step replay tracked 40.71 mm of lateral travel with 3.46 mm XYZ RMS error,
no resets, and no QP or torque-saturation failures. Batch sole vertices stayed
within 0.484 mm of the floor. See
`eval_data/left_foot_slide/auth/authority_validation.json` and the
README replay command, which includes the required fixed-cap override.
This result requires the additional unloading bound; it is not evidence of
successful unloading by the six cost labels alone. Production defaults remain
600/600 N, and the actor still changes costs only.

## Training interface

The actor observes reference motion at frame offsets `0, 1, 5, 10`; the critic
and WBC receive actual state. The actor learns a reference-conditioned cost
schedule and cannot react directly to measured tracking errors. The task uses
only the alive reward. The available action-rate helper measures changes in
the six mapped authorities, but is not enabled by that reward configuration.
The Gaussian logit entropy bonus is disabled. PPO still samples stochastic
actions; the raw Gaussian outputs remain unchanged in PPO storage and likelihood
calculations. Only the controller maps them to bounded authority values.

Train from scratch: historical twelve-output AUTHORITY checkpoints and
two-output FORCE checkpoints are incompatible. Existing log/checkpoint files
are preserved. Defaults are 1,024 environments, 50 Hz policy/QP, 500 Hz physics,
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
`--manual_authority` accepts six cost fractions (left xyz, then right xyz).
If omitted, `--manual_support LEFT RIGHT` supplies one fraction repeated over
all three axes of each foot; its default is full cost authority on both feet. Fractions
are converted to raw outputs internally using `logit(0.10 + 0.80*c)`, with
ordinary endpoint outputs -3 and +3 for
manual zero/full cost authority. For a fixed first-frame standing reference use
`Standing-WBC-AUTHORITY-T1-v0`. `--qp_backend osqp` selects the CPU reference
solver for comparison.

Diagnostics include `contact_authority`, effective six-axis `contact_weights` (angular entries are zero),
`normal_force_limits`, predicted/measured wrenches, and solver failures.

```bash
python scripts/plot_wbc.py \
  --input logs/rsl_rl/t1_wbc_authority/RUN_DIRECTORY/diagnostics.npz
python -m pytest tests/test_wbc_authority.py -q
```

ONNX metadata describes the six raw logits, cost-only formulation, fixed normal-force caps, and the dead-zone transform with epsilon 0.10; downstream
consumers must apply that transform once. Exported network outputs are logits.
Plain-sigmoid checkpoints have different action semantics under this mapping.
Use a fresh training run for comparisons. Migrate manual schedules as described
in [the schedule guide](manual_wbc_schedule.md), preserving their mapped costs.

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

The numerical audit checks 273 fixed-authority/contact-geometry cases on both
the batched solver and OSQP, using the configured robot torque limits. It also
checks unilateral force, friction, fixed normal-force caps, inverse dynamics, and solver
agreement. A numerical pass is not a simulator stability certificate. Run the
same standing validation with `--qp_backend batched --device cuda:0` on the
training GPU before a long training run. Explicit `--device cpu` evaluation
also supports the same batched torch algorithm; training still requires CUDA.
The evaluator disables Isaac Sim fast shutdown so Python failures retain their
tracebacks and nonzero exit status.
