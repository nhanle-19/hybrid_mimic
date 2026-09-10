# Atlas-style T1 controller

This is a separate implementation of the requested centroidal/contact QP, inspired
by Koolen et al., *Design of a Momentum-Based Control Framework and Application to
the Humanoid Robot Atlas*, DOI
[10.1142/S0219843616500079](https://doi.org/10.1142/S0219843616500079).
The `floating_model` controller, its 57 actions, and its checkpoints are preserved.
This task is not a reproduction of the paper's complete walking/hardware system.

## Formulation and conventions

`AtlasModel` constructs a Pinocchio free-flyer model from `atlas_dynamics.json`,
exported from the actual `t1.usd` rigid-body tree. The export includes every body's
mass, CoM, principal inertia, joint axis and joint attachment transforms, and source
layer hashes. Joints are inserted depth first (contiguous subtrees for CRBA).
The nearby retargeting URDF is not silently substituted for the simulator asset.

Generalized configuration has 30 coordinates: root XYZ, root quaternion XYZW,
23 scalar joints. Generalized velocity has 29 coordinates: root **local-frame**
linear/angular velocity and joint velocities. Simulator root-link world velocity
is rotated into that frame. Joint arrays are explicitly reordered by joint name.
Spatial vectors use `[linear, angular]`. Centroidal momentum and contact wrenches
are world aligned and centered at the whole-robot CoM.

At each physics step, Pinocchio computes full `M`, nonlinear bias `b`, `Ag`,
`dAg @ nu`, world-aligned body Jacobians, and `dJ @ nu` analytically. There is no
finite-differenced or stale QP Jacobian. Actual momentum is independently calculated
by summing body linear and angular momentum. Runtime startup checks compare
simulator masses, CoM offsets, inertias, link positions, and momentum with the model.
The simulator's measured body states also supply the recorded actual momentum rate.

The decision vector is `x = [nudot, rho]`:

| Planned support | Acceleration variables | Force coefficients | Total |
|---|---:|---:|---:|
| Flight | 29 | 0 | 29 |
| One foot | 29 | 16 | 45 |
| Two feet | 29 | 32 | 61 |

Each active foot has four configured sole points, with four rays per point:

`B = [n + mu*t1, n - mu*t1, n + mu*t2, n - mu*t2]`, `f = B*rho`, `rho >= 0`.

The tangents are orthonormal to the normalized normal. Their convex hull is an
inscribed friction pyramid, so `||f_t|| <= mu*f_n`. There is no free moment at a
contact point; patch moments arise only from moment arms. Thus positive corner
normal forces constrain CoP to the configured support rectangle. The current
rectangle is x=[-0.09,0.10], y=[-0.04,0.04], z=-0.03 m in the foot-link frame.
It is a conservative foot-patch approximation, not an identified pressure map.

For point `i`, `Jp = Jlinear - skew(r_body_to_point)*Jangular` and
`Q_i = [B_i; skew(p_i-com)*B_i]`. Assemble `Q` and
`G = [Jp_1.T B_1 ... Jp_k.T B_k]`.

The objective is:

```
0.5 ||Ag*nudot + dAg*nu - hdot_des||_Wh^2
+ 0.5 ||rho||_Wrho^2 + 0.5 ||nudot||_Wa^2
+ sum_tasks 0.5 weight_task ||Jtask*nudot - (ades - dJtask*nu)||^2
```

Default weights are `Wh=diag(1,1,1,10,10,10)`, `Wrho=1e-5 I`, and `Wa=1e-5 I`.
Posture, pelvis orientation and swing-foot tasks have weights 0.1, 5 and 10.
All gains and weights are exposed in `T1AtlasControllerCfg`.

Hard constraints are:

```
Ag*nudot - Q*rho = Wgravity + Wexternal - dAg*nu
Jstance*nudot = -dJstance*nu            # six rows per active foot
rho >= 0
-tau_max <= Mj*nudot + b_j - Gj*rho - generalized_external_j <= tau_max
```

`Wgravity = [mass*gravity, 0]`. Known external wrenches are mapped both into
centroidal and generalized coordinates. `ExternalWrench(body, wrench)` denotes
a world-aligned force/moment at that body-frame origin. There is no auxiliary,
residual, or artificial base wrench in the decision vector.

OSQP solves the genuine bounded QP in float64 (absolute/relative tolerances 1e-8,
polishing enabled). Only `solved` status with finite output and checked physical
residuals is accepted. The default maximum component residual is 2e-5; units differ
by residual (force, torque, acceleration), so this is a numerical acceptance test,
not an empirical tracking threshold. No infeasibility fallback or torque clipping
is hidden. The recovered torque is applied directly, with zero actuator stiffness,
damping, armature and joint friction. Simulation contact forces remain PhysX outputs,
not forces injected from the QP.

## Feedback and action interface

There are **29 residual actions**, ordered using the model joint names:

| Slice | Interpretation |
|---|---|
| `[0:23]` | Joint-posture residual, 0.15 rad per action unit |
| `[23:26]` | World CoM velocity residual, 0.25 m/s per unit |
| `[26:29]` | World centroidal angular momentum residual, 0.5 kg m²/s per unit |

The reference motion's full configuration/velocity generates reference whole-robot
CoM and momentum. With `p` linear and `L` angular centroidal momentum:

```
pdot_des = mass*40*(com_ref-com) + 10*(p_ref + mass*dv_policy - p)
Ldot_des = 15*(L_ref + dL_policy - L)
```

Posture acceleration uses position/velocity gains 40/8; pelvis orientation uses
60/12; swing feet use 80/16 with world translation and SO(3) orientation errors.
No wrench-cost logits or torque-reference actions exist. Retraining is required;
old floating_model policies cannot be loaded into this interface.

## Planned contacts and external inputs

The default planner precomputes sole-center heights for the reference trajectory,
normalizes the reference floor once using their tenth percentile, and
declares load-bearing stance when a sole center is within 0.025 m of that floor.
This avoids treating tilted retargeted feet as flight throughout the motion.
This is a documented heuristic, not ground-truth contact labeling or policy gating.
The T1 task permits only foot contacts; it does not supply fictitious hand support.
There is no forced support foot during planned flight.

A labeled boolean NPY array `(motion_frames,2)` replaces this heuristic via
`--contact_schedule PATH`. At runtime, the action term also exposes:

```python
term = env.unwrapped.action_manager.get_term('atlas')
term.set_active_contacts(boolean_mask)  # (num_envs,2), left then right
term.set_active_contacts(None)          # return to reference schedule
term.set_external_wrenches(env_id, known_wrenches)
```

External inputs must match forces physically applied by the caller; this setter
does not apply forces in PhysX. Unknown external pushes and domain randomization
are disabled by default. An incorrect planned contact set can invalidate the
model; compare planned support with the recorded sensor contact signal.

## Tests and evaluation

From the repository root, in the `hybridmimic` environment:

```bash
python -m pip install -r requirements-atlas.txt
python -m pytest tests/test_atlas_qp.py -q --junitxml=eval_data/atlas/numerical_tests.xml
python scripts/check_atlas.py
```

The numerical suite checks the independent body-sum momentum identity, analytical
centroidal/Jacobian biases against centered differences, momentum-rate and stance
residuals, independent recursive Newton–Euler inverse dynamics, base residual,
normal/friction feasibility, CoP bounds, exact zero swing forces, active torque
limits, known external force mapping, and rejection of an impossible hard task.
Cases include flight, either single-support foot, double support and nonzero
contact-consistent velocity. Numerical acceptance is separate from PhysX rollout
and controller tracking performance.

Evaluate the reference-feedback controller before training a policy:

```bash
python scripts/rsl_rl/eval_atlas.py \
  --task Tracking-Atlas-T1-Eval-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1 --steps 141 --zero_policy --headless

python scripts/plot_atlas.py --input eval_data/atlas/diagnostics.npz
```

Train a new policy:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Atlas-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 2 --run_name atlas_g18_push_kick_right --headless
```

For policy evaluation, replace `--zero_policy` with `--load_run RUN_DIRECTORY
--checkpoint model_ITERATION.pt`. Atlas runs are under `logs/rsl_rl/t1_atlas`.
The generic legacy `tracking_play.py` assumes a joint-position action; use the
dedicated `eval_atlas.py` for this task.

Diagnostics are sampled at the 0.002-second physics interval. They include desired,
QP-predicted and actual finite-difference momentum rates, predicted normal/tangential
point forces, sensor normal force, friction utilization, CoP, planned contacts,
contact-acceleration residual, torque utilization, commanded/applied torque and
inverse-dynamics residual. Reset boundaries are excluded from differentiation.
Friction-utilization ratios are undefined at points carrying 1 N or less; absolute
friction-cone residuals remain the acceptance check near zero load. Tiny negative
forces within the stated solver tolerance are reported, not clamped away.
The actual rate uses simulator body CoM velocities, masses and inertias. QP
residuals are explicitly labeled predictions; low QP residuals do not prove that
PhysX realized those accelerations or contact forces. Isaac Lab 2.2 sensor normal
force excludes friction, so measured friction utilization/CoP is not available.

## Scope and deviations

- This implements the specified centroidal/contact formulation, not the Atlas
  hardware controller, hydraulic actuation, ICP walking planner, or grasp logic.
- Secondary motion tasks are weighted objectives by default to avoid conflicting
  hard tasks. The core also supports explicit hard `MotionTask`s. Stance and
  physical feasibility constraints are always hard; no contact slack is used.
- Four conservative sole points and fixed flat-ground normals replace identified
  pressure distributions. The core supports arbitrary normals, but the T1 task
  currently supplies a flat surface and full-foot stance. Toe/heel transitions,
  impact impulses, compliant contacts and support-patch identification are future
  extensions. Acceleration constraints alone do not remove initial stance velocity
  or accumulated position drift; touchdown needs a consistent planned state.
- Reference-derived contacts and PD momentum feedback replace a full gait/ICP
  planner. Reference acceleration feedforward is not included.
- CPU OSQP is used for inspectable correctness. It is not optimized for thousands
  of GPU environments, and no real-time performance claim is made.
- The inertial model is fixed to the exported USD. No random inertial perturbations,
  hidden forces, or actuator PD torques are allowed by the new configuration.
- Simulator success must be assessed separately; passing numerical tests alone
  is not a claim of stable motion tracking on the T1.

## Validation run on 2026-09-10

`python -m pytest tests/test_atlas_qp.py -q` passed **16 tests**. The full
`eval_atlas.py` command above also ran with `--device cpu`, completing 141 control
steps / 1,410 physics samples (2.82 s) with zero policy residuals. The run exercised
flight, single support and double support. Maximum QP stance residual was
1.94e-8, and maximum full inverse-dynamics residual was 1.71e-6. Maximum torque
utilization was 1.000000071 (numerical tolerance); no torque was clipped.

This is **not a successful tracking validation**: the maximum measured stance
acceleration residual was about 8.75e3, and measured/predicted momentum rates
differed substantially. These are actual physics results, not hidden behind the
small optimization residuals. The heuristic support schedule, initial contact
conditions/impacts, unplanned collisions and ideal rigid-contact assumption need
to be assessed using the generated contact and momentum plots. No policy has been
trained for the new action interface.

Results: `eval_data/atlas/numerical_tests.xml`, `numerical_residuals.json`,
`diagnostics.npz`, `rollout_summary.json`, and `plots/`.

Primary implementation references:
[Pinocchio centroidal algorithms](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/jnrh2023/template/algorithm.html),
[OSQP solver settings](https://osqp.org/docs/release-0.6.3/interfaces/solver_settings.html),
[Isaac Lab 2.2 sensors](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sensors.html).
