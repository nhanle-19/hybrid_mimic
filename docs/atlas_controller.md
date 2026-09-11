# Atlas dynamics with the HybridMimic policy interface

`Tracking-Atlas-T1-v0` and `Tracking-Atlas-T1-Eval-v0` now use the same
**57-action policy interface**, observations, rewards, and joint PD plus
feedforward torque application as the existing hybrid/floating-model tasks.
`AtlasEnv` inherits the shared `FloatingModelEnv`/`HybridEnv` runtime and replaces
the controller with batched Torch analytical dynamics and a float64 GPU QP.
The Pinocchio/OSQP implementation remains an explicit evaluation reference.

**Current experiment:** `T1AtlasControllerCfg.enforce_stance = False` disables
the hard contact-acceleration equality `J_contact*qdd + dJ_contact*qdot = 0`
for training and evaluation. Hard stance is currently supported only by the
CPU reference backend (`backend=osqp`) during evaluation. GPU training rejects
`enforce_stance=True`.
Contact forces, friction/nonnegativity, dynamics balance and combined PD + FF
torque limits remain active. Stance acceleration is still recorded, but does
not reject a solution while this switch is off; diagnostics include the switch
value. The standalone `AtlasQP` retains strict stance constraints by default.

This replaces the earlier 29-action reference-residual Atlas design. Old
29-action Atlas checkpoints cannot be resumed. Matching the floating-model
policy dimensions does not establish successful transfer between controllers;
train a new Atlas policy for a controlled comparison.

## Policy output and torque application

Joint channels are in **simulator joint order**, exactly as in HybridMimic.
The adapter explicitly reorders them to/from the analytical model internally.

| Slice | Meaning |
| --- | --- |
| `[0:23]` | Normalized joint-position targets relative to the configured default pose |
| `[23:26]` | Desired body-frame linear velocity, scaled by 0.25 m/s |
| `[26:31]` | Five wrench-cost logits: auxiliary base, then configured end effectors |
| `[31:54]` | Torque references, scaled by 0.1 times each joint's effort limit |
| `[54:57]` | Desired body-frame angular velocity, scaled by 0.5 rad/s |

The end-effector order is inherited from `T1HybridControllerCfg`; all joint and
end-effector names are exported in ONNX metadata. Action decoding and velocity
feedback call the same `ctrl2components` and `highlvlPD` functions as floating.

```
q_target = configured_default_pose + joint_action_scale * policy_joint_output
policy targets + measured state -> Atlas QP -> feedforward torque
applied torque = joint PD torque + feedforward torque (actuator effort-limited)
```

The reference motion reaches the policy through observations and tracking
rewards. It is **not directly added to the posture or velocity targets**. Its
foot heights are also used to plan the constrained solver's support schedule.
Zero policy outputs are a pipeline check, not a reference-tracking controller.

Joint PD owns posture tracking. The Atlas QP has no joint-posture tracking task,
and its momentum target contains no desired joint-posture acceleration. The
small acceleration regularizer remains for numerical conditioning.

At every physics step, the adapter computes `u_PD` from the actual position
action scale/offset, current joint state, velocity target, and actuator gains.
Inverse dynamics determines the combined torque; the controller returns
`u_FF = u_total - u_PD`. The QP enforces:

```
-tau_max <= u_PD + u_FF <= tau_max
# Equivalently, the available feedforward interval is:
-tau_max - u_PD <= u_FF <= tau_max - u_PD
```

The policy's torque-reference cost applies to `u_FF`. Feedforward torque may
exceed an individual effort limit when needed to cancel a large PD contribution;
it is the sum that is bounded. Inverse-dynamics residuals and the torque-limit
reward also use the sum. Diagnostics record `pd_torque`, `feedforward_torque`,
and `commanded_torque` (their sum) separately from simulator `applied_torque`.

## Analytical controller

`AtlasTorchModel` uses `atlas_dynamics.json`, exported from the actual T1 USD
rigid-body tree, to compute batched dynamics on CUDA. `AtlasModel` builds the
independent Pinocchio reference from the same data. Both compute full mass/bias matrices,
centroidal momentum maps and frame Jacobians/biases at every physics step.
Simulator startup checks compare mass, inertia, CoM offsets, link frames and
independently measured body-sum momentum. Hybrid actuator armature is included
in the joint block of the optimization mass matrix.

The hybrid-mode decision vector is `[generalized_acceleration, rho, base_wrench]`.
There are 29 acceleration variables, four nonnegative friction-ray coefficients
per contact point, and six auxiliary-base-wrench variables. Each foot uses four
sole points. The CPU reference compacts inactive contacts. With no hands active, its sizes are:

| Foot support | Variables |
| --- | ---: |
| Flight | 35 |
| One foot | 51 |
| Two feet | 67 |

The objective includes analytical momentum-rate tracking, base acceleration
targets, the policy's torque references, and wrench costs proportional to
`exp(-clip(logit, -10, 10))`. Angular wrench costs use the inherited factor of 20.
The first logit weights the auxiliary base wrench, preserving its role in the
existing hybrid/floating interface. End-effector logits weight each active
contact's resultant force/moment; inactive contacts have zero force.

The auxiliary base wrench is an **optimization aid, not a physical applied
force**. It is recorded separately as `auxiliary_base_wrench`. Consequently,
small inverse-dynamics residuals in this mode include that auxiliary term;
they do not prove physically realizable support. QP torque bounds apply to the
combined PD-plus-feedforward command, and simulator actuator limits remain active.

Hard constraints enforce centroidal balance including the auxiliary wrench,
stance acceleration when enabled, nonnegative friction-ray coefficients, and combined
joint torque limits. OSQP uses float64 and checks solver status and residuals.
An infeasible solve raises an error rather than silently changing the solution.

The GPU solver keeps all contact slots in a fixed batch, with inactive force
maps zeroed, and eliminates the six base-wrench equality variables. It uses
variable/row equilibration and up to 60 predictor/corrector iterations. Numerical
solves stay on the simulation device; convergence/failure checks synchronize
small status values with the host. There is no CPU solver fallback. The OSQP
reference permits up to 100,000 iterations.

If solving or residual validation still fails, the runtime writes the QP matrices,
bounds, state and PD contribution to `eval_data/atlas/failures/qp_failure_*.npz`
and prints the path. Copy that file for diagnosis of the exact failing problem.

The standalone `AtlasQP.solve(..., hybrid=None)` mode remains available for
strict physical numerical tests: it has no auxiliary base wrench or hybrid
policy objective. Its original variable counts are 29/45/61. The registered
training/evaluation tasks use hybrid mode.

## Contacts and configuration

The foot planner normalizes reference sole heights using their tenth percentile
and declares stance within 0.025 m of that floor. This remains a heuristic, not
ground-truth contact labeling. An explicit boolean NPY `(motion_frames, 2)` can
replace it through the evaluation script's `--contact_schedule` option.
Measured hand contact above 10 N adds a single contact point at the hand-link
origin on the flat ground. Hand geometry and normals are approximations;
non-ground contacts need an appropriate contact model before use.

The foot patch is x=[-0.09,0.10], y=[-0.04,0.04], z=-0.03 m in foot-link coordinates.
Four rays per point form an inscribed friction pyramid with coefficient 0.6.
Foot diagnostics record point forces, normal/tangential forces, friction
utilization, CoP and planned versus sensor contact information.

Runtime access is through `env.unwrapped.hybrid_controller`:

```python
controller.set_active_contacts(mask)  # (num_envs, 2), left/right foot
controller.set_active_contacts(None)  # return to planned foot schedule
controller.set_external_wrenches(env_id, known_wrenches)
```

External-wrench inputs describe forces applied by the caller; the setter does
not apply them in PhysX. Fixed inertial/material settings and deterministic
motion initialization remain the Atlas baseline; hybrid domain randomization
is not enabled. Hybrid policy/observation/reward structure and PD gains are
preserved. CPU OSQP is not optimized for thousands of parallel environments.

## Training and evaluation

From the repository root:

```bash
conda activate hybridmimic
python -m pip install -r requirements-atlas.txt

read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-Atlas-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --device cuda:0 \
  --num_envs 128 \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name atlas_hybrid_g18_push_kick_right \
  --headless
```

Runs default to 30,000 iterations and save under `logs/rsl_rl/t1_atlas/`.
Use a new run rather than resuming an earlier 29-action Atlas checkpoint.

```bash
python scripts/rsl_rl/eval_atlas.py \
  --task Tracking-Atlas-T1-Eval-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1 --steps 141 --zero_policy --headless

python scripts/plot_atlas.py --input eval_data/atlas/diagnostics.npz
```

For a trained policy, replace `--zero_policy` with `--load_run RUN_DIRECTORY
--checkpoint model_ITERATION.pt`. Diagnostics are sampled every physics step;
reset boundaries are excluded from measured finite differences. Predicted
combined torque and simulator applied torque are recorded separately.

Numerical and metadata regression checks:

```bash
python -m pytest tests/test_atlas_batched.py tests/test_atlas_qp.py tests/test_exporter_metadata.py -q
```

The September 10 evaluation and its 16 numerical tests documented the earlier
29-action direct-torque design. Those rollout results do not validate this
57-action hybrid integration. Numerical checks alone are not evidence of stable
learned tracking; evaluate a newly trained policy separately.

Validation on September 11: all 25 numerical/metadata tests passed. A CPU run
with two environments completed one PPO iteration (48 transitions), saved a
checkpoint through W&B offline mode, and exported a checked ONNX graph with
57 action outputs. Reloading that checkpoint completed five evaluation steps
(50 physics samples) and saved finite applied torques and diagnostics. The
maximum QP inverse-dynamics residual, including the auxiliary base wrench, was
1.25e-6. Diagnostic plots were generated successfully. These are pipeline checks,
not a completed training run or a GPU performance/learned-tracking validation.

After separating posture PD from the QP and bounding their summed torque,
29 tests passed, including large positive/negative PD contributions that require
feedforward cancellation. A five-step CPU checkpoint evaluation produced 50
physics samples: maximum combined utilization was 1.000000009 (solver roundoff),
and the maximum recorded applied-versus-commanded difference was 2.38e-5 Nm.

GPU migration validation: 39 regression tests passed, including CUDA dynamics
and mixed-contact QP comparisons against Pinocchio/OSQP. A CUDA training smoke
run with 128 environments completed one PPO iteration (3,072 transitions) and
saved `model_0.pt`; collection took 35.107 s and learning 0.232 s on the local
RTX 4060 Laptop GPU. The default is now 128 environments. A 1,024-environment
trial encountered PhysX GPU kernel-launch errors and was stopped; that size is
not validated on this machine. These checks do not establish long-run tracking
quality. Host work remains for initialization, status checks, logging and file
exports; numerical dynamics, QP solves, simulation and PPO use CUDA.
