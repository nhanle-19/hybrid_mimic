# HybridMimic Momentum WBC Task

This repository variant adds a new HybridMimic task for the Booster T1 that keeps the same RSL-RL training structure as the hybrid controller task, but replaces the centroidal controller with a momentum-based whole-body controller inspired by Koolen et al., *Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas*.

The new task IDs are:

| Use | Task |
| --- | --- |
| Training | `Tracking-Momentum-T1-v0` |
| Evaluation | `Tracking-Momentum-T1-Eval-v0` |

## Setup

Install Isaac Lab v2.1.0 with the same Python environment you use to run Isaac Sim.

From the repository root, install this package in editable mode:

```bash
python -m pip install -e source/whole_body_tracking
```

Training uses Weights & Biases motion artifacts, so authenticate before training:

```bash
wandb login
```

Run all commands below from the repository root.

## Train

Use the new Momentum WBC task with the existing RSL-RL training script:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Momentum-T1-v0 \
  --registry_name ENTITY/PROJECT/MOTION_ARTIFACT:latest \
  --headless \
  --logger wandb \
  --log_project_name PROJECT_NAME \
  --run_name RUN_NAME
```

If the artifact path does not include an alias, the training script appends `:latest`.

Remove `--headless` to run with the Isaac Sim GUI.

## Play

To load and visualize a trained Momentum WBC policy from a W&B run:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Momentum-T1-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID
```

Optional flags:

```bash
--headless
--video --video_length 500
```

Playback also exports the loaded policy to ONNX in the checkpoint run directory.

## Tracking Evaluation

Use the evaluation task for repeatable motion-tracking rollouts:

```bash
mkdir -p eval_data

python scripts/rsl_rl/tracking_play.py \
  --task Tracking-Momentum-T1-Eval-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

To evaluate a specific checkpoint:

```bash
python scripts/rsl_rl/tracking_play.py \
  --task Tracking-Momentum-T1-Eval-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --checkpoint_no ITERATION \
  --headless
```

Tracking data is written to:

```text
eval_data/tracking_play_data.npz
```

## Impulse Evaluation

Use `eval_env.py` to measure recovery from randomized external pushes:

```bash
mkdir -p eval_data

python scripts/rsl_rl/eval_env.py \
  --task Tracking-Momentum-T1-v0 \
  --num_envs 36 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

Momentum WBC impulse results are written to:

```text
eval_data/momentum_eval_data.npz
```

## Implementation

The Momentum WBC task is implemented as a sibling of the original hybrid task, so the existing flat and centroidal-hybrid tasks are left unchanged.

Task registration lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_momentum/__init__.py
```

It registers:

```text
Tracking-Momentum-T1-v0
Tracking-Momentum-T1-Eval-v0
```

The task config lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_momentum/flat_env_cfg.py
```

It subclasses the existing T1 hybrid config so the new task keeps the same robot, observations, action layout, rewards, command setup, and RSL-RL structure.

The environment lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_momentum/momentum_env.py
```

`MomentumWBCEnv` subclasses the existing `HybridEnv` and overrides only `_initialize_hybrid_runtime()`. That swap replaces the original `HybridController` with `MomentumBasedWholeBodyController`, while preserving the existing step loop, action manager, reward bookkeeping, and PD-plus-feedforward torque application.

The controller lives in:

```text
source/whole_body_tracking/whole_body_tracking/utils/momentum_wbc.py
```

The controller keeps the same policy action format as the hybrid task:

```text
joint position targets
desired COM linear velocity
contact logits
reference torques
desired COM angular velocity
```

Instead of using the centroidal approximation from `utils/hybrid.py`, the new controller builds a whole-body QP over:

```text
generalized accelerations qdd
contact wrenches f
```

The QP uses the generalized mass matrix, contact Jacobians, PhysX gravity/Coriolis compensation terms, desired COM acceleration, desired base angular acceleration, joint-space acceleration targets, contact-force weighting, and torque-reference weighting.

After solving the QP, it recovers joint torques with the inverse-dynamics form:

```text
tau = M_joint qdd - J_contact_joint^T f + bias_joint
```

The resulting feedforward torque is clamped by Isaac joint effort limits and passed back through the existing hybrid action manager, which combines it with the joint-space PD position target.

Controller tuning parameters live in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_momentum/controller_cfg.py
```

The RSL-RL config lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_momentum/agents/rsl_rl_ppo_cfg.py
```

It inherits the hybrid PPO settings and changes the experiment name to:

```text
t1_momentum
```

Impulse evaluation was updated in:

```text
scripts/rsl_rl/eval_env.py
```

so `Tracking-Momentum-T1-v0` writes its own result file instead of being grouped with the PD baseline.

## Notes

This is a practical Isaac Lab implementation of the paper's momentum-based control structure, not a line-for-line copy of the Atlas controller. The paper formulates a QP over desired generalized accelerations and contact-force basis multipliers, then computes torques through inverse dynamics. This implementation follows that structure using the runtime model quantities available from Isaac/PhysX for the Booster T1.

The controller requires Isaac/PhysX to expose:

```text
root_physx_view.get_generalized_mass_matrices()
root_physx_view.get_gravity_compensation_forces()
root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
root_physx_view.get_jacobians()
```

If your Isaac Lab build does not expose generalized mass matrices, the task will raise a clear runtime error when the Momentum WBC controller initializes.
