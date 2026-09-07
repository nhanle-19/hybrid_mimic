# HybridMimic Momentum WBC Task

This repository variant adds a new HybridMimic task for the Booster T1 that keeps the same RSL-RL training structure as the hybrid controller task, but replaces the centroidal controller with a momentum-based whole-body controller inspired by Koolen et al., *Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas*.

The new task IDs are:

| Use | Task |
| --- | --- |
| Training | `Tracking-Momentum-T1-v0` |
| Evaluation | `Tracking-Momentum-T1-Eval-v0` |

## Setup

Use the following target stack in one Conda environment:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| Isaac Sim | 4.5.0 |
| Isaac Lab | 2.2.0 (Git tag `v2.2.0`) |
| PyTorch | 2.7.0, CUDA 12.8 wheel (`cu128`) |
| torchvision | 0.22.0, CUDA 12.8 wheel (`cu128`) |
| RSL-RL (`rsl-rl-lib`) | 2.3.3 |
| NumPy | 1.26.4 (`numpy<2` required) |
| W&B | `wandb>=0.19` (project requirement) |

Isaac Lab 2.2.0 supports Isaac Sim 4.5 and supplies the quaternion inverse and
filtered contact-force history APIs used by the tasks. See the
[release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.2.0) and
[RSL-RL dependency definitions](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab_rl/setup.py).
This is the migration target; full project training on this stack has not yet
been validated. Run the short PD pipeline check below before full training.

### Migrate an existing Conda environment

These commands assume `hybridmimic` already contains Isaac Sim 4.5.0. Clone it
to retain the existing environment, and disable user-site packages in the clone
to avoid loading conflicting packages from `~/.local`:

```bash
conda create -n hybridmimic22 --clone hybridmimic
conda activate hybridmimic22
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate hybridmimic22
```

From the project repository root, download Isaac Lab alongside the project:

```bash
cd ..
git clone --branch v2.2.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git IsaacLab-2.2.0
cd IsaacLab-2.2.0

python -m pip install \
  "torch==2.7.0" "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install "numpy==1.26.4" \
  -e source/isaaclab \
  -e source/isaaclab_assets \
  -e source/isaaclab_tasks \
  -e source/isaaclab_mimic \
  -e "source/isaaclab_rl[rsl_rl]"

cd ../hybrid_mimic
```

This installs the Python packages without invoking the Isaac Lab installer's
system-package installation step. The CUDA wheel requires a compatible NVIDIA
driver. To return to the previous environment, run `conda activate hybridmimic`.

From the repository root, install this package in editable mode:

```bash
python -m pip install -e source/whole_body_tracking
python -m pip check
```

Resolve reported dependency conflicts before training. For W&B artifact access
or W&B logging, authenticate with the command below or use the per-command API
key examples later in this README:

```bash
wandb login
```

Run all commands below from the repository root.

## Convert a retargeted motion

In your Isaac Lab environment, convert the retargeted G18 kick into the training
format and upload it to W&B. As an alternative to `wandb login`, use this Bash
prompt to pass the API key directly to the conversion command:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo

WANDB_API_KEY="$WANDB_API_KEY" python scripts/csv_to_npz.py \
  --input_file retargeted_motion/g18_push_kick_right_t1.npz \
  --input_fps 30 \
  --output_fps 50 \
  --output_name g18_push_kick_right_t1 \
  --headless
```

Wait for the W&B upload to finish, then stop the replay with Ctrl+C. The key
remains a shell variable and is passed to this command without `export`.
A detached tmux pane keeps its shell and variables alive until the shell exits.

The source motion is approximately 29.7406 FPS; the converter currently uses
integer input FPS, so `--input_fps 30` makes playback approximately 0.9% faster.

For training, use `--registry_name ENTITY/csv_to_npz/g18_push_kick_right_t1:latest`,
replacing `ENTITY` with the W&B username or team that owns the artifact. To use
the key in the same shell, prefix the training command with
`WANDB_API_KEY="$WANDB_API_KEY"` as above.

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

### Train from a local motion file

The converter saves `retargeted_motion/<output_name>_training.npz` in the
repository before uploading to W&B. For the G18 command above, this is:

```text
retargeted_motion/g18_push_kick_right_t1_training.npz
```

For training on another machine, transfer this converted file to that machine
first. Use `--motion_file` instead of `--registry_name`:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Momentum-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1024 \
  --headless \
  --logger tensorboard \
  --run_name momentum_g18_push_kick_right
```

This command needs no W&B authentication. The input must be the converted file
containing joint velocities and body transforms; the raw retargeted NPZ and
terminal logs cannot be used directly for training.

### Train the PD baseline (W&B logging)

Use the original `Tracking-Flat-T1-v0` task with the local converted motion and
log training metrics to W&B. Enter the API key in the same Bash terminal:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo

WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-Flat-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name pd_g18_push_kick_right
```

This runs the full configured iteration count. For a short pipeline check, use
`--num_envs 64 --max_iterations 20 --run_name pd_pipeline_check` instead.
Check that PPO iterations complete with finite losses and checkpoints are saved
under `logs/rsl_rl/t1_flat/`.

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
