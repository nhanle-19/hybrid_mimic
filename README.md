# HybridMimic Momentum WBC Task

This repository variant adds a new HybridMimic task for the Booster T1 that keeps the same RSL-RL training structure as the hybrid controller task, but replaces the centroidal controller with a momentum-based whole-body controller inspired by Koolen et al., *Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas*.

The new task IDs are:

| Use | Task |
| --- | --- |
| Training | `Tracking-Momentum-T1-v0` |
| Evaluation | `Tracking-Momentum-T1-Eval-v0` |

## Setup

Use a dedicated Conda environment named `hybridmimic` with the following stack:

| Component | Version |
| --- | --- |
| Python | 3.10.21 |
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
The reported server training environment uses Python 3.10.21, Isaac Sim
4.5.0.0, the Isaac Lab 2.2.0 checkout (`isaaclab` package 0.44.9 and
`isaaclab_rl` package 0.2.3), RSL-RL 2.3.3, and PyTorch 2.7.0+cu128.
The remaining pins above are repository setup requirements, not a complete
server dependency snapshot. Exact reproduction also requires the server's
dependency freeze and any source modifications. A fresh local installation
still needs the short PD pipeline check below before full training.

### Create a fresh Conda environment

Run these commands on the local machine. If `hybridmimic` already exists,
inspect it before proceeding; these instructions do not remove existing
environments. Disable user-site packages to avoid loading conflicting packages
or editable checkouts from `~/.local`:

```bash
conda create -n hybridmimic python=3.10.21 pip -y
conda activate hybridmimic
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate hybridmimic

python -m pip install --upgrade pip
python -m pip install \
  "torch==2.7.0" "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  "isaacsim[all,extscache]==4.5.0" \
  --extra-index-url https://pypi.nvidia.com
```

From the project repository root, download Isaac Lab alongside the project:

```bash
cd ..
git clone --branch v2.2.0 --depth 1 \
  https://github.com/isaac-sim/IsaacLab.git IsaacLab-hybridmimic-2.2.0
cd IsaacLab-hybridmimic-2.2.0

python -m pip install "numpy==1.26.4" \
  "torch==2.7.0" "torchvision==0.22.0" "rsl-rl-lib==2.3.3" \
  -e source/isaaclab \
  -e source/isaaclab_assets \
  -e source/isaaclab_tasks \
  -e source/isaaclab_mimic \
  -e source/isaaclab_rl

cd ../hybrid_mimic
```

This installs the Python packages without invoking the Isaac Lab installer's
system-package installation step. The CUDA wheel requires a compatible NVIDIA
driver. Installation references: [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_python.html)
and [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

From the repository root, install this package in editable mode:

```bash
python -m pip install -e source/whole_body_tracking
python -m pip check
```

Verify versions and import locations in the activated environment:

```bash
python -m pip show isaacsim isaaclab isaaclab-rl rsl-rl-lib torch
python -c "import sys, site, rsl_rl; print(sys.executable); print('User packages enabled:', site.ENABLE_USER_SITE); print(rsl_rl.__file__)"
```

User packages should be disabled (`False`), and RSL-RL should load from the
`hybridmimic` environment rather than another project's editable checkout.

Resolve reported dependency conflicts before training. The examples below pass
the W&B API key to each command explicitly.

Run all commands below from the repository root.

## Convert a retargeted motion

In your Isaac Lab environment, convert the retargeted G18 kick into the training
format and upload it to W&B. Use this Bash prompt to pass the API key directly
to the conversion command:

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

All training examples use W&B logging in the `hybrid_mimic` project. Activate
`hybridmimic` before running them. Each example prompts for an API key and
passes it to that training process with `WANDB_API_KEY="$WANDB_API_KEY"`.
The motion source (local NPZ or W&B artifact) is independent of the logger.
`CUDA_VISIBLE_DEVICES=0` selects GPU 0 for each training process; change `0` to
the GPU index you want to use. Training defaults to 30,000 iterations; use
`--max_iterations` only to override that default.

Use the new Momentum WBC task with the existing RSL-RL training script:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-Momentum-T1-v0 \
  --registry_name ENTITY/PROJECT/MOTION_ARTIFACT:latest \
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
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
read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-Momentum-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name momentum_g18_push_kick_right
```

This loads the motion locally and logs training metrics to W&B using the API
key supplied to the command. The input must be the converted file
containing joint velocities and body transforms; the raw retargeted NPZ and
terminal logs cannot be used directly for training.

### Train the PD baseline

Use the original `Tracking-Flat-T1-v0` task with the local converted motion and
log training metrics to the same W&B project:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
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

## Record the reference motion

Replay the converted training NPZ directly on T1 and record one full motion
cycle, without a policy or physics stepping:

```bash
conda activate hybridmimic
python scripts/replay_npz.py \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --headless --video \
  --output_file eval_data/g18_reference.mp4
```

The recording starts at frame zero, uses the NPZ's FPS, and exits after the
last frame. Add `--video_length 500` to cap the recording at 500 frames.
Omit `--headless --video` for looping interactive playback. The existing
`--registry_name ENTITY/PROJECT/ARTIFACT:latest` input remains available as an
alternative to `--motion_file`. Use the converted `_training.npz`, which has
full-body transforms and velocities; raw retargeted files need conversion first.

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
