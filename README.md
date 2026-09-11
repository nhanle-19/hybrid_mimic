# HybridMimic Floating Model Task

The separate **Atlas-style centroidal controller** is available as
`Tracking-Atlas-T1-v0` and `Tracking-Atlas-T1-Eval-v0`. It uses physical point
contacts and an inequality-constrained QP behind the same 57-action HybridMimic
policy interface and PD-plus-feedforward step loop as `floating_model`.
In Atlas, joint PD handles posture; the QP handles balance/contact dynamics
without a duplicate posture-tracking objective and limits the combined PD plus
feedforward torque.
See [the formulation, tests, evaluation commands, and documented deviations](docs/atlas_controller.md).
The `floating_model` task remains the baseline.

This repository variant adds a new HybridMimic task for the Booster T1 that keeps the same RSL-RL training structure as the hybrid controller task, but replaces the centroidal controller with a floating-base whole-body controller inspired by Koolen et al., *Design of a Momentum-Based Control Framework and Application to the Humanoid Robot Atlas*.

The new task IDs are:

| Use | Task |
| --- | --- |
| Training | `Tracking-FloatingModel-T1-v0` |
| Evaluation | `Tracking-FloatingModel-T1-Eval-v0` |

The task formerly named `momentum` is now `floating_model`. New training runs
use `logs/rsl_rl/t1_floating_model/`. To evaluate or resume a checkpoint saved
under the old experiment folder, use the new task ID and add
`--experiment_name t1_momentum`. Existing run directories and comparison NPZs
retain their original names. The comparison plotter accepts `--floating_model`
(`--momentum` remains an alias) and labels this controller as "Floating model".

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

Use the new Floating Model task with the existing RSL-RL training script:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-FloatingModel-T1-v0 \
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
  --task Tracking-FloatingModel-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name floating_model_g18_push_kick_right
```

This loads the motion locally and logs training metrics to W&B using the API
key supplied to the command. The input must be the converted file
containing joint velocities and body transforms; the raw retargeted NPZ and
terminal logs cannot be used directly for training.

### Train the Atlas controller

Use W&B logging in the same `hybrid_mimic` project, with GPU 0 exposed:

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
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name atlas_g18_push_kick_right
```

Simulation, analytical dynamics, batched QP solves, and PPO training use the
same exposed GPU. Atlas training requires CUDA and the `batched` backend; it
rejects CPU/OSQP configurations. File loading, logging, and checkpoint exports
still use the host. This runs the default 30,000 iterations
and saves checkpoints under `logs/rsl_rl/t1_atlas/`. Train a fresh policy because
old 29-action Atlas checkpoints are incompatible with this 57-action interface.
The policy layout now matches HybridMimic/Floating Model; controller behavior
still differs, so matching dimensions do not establish policy transfer quality.
See [Atlas validation and evaluation](docs/atlas_controller.md) for the current
tracking limitations and evaluation commands.

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

To load and visualize a trained Floating Model policy from a W&B run:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-FloatingModel-T1-v0 \
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
  --task Tracking-FloatingModel-T1-Eval-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

To evaluate a specific checkpoint:

```bash
python scripts/rsl_rl/tracking_play.py \
  --task Tracking-FloatingModel-T1-Eval-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --checkpoint_no ITERATION \
  --headless
```

Tracking data is written to:

```text
eval_data/tracking_play_data.npz
```

### Compare PD and floating_model with video recording

Run these commands from the repository root to evaluate the local PD and floating_model
runs below. Both use the same reference motion and checkpoint iteration.
Each evaluation overwrites `eval_data/tracking_play_data.npz`, so copy its output
before running the other controller:

```bash
conda activate hybridmimic
mkdir -p eval_data/comparison

python scripts/rsl_rl/tracking_play.py \
  --task Tracking-Flat-T1-Eval-v0 \
  --num_envs 2 \
  --load_run 2026-09-07_19-19-43_momentum_g18_push_kick_right \
  --checkpoint model_29999.pt \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --headless --video --video_length 500 &&
cp eval_data/tracking_play_data.npz eval_data/comparison/pd.npz

python scripts/rsl_rl/tracking_play.py \
  --task Tracking-FloatingModel-T1-Eval-v0 \
  --num_envs 2 \
  --experiment_name t1_momentum \
  --load_run 2026-09-09_00-17-56_momentum_g18_push_kick_right \
  --checkpoint model_29999.pt \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --headless --video --video_length 500 &&
cp eval_data/tracking_play_data.npz eval_data/comparison/momentum.npz
```

The PD run is under `logs/rsl_rl/t1_flat`, despite having `momentum` in its name.
Videos are saved in each run's `videos/play/` directory. The `&&` copies the data
only when evaluation exits successfully; if shutdown fails after saving data,
verify the output belongs to that evaluation and copy it before starting the next run.

After both NPZ files are saved, generate comparison plots without launching Isaac Sim:

```bash
python scripts/compare_tracking.py
```

This writes `tracking_errors.png`, `body_position_rmse.png`, `kick_height.png`,
`control_effort.png`, and `metrics.csv` to `eval_data/comparison/plots/`.
The script checks that reference trajectories match and trims trailing zero-filled
frames from legacy recordings. Time uses the default 0.02-second control interval
(override with `--dt` if needed). Shading shows the range across environments,
not a confidence interval. Absolute joint power is mechanical effort, not electrical
consumption. These plots describe the saved rollouts and do not measure failure rates.

### Contact-model diagnostics

Plot each foot's height, horizontal speed, vertical velocity, and angular speed:

```bash
python scripts/plot_contact_diagnostics.py
```

The existing NPZs contain only kinematic proxies, not measured contact states.
The plotter uses saved body-name metadata when available. For legacy T1 files it
corrects the old evaluator's order (trunk, left foot, right foot, left hand,
right hand); new evaluations explicitly save the requested body order.

To add normal contact-force measurements and floating_model QP wrench predictions,
rerun evaluation with `--record_contacts`:

```bash
python scripts/rsl_rl/tracking_play.py \
  --task Tracking-FloatingModel-T1-Eval-v0 \
  --num_envs 2 \
  --experiment_name t1_momentum \
  --load_run 2026-09-09_00-17-56_momentum_g18_push_kick_right \
  --checkpoint model_29999.pt \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --headless --record_contacts &&
cp eval_data/tracking_play_data.npz eval_data/comparison/momentum_contacts.npz

python scripts/plot_contact_diagnostics.py \
  --input eval_data/comparison/momentum_contacts.npz
```

Figures go to `eval_data/comparison/contact_plots/`, one per environment.
Green shading marks net normal force magnitude above 10 N (override with
`--contact_threshold`); red lines mark reset samples, which should be excluded
from physical interpretation. Sensor forces are sampled after each control step;
QP predictions come from the final physics substep. This 50 Hz recording can miss
brief impacts; impact and acceleration-residual studies need physics-rate logging.
Isaac Lab 2.2 `net_forces_w` contains summed normal forces, not friction forces or
contact moments. It cannot establish a measured friction ratio or center of pressure.
The sensor is not ground-filtered, so other collisions can also activate it.

The current floating_model QP enforces floating-base dynamics, but does not impose
`J_c qdd + Jdot_c qdot = 0`, unilateral/friction constraints, or a support polygon.
It includes a penalized auxiliary base wrench and wrench variables for every
configured end effector, without explicit contact activation. Compare predicted
normal forces with measured contact timing before attributing tracking errors to
a rigid flat-foot constraint. Motion during force-detected contact can motivate
investigating sliding or rocking, but body velocity is not sole-point slip velocity.

## Impulse Evaluation

Use `eval_env.py` to measure recovery from randomized external pushes:

```bash
mkdir -p eval_data

python scripts/rsl_rl/eval_env.py \
  --task Tracking-FloatingModel-T1-v0 \
  --num_envs 36 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

Floating Model impulse results are written to:

```text
eval_data/floating_model_eval_data.npz
```

## Implementation

The Floating Model task is implemented as a sibling of the original hybrid task, so the existing flat and centroidal-hybrid tasks are left unchanged.

Task registration lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_floating_model/__init__.py
```

It registers:

```text
Tracking-FloatingModel-T1-v0
Tracking-FloatingModel-T1-Eval-v0
```

The task config lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_floating_model/flat_env_cfg.py
```

It subclasses the existing T1 hybrid config so the new task keeps the same robot, observations, action layout, rewards, command setup, and RSL-RL structure.

The environment lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_floating_model/floating_model_env.py
```

`FloatingModelEnv` subclasses the existing `HybridEnv` and overrides only `_initialize_hybrid_runtime()`. That swap replaces the original `HybridController` with `FloatingModelController`, while preserving the existing step loop, action manager, reward bookkeeping, and PD-plus-feedforward torque application.

The controller lives in:

```text
source/whole_body_tracking/whole_body_tracking/utils/floating_model.py
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
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_floating_model/controller_cfg.py
```

The RSL-RL config lives in:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_floating_model/agents/rsl_rl_ppo_cfg.py
```

It inherits the hybrid PPO settings and changes the experiment name to:

```text
t1_floating_model
```

Impulse evaluation was updated in:

```text
scripts/rsl_rl/eval_env.py
```

so `Tracking-FloatingModel-T1-v0` writes its own result file instead of being grouped with the PD baseline.

## Notes

This is a practical Isaac Lab implementation of the paper's momentum-based control structure, not a line-for-line copy of the Atlas controller. The paper formulates a QP over desired generalized accelerations and contact-force basis multipliers, then computes torques through inverse dynamics. This implementation follows that structure using the runtime model quantities available from Isaac/PhysX for the Booster T1.

The controller requires Isaac/PhysX to expose:

```text
root_physx_view.get_generalized_mass_matrices()
root_physx_view.get_gravity_compensation_forces()
root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
root_physx_view.get_jacobians()
```

If your Isaac Lab build does not expose generalized mass matrices, the task will raise a clear runtime error when the Floating Model controller initializes.
