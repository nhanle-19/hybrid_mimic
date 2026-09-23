# HybridMimic WBC tasks

Two Booster T1 tasks learn reference-conditioned settings for a whole-body QP
controller. WBC-ACC learns force caps and foot-acceleration tracking weights;
WBC-FORCE learns only force caps and removes the foot-acceleration objectives.

| Use | Task |
| --- | --- |
| Training | `Tracking-WBC-ACC-T1-v0` |
| Evaluation | `Tracking-WBC-ACC-T1-Eval-v0` |
| Standing controller test | `Standing-WBC-ACC-T1-v0` |
| Force-cap training | `Tracking-WBC-FORCE-T1-v0` |
| Force-cap evaluation | `Tracking-WBC-FORCE-T1-Eval-v0` |
| Force-cap standing test | `Standing-WBC-FORCE-T1-v0` |

See [the controller formulation and validation commands](docs/wbc_acc_controller.md).
See [WBC-FORCE's learned force inequalities and commands](docs/wbc_force_controller.md).
The original PD and centroidal-hybrid tasks remain available.

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

### Train the WBC-ACC controller

Use W&B logging in the same `hybrid_mimic` project, with GPU 0 exposed:

```bash
conda activate hybridmimic
python -m pip install -r requirements-wbc_acc.txt

read -rsp "W&B API key: " WANDB_API_KEY
echo

CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY="$WANDB_API_KEY" python scripts/rsl_rl/train.py \
  --task Tracking-WBC-ACC-T1-v0 \
  --motion_file retargeted_motion/g18_push_kick_right_t1_training.npz \
  --device cuda:0 \
  --num_envs 1024 \
  --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name wbc_acc_g18_push_kick_right
```

Simulation, policy training, analytical dynamics, and batched QP solves use the
exposed GPU. WBC-ACC defaults to 1,024 parallel environments and 30,000 iterations
and saves checkpoints under `logs/rsl_rl/t1_wbc_acc/`. Train a fresh policy because
the 14-output contact-policy interface is incompatible with older action layouts. For each foot the reference-only actor produces support
activation and six motion weights. Actual collider geometry gates force
availability; all six contact-motion requests are soft. Tracking rewards and
fall termination are retained, with a measured-slip penalty. An alive reward adds
+1 per simulated second survived, excluding failure steps and including timeouts,
in place of the fixed termination penalty.
See [WBC-ACC validation and evaluation](docs/wbc_acc_controller.md) for the current
tracking limitations and evaluation commands.

### Train WBC-FORCE

The NN outputs two normalized normal-force capacities, left then right. The QP
enforces `0 <= Fz_i <= 600 * clip(action_i, 0, 1)` N for each foot. Actual
support geometry, friction, dynamics, and torque limits still apply. Momentum,
posture, and pelvis tracking remain; foot-acceleration objectives are absent.

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-WBC-FORCE-T1-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --device cuda:0 --num_envs 1024 --headless \
  --logger tensorboard --run_name wbc_force_side_step
```

This uses the same PPO settings and reference-only actor observations as
WBC-ACC. Checkpoints go to `logs/rsl_rl/t1_wbc_force/`. Train a fresh policy:
the two-output policy cannot load a fourteen-output WBC-ACC checkpoint.

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

## Evaluation

Evaluate a saved policy with controller diagnostics:

```bash
python scripts/rsl_rl/eval_wbc.py \
  --task Tracking-WBC-ACC-T1-Eval-v0 \
  --motion_file retargeted_motion/b22_side_step_left_female1_t1_training.npz \
  --load_run RUN_DIRECTORY \
  --checkpoint model_29999.pt \
  --num_envs 1 --steps 500 --device cuda:0 --headless

python scripts/plot_wbc.py --input logs/rsl_rl/t1_wbc_acc/RUN_DIRECTORY/diagnostics.npz
```

For a controller-only standing test, use `Standing-WBC-ACC-T1-v0` with
`--qp_only --manual_support 1 1 --manual_weight 50` instead of the checkpoint
arguments. Add `--video` to record the rollout.

The same evaluator supports `Tracking-WBC-FORCE-T1-Eval-v0` and
`Standing-WBC-FORCE-T1-v0`. For WBC-FORCE, `--manual_support LEFT RIGHT` sets
the two normalized force caps during `--qp_only`; `--manual_weight` does not
apply. For either learned policy, videos go to the checkpoint run's
`videos/play/` folder and diagnostics to its `diagnostics.npz`. WBC-FORCE runs
are under `logs/rsl_rl/t1_wbc_force/`. Plot a run with
`scripts/plot_wbc.py --input <run folder>/diagnostics.npz`.
QP-only evaluations have no checkpoint folder and use `eval_data/<controller>/`.
Explicit `--video_dir` and `--output` options override these defaults.

For generic tracking recordings, `scripts/rsl_rl/tracking_play.py` accepts the
same evaluation task, `--load_run`, `--checkpoint`, and `--motion_file`. It saves
`eval_data/tracking_play_data.npz`. Save PD and controller recordings separately
and compare matching reference trajectories with:

```bash
python scripts/compare_tracking.py \
  --pd eval_data/comparison/pd.npz \
  --controller eval_data/comparison/wbc_acc.npz \
  --controller_label wbc_acc
```

The preserved long training runs remain under `logs/rsl_rl/`. Saved parameter
snapshots and historical metrics retain their original labels; use the current
task ID for loading a compatible checkpoint. Runs from the removed controller
are retained as historical artifacts when they have at least ten checkpoints.
