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

[WBC-AUTHORITY](docs/wbc_authority_controller.md) adds six bounded
translational contact-cost authorities (xyz per foot), with no rotational contact
cost or authority-dependent force/moment bounds. Physical contact constraints
and fixed normal-force caps remain, with separate training and evaluation tasks.
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

The NN outputs two raw force-cap actions, left then right. Decode each with
`c_i = clip((sigmoid(action_i)-0.1)/0.8, 0, 1)`; the QP enforces
`0 <= Fz_i <= 600 * c_i` N for each foot. Actual
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
WBC-ACC. FORCE rewards executed motion completion (100 points for a full clip),
minus a nonzero-cap penalty during reference swing and a decoded-action change
penalty. Initial penalty weights are 0.5 and 0.05, with all three terms normalized
by clip length. Training ends at the final reference frame; random phase starts
receive no credit for the skipped prefix. See the [reward definition](docs/wbc_force_controller.md#learning).
AUTHORITY retains its alive-only reward. Tracking and QP failure terminations
still apply. Checkpoints go to `logs/rsl_rl/t1_wbc_force/`. Train a fresh policy:
the two-output policy cannot load a fourteen-output WBC-ACC checkpoint.

For a small PPO-from-scratch test on the manually validated five-step walk,
use [first_force_walk.py](scripts/rsl_rl/first_force_walk.py). It verifies the
reference and label hashes/settings, applies the saved walk controller profile,
and uses the three-term FORCE reward with the existing network architecture.
The labels are evaluation baselines only; training receives no label loss or
pretrained policy. Run these stages in the `hybridmimic` environment on a CUDA host:

```bash
python scripts/rsl_rl/first_force_walk.py --stage check
python scripts/rsl_rl/first_force_walk.py --stage baselines
python scripts/rsl_rl/first_force_walk.py --stage smoke
python scripts/rsl_rl/first_force_walk.py --stage train
```

The baseline stage compares the manual table with constant mean, 500 N, and
1000 N caps under the same controller settings. Inspect its `comparison.json`
before training: completion alone is not sufficient; also compare swing-cap
penalties and deterministic output variation. The smoke test uses 32 environments and
two iterations; the first training run uses 256 environments and 1,000 iterations
(override with `--num-envs` and `--iterations`). Add `--dry-run` to inspect a
command. The launcher converts scalar controller values to floats when building
Isaac Lab overrides, including whole-valued gains in the saved JSON profile.
Judge output variation using the deterministic policy across the motion,
separately from exploration noise.

The training path was verified on an RTX 4060 Laptop GPU on October 25, 2026:
the 32-environment/two-iteration smoke test and a 256-environment/five-iteration
run both completed and saved checkpoints. The larger check executed 30,720
transitions with zero QP failures; actor and critic weights updated and all
logged scalar metrics remained finite. It used approximately 3.5 GiB of GPU
memory. This verifies startup and PPO updates; full-motion learned tracking
still requires training and deterministic evaluation. The check and its metrics
are in `logs/rsl_rl/t1_wbc_force/2026-10-25_19-12-38_force_walk_first_train/`,
including `readiness_validation.json`.

The separate GPU baseline check remains incomplete: the saved CPU manual
schedule terminated at step 211 of 261 on `ee_body_pos`, with zero QP failures.
Constant mean and 1,000 N caps survived all 261 steps, but did not reproduce the
left-foot lifts. The baseline simulator also exited with signal 11 during
shutdown after writing all four trials; both PPO validation runs exited cleanly.
The baseline traces and postprocessed `comparison.json` are in
`outputs/force_walk_first_run/20261025_191316_302556_baselines/`.
Use the command below for a training experiment; a full five-step walking result
has not yet been demonstrated on this GPU.

Start a fresh walk training run from the repository root:

```bash
conda activate hybridmimic
python scripts/rsl_rl/first_force_walk.py --stage train \
  --device cuda:0 --num-envs 256 --iterations 1000 --seed 42
```

| Setting | Walk training value |
| --- | --- |
| Reference | `retargeted_motion/hdm_bk_walk_in_place_settled_t1_training.npz` (261 frames, 5.22 s) |
| Environments / iterations | 256 / 1,000 |
| Rollout / PPO minibatch | 24 steps per environment; 6,144 transitions per rollout; 1,536 per minibatch |
| PPO epochs / minibatches | 5 / 4 |
| Actor and critic hidden layers | 1,024 → 512 → 256 → 256, ELU; observation normalization enabled |
| Initial learning rate / schedule | 0.001 / adaptive KL, target 0.01 |
| Discount / GAE lambda / PPO clip | 0.99 / 0.95 / 0.2 |
| Entropy / value loss / gradient norm limit | 0.005 / 1.0 / 1.0 |
| Initial action standard deviation | 1.0 |
| Physics / policy and QP rate | 500 Hz / 50 Hz |
| Per-foot force maximum / friction | 1,000 N / 0.6 |
| Posture position gain / velocity gain / weight | 200.0 / 28.284271247461902 / 3.0 |
| Reward weights: completion / swing cap / cap change | 1.0 / −0.5 / −0.05 |
| Checkpoint interval / logging | Every 100 iterations and at completion / TensorBoard |

The launcher applies the complete saved controller profile automatically.
Checkpoints and events are written under `logs/rsl_rl/t1_wbc_force/`.

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

All nine current FORCE cases are linked in the
[FORCE video gallery](eval_data/force/videos.html), with the reference on the
left and actual physics on the right. The six toy comparisons use the original
recordings; walk-in-place retains its 0.5x playback. Crawl and turn-right use
their existing comparisons. The presentation change does not alter simulations.

The ACCAD E1 **Turn around right** case is in
`eval_data/reference_e1_turn_right/short_reanchor/`. Its single current FORCE
result is the [complete CPU video](eval_data/reference_e1_turn_right/short_reanchor/force/video.mp4):
661 frames / 13.22 seconds, one left-foot step, a stationary ending, zero resets
and zero QP failures. Independent CPU replays match poses, torques and forces
exactly. Static, dynamic and QP friction are all **0.3**.

The editable decoded NN weights are in `configs/turn_reanchor_manual_outputs.json`;
`scripts/turn_reanchor_manual_outputs.py` exports the standard planned table.
Smooth unloading phases are retained. The left output stays exactly zero through
frame 480, restores gradually to an 85 N ceiling over frames 481–490, then returns
to the 800 N stance ceiling over frames 500–540. Every unavailable foot has zero
weight in the recorded replay. Runtime manual-output replacement has been removed
from the evaluator. The reference, QP formulation and controller gains are unchanged.

This is a stable preliminary result, with remaining tracking limits: peak root
error 6.73 cm, touchdown 0.82 s late, and final sole corner height 6.09 mm. Strict
tracking does not fully pass. `force/validation.json`, `force/verification.json`
and `force/short_reanchor_audit.json` report these separately. The current input,
video, diagnostics and replay command are together in `force/`; no partial-video
or alternate-method folder remains. Obsolete results and the old label generator
are archived under `outputs/turn_reanchor_cpu_tuning_20261023/`.

The ACCAD Female1 A11 **forward crawl** case is
`eval_data/reference_crawl_forward/`, with shared `reference.npz` and
`reference.mp4` and the [eight-channel FORCE results](docs/crawl_force_qp.md)
in `force/`. Its canonical reference copy is
`retargeted_motion/a11_crawl_forward_female1_t1.npz`.

The 599-frame reference was smoothed and projected back onto joint and floor
constraints. Joint jerk RMS fell 65%; root changes are at most 6.1 mm and
2.51 degrees. Timing stays unchanged. All frames pass independent geometric
checks and necessary acceleration/force/torque feasibility checks at friction
0.2. The stricter dissipative-friction check passes 580/599 frames, so this is
still a preliminary reference. Both hands and knees remain geometrically
grounded; source hand recovery lifts have not been restored.

Contact annotation version 3 checks every visible mesh vertex across 16 body
regions. The smoothed reference has six regions touching somewhere: both hands,
both knees, left forearm and right foot. The controller retains eight NN slots,
including right forearm and left upper arm, with manually tuned outputs.
Those outputs directly set normal-force ceilings; no second reference-label
mask is applied. Actual geometry removes unavailable support.

The corrected manual-weight table completes all **599 frames / 19.992 seconds**
on CPU, with **zero QP failures** and no positive output at an unavailable
contact in the recorded rollout. It travels 1.204 m; peak root error is
0.382 m and peak joint RMSE is 4.00 degrees. This remains a
preliminary result under the 0.5 m root-error cutoff.

The fixed input has 2,396 samples at 119.85 Hz for 599 reference frames.
Decoded weights taper through short quintic ramps (two QP intervals,
16.7 ms) into exact-zero OFF plateaus, then restore. Both knees remain
geometrically available throughout. No runtime manual-output override is used.
Reference, pose gains and static/dynamic/QP friction **0.2** remain unchanged.

Height overshoot is 4.12 mm and peak downward speed is
0.093 m/s. The hand-force peak is 309 N at every physics
step (141 N at QP sampling). A brief incidental head touch peaks at
3.64 N; incidental arm/right-foot contacts remain. All 2,396 runtime QPs satisfy
the force/torque constraints and decoded OFF forces are exactly zero.

This revised table has one completed physical validation; the previous
table's exact-repeat evidence is historical. `force/manual_weight_fix.json`
and `force/verification.json` record the current scope and comparison.

`force/` contains full planned/executed labels, diagnostics, validation, plots,
`video.mp4`, `preview.gif`, and the exact replay command. `case.json` and
`reproduction.json` describe the case and its provenance. Raw tuning trials
remain outside `eval_data`, in `outputs/crawl_smooth_tuning_20261023/` and
`outputs/crawl_soft_landing_20261023/`.
Earlier work is retained in `outputs/crawl_force_archive_20261023/` and
`outputs/crawl_preliminary_20261023/`. See the [diagnosis and limitations](docs/crawl_force_diagnosis.md)
for contact evidence and solver-check scope. Existing two-foot Gym tasks remain separate.

The [crawl overlay video](eval_data/reference_crawl_forward/force/overlay/video.mp4)
shows the reference in translucent orange with the solid actual motion overlaid,
preserving both trajectories' world positions and timing. Reproduce it with
`MUJOCO_GL=osmesa python scripts/render_crawl_overlay.py eval_data/reference_crawl_forward/force/diagnostics.npz --output eval_data/reference_crawl_forward/force/overlay`.

## Evaluation

WBC training and evaluation prepare the reference consistently: resets use the
robot's physical joint limits, penetrating reference frames are raised using the
actual sole geometry, and body transforms and velocities are rebuilt. The source
NPZ is unchanged. Preparation statistics are printed at startup and saved in
evaluation diagnostics. This changes the observations and targets; start a fresh
training run instead of resuming a checkpoint trained on the unprepared clip.

The batched QP uses a `1e-6` scaled optimality tolerance, while the separate
physical feasibility tolerance remains `2e-5`. AUTHORITY values affect only
translational contact costs; zero authority does not remove support forces.
`--force_full_contact --qp_only` is an evaluation diagnostic that bypasses actual
contact geometry; it is not a training fix or a physical contact model.

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
QP-only evaluations have no checkpoint folder and use `eval_data/<controller>/`. Discarded test directories were cleaned up on 2026-10-20;
selected replay artifacts remain in place. `eval_data/cleanup_20261020.json`
records the removed folders and hashes of retained results. Historical reports
may name removed trials; the final slide replay commands below use retained files.
Explicit `--video_dir` and `--output` options override these defaults.

### Manual actions over a complete motion

See [per-frame FORCE and AUTH schedules](docs/manual_wbc_schedule.md) for the
142-frame kick experiment, saved action NPZs, first-failure CPU results, and
one-command playback. These schedules have not completed the kick successfully.

### Toes-up standing reference

Generate a stationary pose with the left heel grounded, toes raised 15 degrees,
and the right foot flat. Hip/knee pitch compensate for the ankle tilt to keep
the heel grounded. The generator exports a reference without a physics rollout.

```bash
conda activate hybridmimic
python scripts/toy_standing_reference.py --device cpu --headless --tilt_deg 15 --seconds 10
```

The output is `eval_data/toy_standing/reference.npz`: a 10-second stationary clip
at 50 FPS, in simulator joint/body order with zero velocities. Use `--side right`
to switch feet or `--tilt_deg 0` for flat feet; supported angles are 0–30 degrees.

Record the reference as a video (direct pose playback, without physics):

```bash
python scripts/replay_npz.py --device cuda:0 --headless --video \
  --motion_file eval_data/toy_standing/reference.npz \
  --output_file eval_data/toy_standing/reference.mp4
```

For interactive playback, omit `--headless --video --output_file ...`.
The same NPZ is ready for `--motion_file` in `Standing-WBC-FORCE-T1-v0` and
`Standing-WBC-AUTHORITY-T1-v0`. NN/manual force and authority settings are left
for the subsequent WBC experiment. Do not use `--nominal_standing`, which would
replace this reference with a symmetric pose.

To add sideways CoM motion and a matching torso lean while retaining the same
heel/sole contacts:

```bash
python scripts/toy_standing_reference.py --sway --device cpu --headless \
  --com_sway_m 0.020 --torso_roll_deg 2.4 --period_s 4 --seconds 10
python scripts/replay_npz.py --device cuda:0 --headless --video \
  --motion_file eval_data/toy_standing_sway/reference.npz \
  --output_file eval_data/toy_standing_sway/reference.mp4
```

This writes `eval_data/toy_standing_sway/reference.npz`. The actual mass-weighted
CoM moves up to 2 cm left/right at constant x and height; the torso leans up to
2.4 degrees toward the shift. Leg IK preserves both foot positions and rotations,
including the 15-degree toes-up foot. Each clip starts and ends with a 0.5-second
neutral hold, with one-second smooth amplitude ramps. Body and joint velocities
are derived from the moving poses. `--tilt_deg 0` gives two flat feet. Sway clips
need at least four seconds; shorter periods than two seconds are rejected.
Joint-limit or IK failures require reducing the requested sway/lean.

Add `--stance_widen_m 0.02` to the sway command to move each foot 1 cm outward.
IK preserves foot heights and rotations, including the raised toes, and the
original CoM trajectory. Use a separate `--output_dir` when comparing stance
widths; this changes reference geometry and requires a physics rollout to check balance.

The saved standard and widened sway cases each include a 10-second kinematic
reference video beside their motion file: `eval_data/toy_standing_sway/reference.mp4`
and `eval_data/toy_standing_sway_wide02/reference.mp4`. These preview the requested
motion; the controller videos show the separate physics runs.

For a normal upright pose with the left foot sliding sideways on the floor:

```bash
python scripts/toy_standing_reference.py --slide --side left --device cpu --headless \
  --slide_amplitude_m 0.02 --period_s 4 --seconds 10
python scripts/replay_npz.py --device cuda:0 --headless --video \
  --motion_file eval_data/toy_left_foot_slide/reference.npz \
  --output_file eval_data/toy_left_foot_slide/reference.mp4
```

Only the left leg changes: the foot moves 2 cm either side of neutral with its
sole flat on the floor. Root position/orientation, upper-body joints, and the
right leg remain fixed. The clip has smooth start/end ramps and neutral holds.
`--side right` swaps the moving leg. Slide mode defaults to zero heel tilt and
rejects nonzero `--tilt_deg`. IK checks the actual joint limits and collider
heights. The video is direct reference playback; physics tracking under floor
friction must be evaluated separately. For AUTH, the moving foot's lateral
contact cost can oppose the intended slide even when its vertical cost is useful.

The original upright slide and the first right-shifted version did not track
successfully. The loaded foot stayed planted, or drifted when unloaded; the
upright reference also exceeded the right hip-roll torque limit during sampled
outward slides at friction 0.6. Those discarded test folders and parameter
searches were removed after selecting the support-shifted runs below.

The final sliding case is grouped in `eval_data/left_foot_slide/`: shared
`reference.npz` and `reference.mp4`, with FORCE results in `force/` and AUTH
results in `auth/`. Each controller folder contains `video.mp4`,
`diagnostics.npz`, its labels and validation report. The reference video is
kinematic playback; the controller videos show the saved physics runs.

The root starts 8 cm toward the right,
2 cm lower, and with a 5-degree rightward roll. Both feet retain the original
flat-ground paths and 40 mm target travel. This small lean removes the hip
torque bottleneck: the sliding reference passed all 40 sampled dynamic
feasibility checks at friction 0.6. The selected labels release the left normal
force cap to zero over 0.28–0.48 s, keep the right cap at 800 N during motion,
and restore the standing labels over 9.50–9.70 s.

The initial replay with default posture gains achieved 36.93 mm travel and
4.69 mm XYZ tracking RMS. Its metrics remain in `force_validation.json` for
comparison; the superseded replay folder was removed. The final selected run
is the refined FORCE replay below. It starts in the shifted pose; transfer
from centred standing has not been validated.

The refined FORCE replay uses the **same reference and force-label table** with
posture gains `Kp=1600`, `Kd=40` (previously 40 and 8), retaining posture weight
0.1 and friction 0.6. Two batches tested 76 timing, loading, and feedback
candidates. Label-only changes reduced tracking error but also shortened the
slide; the selected feedback tuning preserved 39.59 mm of the 40 mm target
travel. A fresh 500-step replay reduced XYZ RMS from 4.686 mm to 0.416 mm
(91.1%), with maximum error 0.700 mm, no resets, QP failures, or torque
saturation. Direct replay measurements show maximum root error 0.243 mm and
sole-height error 0.021 mm. These are slide-specific overrides; production
controller defaults remain unchanged.

The refined schedule, `force_refined_validation.json`,
`force_refined_comparison.png`, `diagnostics.npz`, and `video.mp4` are in the same
`eval_data/left_foot_slide/force/` folder. Replay with both required gains:

```bash
python scripts/rsl_rl/eval_wbc.py --task Toy-Standing-WBC-FORCE-T1-v0 \
  --qp_only --track_motion --num_envs 1 --steps 500 --fail_on_reset \
  --motion_file eval_data/left_foot_slide/reference.npz \
  --manual_schedule eval_data/left_foot_slide/force/force_refined_schedule.npz \
  --force_max 1000 1000 --device cuda:0 --headless --video \
  --output eval_data/left_foot_slide/force/diagnostics.npz \
  --video_dir eval_data/left_foot_slide/force \
  env.actions.wbc_force.controller.posture_position_gain=1600.0 \
  env.actions.wbc_force.controller.posture_velocity_gain=40.0
```

AUTH also tracked this support-shifted reference when given an **additional
fixed left normal-force maximum of 0.01 N** (right maximum 600 N). Its active
cost labels are left `(0, 0, 1)` and right `(0, 0, 0)`, with the same release
and restore ramps. The independent 500-step replay had no resets, QP failures,
or torque saturation: left travel was 40.71 mm and XYZ tracking RMS 3.46 mm.
The measured active left load averaged 0.083 N; the batch sole-height error
stayed below 0.484 mm. Results, named labels, and the physics video are in
`eval_data/left_foot_slide/auth/`; `comparison.png` records the earlier default-gain FORCE replay and AUTH
against the common reference; the final FORCE result is in the sibling `force/` folder.

This AUTH result depends on the explicit fixed unloading bound. All 64 cost
label candidates at the default 600/600 N caps remained planted in the same
support pose. Larger weight transfers were also tested and did not produce a
successful default-cap AUTH rollout. The successful experiment does not show
that AUTH cost labels alone can learn unloading. Its six-output formulation
and production defaults remain unchanged; the bound is an evaluation override.

Replay AUTH with that required override:

```bash
python scripts/rsl_rl/eval_wbc.py --task Toy-Standing-WBC-AUTHORITY-T1-v0 \
  --qp_only --track_motion --num_envs 1 --steps 500 --fail_on_reset \
  --motion_file eval_data/left_foot_slide/reference.npz \
  --manual_schedule eval_data/left_foot_slide/auth/authority_slide_schedule.npz \
  --device cuda:0 --headless --video \
  --output eval_data/left_foot_slide/auth/diagnostics.npz \
  --video_dir eval_data/left_foot_slide/auth \
  'env.actions.wbc_authority.controller.contact_force_max=[0.01,600.0]'
```

For a flat foot rotating in place, the generator also supports `--rotate`:

```bash
python scripts/toy_standing_reference.py --rotate --side left --yaw_amplitude_deg 15 \
  --support_shift_m 0.08 --root_lower_m 0.02 --support_roll_deg 5 \
  --period_s 4 --seconds 10 --device cpu --headless \
  --output_dir eval_data/left_foot_rotation
```

This turns the left foot ±15 degrees about its fixed sole centre, keeping both
soles flat and the right foot fixed. The supported root pose matches the slide:
8 cm toward the right foot, 2 cm lower, and 5 degrees of rightward roll. The
root and upper-body pose remain constant during rotation. The foot-link origin
moves along a small arc because the pivot is the centre of the sole. The motion
has the same 0.5-second endpoint holds and smooth amplitude ramps as sliding.
Reference generation does not change the physical friction coefficient of 0.6.

The case uses `eval_data/left_foot_rotation/reference.npz` and `reference.mp4`,
with `force/` and `auth/` result folders. Each contains `labels.npz` / `labels.csv`,
`diagnostics.npz`, `validation.json`, and `video.mp4`. Labels retain the existing
schema: 500 rows at 50 Hz, two raw FORCE outputs or six raw xyz AUTH outputs.
The schedules store their required controller settings and motion fingerprints.

The accepted 500-step physics runs both completed without resets, QP failures,
or torque saturation. FORCE reached −14.89° / +14.87°, with 0.134° yaw RMS error
and 0.514 mm maximum sole-centre drift. AUTH reached −14.97° / +14.99°, with
0.067° yaw RMS error and 0.817 mm maximum drift. Maximum sole-height errors were
0.022 mm and 0.062 mm, respectively. Both use native posture gains 1600 / 40
with posture weight 0.1; these are case-specific overrides.

FORCE releases the left cap to zero before turning and keeps the right cap at
800 N during motion. AUTH uses the same six cost-label channels as before,
with active left `(0, 0, 1)` and right `(0, 0, 0)`, plus the **required fixed
left/right force maxima of 0.01 / 600 N**. The tested 600 N and 10 N left-bound
variants did not meet pivot-tracking checks. This accepted AUTH run therefore
does not demonstrate unloading through cost labels alone. Friction remains 0.6.

```bash
python scripts/rsl_rl/eval_wbc.py --task Toy-Standing-WBC-FORCE-T1-v0 \
  --qp_only --track_motion --num_envs 1 --steps 500 --fail_on_reset \
  --motion_file eval_data/left_foot_rotation/reference.npz \
  --manual_schedule eval_data/left_foot_rotation/force/labels.npz \
  --force_max 1000 1000 --device cuda:0 --headless --video \
  --output eval_data/left_foot_rotation/force/diagnostics.npz \
  --video_dir eval_data/left_foot_rotation/force \
  env.actions.wbc_force.controller.posture_position_gain=1600.0 \
  env.actions.wbc_force.controller.posture_velocity_gain=40.0
python scripts/rsl_rl/eval_wbc.py --task Toy-Standing-WBC-AUTHORITY-T1-v0 \
  --qp_only --track_motion --num_envs 1 --steps 500 --fail_on_reset \
  --motion_file eval_data/left_foot_rotation/reference.npz \
  --manual_schedule eval_data/left_foot_rotation/auth/labels.npz \
  --device cuda:0 --headless --video \
  --output eval_data/left_foot_rotation/auth/diagnostics.npz \
  --video_dir eval_data/left_foot_rotation/auth \
  env.actions.wbc_authority.controller.posture_position_gain=1600.0 \
  env.actions.wbc_authority.controller.posture_velocity_gain=40.0 \
  'env.actions.wbc_authority.controller.contact_force_max=[0.01,600.0]'
```

The replay commands and case settings are also in `case.json`; `comparison.png`
shows yaw, pivot drift and measured normal force. Discarded trials are kept
outside `eval_data`, with a compact record in `tuning_summary.json`.

For toe contact while rotating, add `--toe_pitch_deg 20`:

```bash
python scripts/toy_standing_reference.py --rotate --side left \
  --toe_pitch_deg 20 --yaw_amplitude_deg 15 \
  --support_shift_m 0.08 --root_lower_m 0.02 --support_roll_deg 5 \
  --period_s 4 --seconds 10 --device cpu --headless \
  --output_dir eval_data/toy_left_foot_toe_rotation
```

The left heel stays raised by 20 degrees (about 76 mm), while the toe-edge
midpoint stays fixed on the floor and the foot turns ±15 degrees. The right
sole stays flat. The clip starts in the supported, heel-raised pose; it does
not include the initial weight transfer or heel lift. Friction remains 0.6.

`eval_data/toy_left_foot_toe_rotation/` contains the shared `reference.npz` and
`reference.mp4`, with `force/` and `auth/` folders containing the actual physics
videos, diagnostics, validation reports, and manual `labels.npz` / `labels.csv`.
The tables keep the same format: 500 frames at 50 Hz, two FORCE outputs or six
xyz AUTH outputs. `case.json` records exact replay arguments and overrides;
`comparison.png` plots yaw, pitch, toe-pivot movement, toe/heel height and force.

Both physics runs completed all 500 steps with zero resets, QP failures or
torque saturation. FORCE/AUTH yaw RMS errors were 0.046° / 0.053°, maximum toe
pivot XY errors were 0.762 / 1.614 mm, and maximum toe-height deviations were
0.036 / 0.191 mm. In every frame, contact geometry admitted only the two left
toe vertices and all four right sole vertices.

Both use native posture gains 1600 / 40 and weight 0.1. FORCE phases its caps to
0 / 800 N during turning. AUTH uses active left `(0, 0, 1)` and right `(0, 0, 0)`
cost labels, with the additional **fixed 0.01 / 600 N force maxima** carried over
from flat rotation. This toe case validates that combined configuration; it
does not test whether the six AUTH cost labels alone can unload the left foot.
These are case-specific settings; global controller defaults are unchanged.

Follow the moving reference with the manual-output WBC toy task:

```bash
python scripts/rsl_rl/eval_toy_wbc.py --controller authority --track_motion \
  --device cuda:0 --headless --video
```

`--track_motion` selects the sway reference by default, advances through it once,
and uses its full length unless `--steps` is supplied. It loads the separate
`configs/toy_sway_wbc_inputs.json`; `--inputs` can select another input file.
Results go under
`eval_data/toy_standing_sway/authority_cost_only/` (or `force/`). The supplied
manual controller outputs remain constant. Without `--track_motion`, the toy
launcher uses `configs/toy_wbc_inputs.json` and retains its frame-zero pose hold.

For motion-dependent FORCE caps, the toy launcher also accepts the same
per-reference-frame manual schedules as the stepping experiments. Generate a
wider reference and build a separate cap table from its support geometry and
CoM load estimate:

```bash
python scripts/toy_standing_reference.py --sway --device cpu --headless \
  --stance_widen_m 0.02 --output_dir eval_data/toy_standing_sway_wide02
python scripts/rsl_rl/eval_toy_wbc.py --controller force --track_motion \
  --motion_file eval_data/toy_standing_sway_wide02/reference.npz \
  --build_manual_schedule eval_data/toy_standing_sway_wide02/force_reference_schedule.npz \
  --output_dir eval_data/toy_standing_sway_wide02/force_reference_schedule \
  --device cuda:0 --headless
```

To replay the saved table, replace `--build_manual_schedule` with
`--manual_schedule` and optionally add `--video`. Building refuses to overwrite
an existing schedule. The NPZ contains raw logits and fractions; the adjacent
CSV exports every frame. Each cap is a fraction of the 1000 N per-foot maximum,
and may vary independently with the motion. This mode replaces `--inputs` and
requires `--track_motion`. Playback checks the motion hash, controller settings,
and frame alignment, and stops on the first tracking or QP failure.

The widened 2 cm / 2.4-degree reference failed at step 148 with constant
120 N / 800 N caps. Its reference-derived schedule completed all 500 steps in
both the initial physics trial and a fresh-process video replay, with no reset
or QP failure. The scheduled ceilings
vary approximately 279–370 N left and 288–378 N right. These are upper bounds;
the controller chooses the forces. Detailed repeatability results are saved in
`eval_data/toy_standing_sway_wide02/verification.json`.

AUTH supports the same per-frame schedule playback on the widened reference:

```bash
python scripts/rsl_rl/eval_toy_wbc.py --controller authority --track_motion \
  --motion_file eval_data/toy_standing_sway_wide02/reference.npz \
  --manual_schedule eval_data/toy_standing_sway_wide02/authority_xyz_schedule.npz \
  --output_dir eval_data/toy_standing_sway_wide02/authority_xyz_replay \
  --device cuda:0 --headless --video
```

For a new table, use `--build_manual_schedule` with a new filename. AUTH varies
six xyz contact-cost authorities with estimated support load, clearance and
foot speed. Effective weights are `100 * authority**2`; the force ceilings
remain 600 N per foot. Unlike the earlier speed-only labels, this also varies
authority during fixed-foot sway. The schedule CSV exports the six raw outputs
and their fractions for every reference frame.

The selected `authority_xyz_schedule.npz` starts from xyz costs of 25 / 75 / 50
and varies them by the estimated load share. The generic builder produces the
initial `authority_reference_schedule.npz` table; the selected sway table is a
separate manual revision. These sway references keep both feet planted and do
not validate swing-foot lift-off. High z authority opposes the initial upward
acceleration while contact is available; stepping labels need to release the
contact cost before planned lift-off. Detailed AUTH measurements are saved in
`eval_data/toy_standing_sway_wide02/authority_verification.json`.

The selected smaller-sway settings use FORCE caps of 120 N left / 800 N right
(independent fractions 0.12 / 0.80 of 1000 N) and AUTH xyz contact weights of
25 / 75 / 50 on each foot. Both completed 500-step physics trials with no
termination in the original serial tuning trials. A later fresh FORCE replay
also completed 500 steps; its diagnostics are in
`eval_data/toy_standing_sway/force_recheck/`. The original selection and measured
tracking errors are saved in `eval_data/toy_standing_sway/selected_settings.json`.

`scripts/rsl_rl/tune_toy_wbc.py` compares constant input vectors in serial GPU
physics rollouts with the toy task's original termination checks. Supply
`--controller`, `--motion_file`, `--candidates` (a JSON list of fraction vectors),
and `--output_dir`, plus `--device cuda:0 --headless`. FORCE candidates have two
fractions, each multiplying its independent per-foot maximum; there is no
sum-to-one constraint. The sway FORCE launcher defaults to 1000 N per foot
(`--force_max LEFT_N RIGHT_N` overrides this). Pass `--force_max 1000 1000`
to the tuning script to match it. AUTH candidates have six xyz
authorities, with effective contact weights `100 * authority**2`. Trials rank
completion first, then CoM, torso and foot tracking errors normalized by 2 cm,
2.4 degrees and 2 cm respectively. It saves each diagnostic trace, `trials.json`
and `best_inputs.json`. A fresh simulator replay provides an additional
repeatability check. Kinematic checks alone do not establish closed-loop stability.

### Manual-output WBC toy tasks

Two dedicated tasks hold frame zero of the toes-up reference with zero reference
velocities: `Toy-Standing-WBC-FORCE-T1-v0` and
`Toy-Standing-WBC-AUTHORITY-T1-v0`. They use the existing WBC solvers and
direct-torque actuation, with deterministic initialization, no observation noise,
and no learned policy. Actual foot geometry determines the available support.
Both stop evaluation at the first fall/tracking termination or QP failure and
save available diagnostics. The episode timeout extends past the requested run.
The toy tasks use a 0.5 mm contact-patch height tolerance to avoid switching a
nearly level foot between sole, edge, and corner support due to GPU PhysX jitter.
Every vertex must still pass the 1 mm ground-gap check; the raised left toes
remain excluded. Solver tolerances and manual force limits are unchanged.

Manual NN outputs belong in `configs/toy_wbc_inputs.json`. Force is configured
with raw outputs `[-1.045969, 0.322773]`: 20% left and 60% right, giving normal-force
ceilings of 120 N and 360 N. Authority defaults to six zero logits, giving 0.5
cost authority on x/y/z for each foot (effective weights of 25 with base weights
of 100). This is a neutral test baseline, not a validated balance solution.
There is no rotational contact cost or authority-dependent wrench bound;
the fixed normal-force caps remain 600 N per foot. If the selected
controller has unset entries, the launcher exits before starting Isaac Sim;
it does not substitute defaults. Only that controller's entries need filling.

| JSON key | Raw NN output order |
| --- | --- |
| `force` | Left normal-force limit logit, right normal-force limit logit |
| `authority` | Left x/y/z cost-authority logits, then right x/y/z |

These are raw policy outputs, not forces in newtons or fractions in [0,1]. The
FORCE and AUTH controllers apply `clip((sigmoid(u)-0.10)/0.80, 0, 1)` internally.
Both endpoints are reachable: -3 maps to 0 and +3 maps to 1; zero maps to 0.5.
The selected vector is held
constant throughout the run and recorded as `manual_nn_outputs` in diagnostics.
Time-varying inputs and their values can be added when the experiment is defined.

The checked-in inputs are ready to run. Hold the existing toes-up pose with
the six-output cost-only AUTH controller:

```bash
conda activate hybridmimic
export PYTHONNOUSERSITE=1
python scripts/rsl_rl/eval_toy_wbc.py --controller authority \
  --device cuda:0 --headless --video --steps 500
```

The launcher prints the six raw logits and mapped xyz cost authorities before
starting the simulator. Adjust the `authority` array in
`configs/toy_wbc_inputs.json` to test another constant vector. The reference
NPZ is reused; no motion regeneration or trained checkpoint is needed. Run
the FORCE comparison by replacing `--controller authority` with
`--controller force`.

The default run is 500 control steps (10 seconds). Use `--steps`, `--motion_file`,
`--inputs`, or `--output_dir` to override these settings. Omit `--headless` for
the simulator window, or use `--device cpu` without video for CPU evaluation.
Outputs are separated into `eval_data/toy_standing/force/` and
`eval_data/toy_standing/authority_cost_only/`, each containing `diagnostics.npz` and, when
requested, a `videos/` directory. Unlike reference playback, these videos show
the robot's simulated response to the WBC torques.

Plot the new AUTH result with:

```bash
python scripts/plot_wbc.py --input eval_data/toy_standing/authority_cost_only/diagnostics.npz
```

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
