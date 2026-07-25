
<h1>
  HybridMimic: Hybrid RL-Centroidal Control for Humanoid Motion Mimicking
</h1>

<p>
  <a href="mailto:tay3@purdue.edu">Ludwig Chee-Ying Tay</a><sup>1</sup>,
  <a href="mailto:chang970@purdue.edu">I-Chia Chang</a><sup>2</sup>,
  <a href="https://engineering.purdue.edu/ME/People/ptProfile?resource_id=273141">Yan Gu</a><sup>2,†</sup>
</p>

<p>
  <sup>1</sup> Department of Computer Science, Purdue University<br>
  <sup>2</sup> School of Mechanical Engineering, Purdue University<br>
  <sup>†</sup> Corresponding author
</p>

<p>
  <strong>Preprint, 2026.</strong>
</p>

<p>
  <a href="https://arxiv.org/pdf/2603.06775">Paper</a> |
  <a href="https://arxiv.org/abs/2603.06775">arXiv</a> |
  <a href="https://youtu.be/1d5vkqNtCOY">Video</a>
</p>

---

## Overview

HybridMimic is a humanoid motion-mimicking framework that combines reinforcement learning with a
centroidal-model-based controller. In addition to joint-position targets, the policy predicts desired centroidal
velocities, continuous foot-contact states, and reference torques. The centroidal controller uses these predictions
to generate physically grounded feedforward torques, which are combined with joint-space PD torques. This removes
the need for a predefined contact schedule while physics-informed rewards encourage consistent ground-reaction
forces, contact estimates, centroidal accelerations, and torque-limit compliance. The paper demonstrates the method
on the Booster T1 and reports a 13% reduction in average sim-to-real base-position tracking error over a standard
RL/PD baseline.

This repository is the Isaac Lab training implementation for HybridMimic and also includes evaluation utilities. It
is based on the
[BeyondMimic motion-tracking training repository](https://github.com/HybridRobotics/whole_body_tracking), which
provides the underlying motion-tracking environment and RSL-RL training structure. If you use this repository,
please cite both HybridMimic and BeyondMimic as described in the [Citation](#citation) section.

## Installation

- Install Isaac Lab v2.1.0 by following the
  [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation because it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (that is, outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:purdue-tracelab/hybrid_mimic.git

# Option 2: HTTPS
git clone https://github.com/purdue-tracelab/hybrid_mimic.git

cd hybrid_mimic
```

- Using a Python interpreter that has Isaac Lab installed, install the library:

```bash
python -m pip install -e source/whole_body_tracking
```

## Training

Run all commands from the repository root with the Python environment used to install Isaac Lab. Training downloads
the reference motion from Weights & Biases, so authenticate first:

```bash
wandb login
```

The repository provides two training environments:

| Controller | Task |
| --- | --- |
| Joint-space PD | `Tracking-Flat-T1-v0` |
| Hybrid | `Tracking-Hybrid-T1-v0` |

Pass the Weights & Biases motion artifact to `--registry_name`. If the artifact path does not contain an alias, the
training script uses `:latest`.

For the joint-space PD environment:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-T1-v0 \
  --registry_name ENTITY/PROJECT/MOTION_ARTIFACT:latest \
  --headless \
  --logger wandb \
  --log_project_name PROJECT_NAME \
  --run_name RUN_NAME
```

For the hybrid environment:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Hybrid-T1-v0 \
  --registry_name ENTITY/PROJECT/MOTION_ARTIFACT:latest \
  --headless \
  --logger wandb \
  --log_project_name PROJECT_NAME \
  --run_name RUN_NAME
```

Remove `--headless` to run with the Isaac Sim GUI. The root-level
[`train.sh`](train.sh) and [`train_hybrid.sh`](train_hybrid.sh) files contain concrete examples.

## Evaluation

Evaluation can load the latest checkpoint from a Weights & Biases run using
`--wandb_path ENTITY/PROJECT/RUN_ID`. The motion artifact associated with that run is downloaded automatically.

### Policy playback

Use `play.py` to view a trained policy:

```bash
# Joint-space PD
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-T1-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID

# Hybrid
python scripts/rsl_rl/play.py \
  --task Tracking-Hybrid-T1-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID
```

Add `--headless` for headless playback, or `--video --video_length 500` to record a finite rollout. Playback also
exports the loaded policy to ONNX. See [`eval.sh`](eval.sh) and [`eval_hybrid.sh`](eval_hybrid.sh) for concrete
examples.

### Tracking-data evaluation

The evaluation task variants disable training-specific randomization and are intended for repeatable motion
tracking measurements:

| Controller | Evaluation task |
| --- | --- |
| Joint-space PD | `Tracking-Flat-T1-Eval-v0` |
| Hybrid | `Tracking-Hybrid-T1-Eval-v0` |

Create the output directory and run `tracking_play.py`:

```bash
mkdir -p eval_data

# Joint-space PD
python scripts/rsl_rl/tracking_play.py \
  --task Tracking-Flat-T1-Eval-v0 \
  --num_envs 256 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless

# Hybrid
python scripts/rsl_rl/tracking_play.py \
  --task Tracking-Hybrid-T1-Eval-v0 \
  --num_envs 2 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

Use `--checkpoint_no ITERATION` to evaluate a specific `model_ITERATION.pt` checkpoint. Results are written to
`eval_data/tracking_play_data.npz`; rename that file between runs if both controller types need to be retained.
Concrete examples are provided in [`eval_pd_2.sh`](eval_pd_2.sh) and
[`eval_hybrid_2.sh`](eval_hybrid_2.sh).

### Impulse evaluation

Use `eval_env.py` to measure recovery from randomized external forces:

```bash
mkdir -p eval_data

# Joint-space PD
python scripts/rsl_rl/eval_env.py \
  --task Tracking-Flat-T1-v0 \
  --num_envs 36 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless

# Hybrid
python scripts/rsl_rl/eval_env.py \
  --task Tracking-Hybrid-T1-v0 \
  --num_envs 36 \
  --wandb_path ENTITY/PROJECT/RUN_ID \
  --headless
```

The results are saved as `eval_data/pd_eval_data.npz` or `eval_data/hybrid_eval_data.npz`. See
[`eval_impulse.sh`](eval_impulse.sh) and [`eval_impulse_hybrid.sh`](eval_impulse_hybrid.sh) for concrete examples.

## Citation

If you use this repository, please cite the HybridMimic paper:

```bibtex
@article{tay2026hybridmimic,
  title   = {HybridMimic: Hybrid RL-Centroidal Control for Humanoid Motion Mimicking},
  author  = {Tay, Ludwig Chee-Ying and Chang, I-Chia and Gu, Yan},
  journal = {arXiv preprint arXiv:2603.06775},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.06775}
}
```

This training code builds on the
[BeyondMimic motion-tracking implementation](https://github.com/HybridRobotics/whole_body_tracking). Please also
cite the upstream work:

```bibtex
@article{liao2025beyondmimic,
  title   = {BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion},
  author  = {Liao, Qiayuan and Truong, Takara E. and Huang, Xiaoyu and Gao, Yuman and Tevet, Guy and
             Sreenath, Koushil and Liu, C. Karen},
  journal = {arXiv preprint arXiv:2508.08241},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.08241}
}
```
