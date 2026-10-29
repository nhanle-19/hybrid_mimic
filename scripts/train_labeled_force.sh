#!/usr/bin/env bash
# Run after activating the hybridmimic environment; trailing arguments override defaults.
set -euo pipefail

FORCE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$FORCE_REPO_ROOT"

exec "${FORCE_PYTHON:-python}" scripts/rsl_rl/train.py \
  --task Tracking-WBC-FORCE-T1-v0 \
  --motion_file retargeted_motion/training_list \
  --num_envs 1024 \
  --max_iterations 7000 \
  --device cuda:0 --headless \
  --logger tensorboard \
  --run_name force_labeled_caps_only \
  env.episode_length_s=15.0 \
  agent.resume=false \
  "$@"
