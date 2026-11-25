#!/usr/bin/env bash
# Activate hybridmimic first; trailing arguments override these defaults.
set -euo pipefail

FORCE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$FORCE_REPO_ROOT"

exec "${FORCE_PYTHON:-python}" scripts/rsl_rl/train.py \
  --task Tracking-WBC-FORCE-T1-v0 \
  --motion_dir retargeted_motion/training_list \
  --num_envs 1024 \
  --max_iterations 30000 \
  --device cuda:0 --headless \
  --logger wandb \
  --log_project_name hybrid_mimic \
  --run_name force \
  agent.resume=false \
  "$@"
