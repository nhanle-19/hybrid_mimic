#!/usr/bin/env bash
# Compatibility entry point for the default FORCE training launcher.
set -euo pipefail

FORCE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$FORCE_REPO_ROOT"

exec bash "$FORCE_REPO_ROOT/scripts/train_force.sh" "$@"
