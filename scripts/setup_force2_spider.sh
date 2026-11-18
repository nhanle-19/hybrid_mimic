#!/usr/bin/env bash
# Rebuild the tested headless runtime on the destination Linux server.
set -euo pipefail
# A different active Python/Isaac environment may redirect standard-library
# imports even when we invoke this environment's interpreter by absolute path.
unset PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
spider_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$spider_root"
spider_python="$spider_root/.venvs/spider/bin/python"
if [[ ! -x "$spider_python" ]]; then
    if command -v conda >/dev/null 2>&1; then
        conda create -y --prefix "$spider_root/.venvs/spider" python=3.12 pip
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -I -m venv .venvs/spider
    else
        echo 'Load Python 3.12 (with venv) or Conda, then rerun this script.' >&2
        exit 1
    fi
fi
# A failed system-venv attempt followed by conda create at the same prefix can
# leave this file pointing Python at /usr instead of the Conda standard library.
# Keep real venvs intact; only remove the redirect from confirmed Conda prefixes.
if [[ -f .venvs/spider/conda-meta/history && -f .venvs/spider/pyvenv.cfg ]]; then
    spider_venv_backup="$(mktemp "$spider_root/.venvs/spider/pyvenv.cfg.before-conda.XXXXXX")"
    mv -- .venvs/spider/pyvenv.cfg "$spider_venv_backup"
    echo "Preserved stale venv configuration at $spider_venv_backup"
fi
"$spider_python" -I - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for this runtime'
import subprocess
import _posixsubprocess
print(f'Python runtime: {sys.executable}; subprocess: {subprocess.__file__}')
PY
"$spider_python" -I -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
"$spider_python" -I -m pip install -r configs/spider_runtime.txt
"$spider_python" -I -m pip check
"$spider_python" -I - <<'PY'
import sys
from pathlib import Path
import torch
import mujoco
import mujoco_warp
import warp
from pxr import Usd
sys.path.insert(0, str(Path('spider').resolve()))
from spider.config import Config
from spider.optimizers.sampling import make_optimize_fn
print(f'Runtime ready: Python {sys.version.split()[0]}, Torch {torch.__version__}, MuJoCo {mujoco.__version__}, Warp {warp.__version__}')
if torch.cuda.is_available():
    print(f'Visible GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU visible in this shell. Launch the retargeting command inside your GPU allocation.')
PY
