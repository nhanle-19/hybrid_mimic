#!/usr/bin/env bash
# Rebuild the tested headless runtime on the destination Linux server.
set -euo pipefail
spider_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$spider_root"
spider_python="$spider_root/.venvs/spider/bin/python"
if [[ ! -x "$spider_python" ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv .venvs/spider
    elif command -v conda >/dev/null 2>&1; then
        conda create -y --prefix "$spider_root/.venvs/spider" python=3.12 pip
    else
        echo 'Load Python 3.12 (with venv) or Conda, then rerun this script.' >&2
        exit 1
    fi
fi
"$spider_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for this runtime"'
"$spider_python" -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
"$spider_python" -m pip install -r configs/spider_runtime.txt
"$spider_python" -m pip check
"$spider_python" - <<'PY'
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
