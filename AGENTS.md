# Repository guidance

## Training preferences

- Always recommend Weights & Biases (`--logger wandb`) with
  `--log_project_name hybrid_mimic` for training. Use another logger only when
  the user explicitly requests it. Apply this to smoke runs as well.
- The user's "default", "flat", or "BeyondMimic" baseline means
  `Tracking-Flat-T1-v0`. Use an explicit `--task` in suggested commands;
  `train.py` currently defaults to WBC-FORCE, which is a separate task.
- `Tracking-Hybrid-T1-v0` is the separate centroidal-hybrid task. Do not
  substitute it or WBC-FORCE for the flat baseline.
- Flat training currently supports one motion per run. For side-step training,
  use `retargeted_motion/training_list/20_side_step_left.npz` or
  `retargeted_motion/training_list/21_side_step_right.npz`. Multi-motion flat
  training requires implementation changes; a folder is not currently supported.
- Run from the repository root with `conda activate hybridmimic`. Include
  `CUDA_VISIBLE_DEVICES` when giving server launch commands. With one physical
  GPU exposed, use `--device cuda:0` regardless of its physical index.
- Authenticate W&B with `wandb login` or an existing `WANDB_API_KEY`.
  Never put actual credentials in repository files.

## Server cache permissions

The server has previously failed during Warp initialization with
`PermissionError: [Errno 13] Permission denied: '/ephemeral/cache'`.
For server launch instructions, set writable cache paths after activating the
environment and from the repository root, in the same shell as training:

```bash
export XDG_CACHE_HOME="$PWD/.cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp/1.5.0"
mkdir -p "$WARP_CACHE_PATH"
```

Keep generated `.cache/` contents out of version control. The cache fix does
not require changing GPU selection or robot/controller gains.
