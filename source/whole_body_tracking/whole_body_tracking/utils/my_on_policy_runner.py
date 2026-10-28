import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        metrics = getattr(self.env.unwrapped, '_wbc_force_metrics', None)
        if metrics is None:
            return super().log(locs, width, pad)
        values = metrics.drain()
        policy = self.alg.policy
        std = policy.std if hasattr(policy, 'std') else policy.log_std.exp()
        for i, side in enumerate(('left', 'right')):
            values[f'WBCForce/Policy/{side}_noise_std'] = float(std[i].detach())
        for name, value in values.items():
            self.writer.add_scalar(name, value, locs['it'])
        # FORCE's old reset-time means and per-step termination counts are
        # replaced by rollout totals with explicit, population-weighted units.
        # Keep the standard PPO losses/reward/length curves and concise output.
        forwarded = dict(locs, ep_infos=[])
        super().log(forwarded, width, pad)

        def number(name, suffix='', digits=1):
            value = values.get('WBCForce/'+name)
            return 'n/a' if value is None else f'{value:.{digits}f}{suffix}'

        print('  FORCE: full-motion success '+number('FullMotion/success_percent', '%')
              +' ('+number('FullMotion/episode_count', digits=0)+' episodes), suffix success '
              +number('Episode/suffix_success_percent', '%')+', mean start '
              +number('Episode/start_phase_percent', '%')+' into clip')
        print('  Swing zero cap L/R: '+number('Contact/left_swing_zero_cap_percent', '%')+' / '
              +number('Contact/right_swing_zero_cap_percent', '%')+'; swing mean cap L/R: '
              +number('Contact/left_swing_cap_mean', digits=3)+' / '+number('Contact/right_swing_cap_mean', digits=3))

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
