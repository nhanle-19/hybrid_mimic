import gymnasium as gym
from . import flat_env_cfg

for task_id, cfg in [('Tracking-Atlas-T1-v0', flat_env_cfg.T1AtlasEnvCfg),
                     ('Tracking-Atlas-T1-Eval-v0', flat_env_cfg.T1AtlasEnvEvalCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:T1AtlasPPORunnerCfg'})
