import gymnasium as gym
from . import flat_env_cfg

for task_id, cfg in [('Tracking-WBC-AUTHORITY-T1-v0', flat_env_cfg.T1WBCAuthorityEnvCfg),
                     ('Tracking-WBC-AUTHORITY-T1-Eval-v0', flat_env_cfg.T1WBCAuthorityEnvEvalCfg),
                     ('Standing-WBC-AUTHORITY-T1-v0', flat_env_cfg.T1WBCAuthorityStandingEnvCfg),
                     ('Toy-Standing-WBC-AUTHORITY-T1-v0', flat_env_cfg.T1WBCAuthorityToyStandingEnvCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:T1WBCAuthorityPPORunnerCfg'})
