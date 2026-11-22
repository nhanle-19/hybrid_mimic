import gymnasium as gym
from . import flat_env_cfg

# Archived FORCE policies retain their original observation/action interface.
# The canonical tracking tasks are registered by t1_wbc_force2.
for task_id, cfg in [('Tracking-WBC-FORCE-LEGACY-T1-v0', flat_env_cfg.T1WBCForceEnvCfg),
                     ('Tracking-WBC-FORCE-LEGACY-T1-Eval-v0', flat_env_cfg.T1WBCForceEnvEvalCfg),
                     ('Standing-WBC-FORCE-T1-v0', flat_env_cfg.T1WBCForceStandingEnvCfg),
                     ('Toy-Standing-WBC-FORCE-T1-v0', flat_env_cfg.T1WBCForceToyStandingEnvCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:T1WBCForcePPORunnerCfg'})
