import gymnasium as gym
from . import flat_env_cfg

# FORCE is the default training task: full FORCE2 domain randomization and
# learned posture gains. Evaluation uses the matching nominal environment.
for task_id, cfg in [('Tracking-WBC-FORCE-T1-v0', flat_env_cfg.T1WBCForceRandomizedEnvCfg),
                     ('Tracking-WBC-FORCE-T1-Eval-v0', flat_env_cfg.T1WBCForceEnvEvalCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:ForcePPORunnerCfg'})

# Keep explicit FORCE2 task names for existing runs and nominal comparisons.
for task_id, cfg in [('Tracking-WBC-FORCE2-T1-v0', flat_env_cfg.T1WBCForceEnvCfg),
                     ('Tracking-WBC-FORCE2-T1-Eval-v0', flat_env_cfg.T1WBCForceEnvEvalCfg),
                     ('Standing-WBC-FORCE2-T1-v0', flat_env_cfg.T1WBCForceStandingEnvCfg),
                     ('Toy-Standing-WBC-FORCE2-T1-v0', flat_env_cfg.T1WBCForceToyStandingEnvCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:T1WBCForcePPORunnerCfg'})

for task_id, cfg in [('Tracking-WBC-FORCE2-DR-T1-v0', flat_env_cfg.T1WBCForceRandomizedEnvCfg),
                     ('Tracking-WBC-FORCE2-DR-T1-Eval-v0', flat_env_cfg.T1WBCForceEnvEvalCfg)]:
    gym.register(id=task_id, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True,
                 kwargs={'env_cfg_entry_point': cfg,
                         'rsl_rl_cfg_entry_point': f'{__name__}.agents.rsl_rl_ppo_cfg:T1WBCForceDRPPORunnerCfg'})
