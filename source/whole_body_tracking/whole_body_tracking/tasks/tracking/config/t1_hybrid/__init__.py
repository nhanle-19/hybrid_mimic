import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Hybrid-T1-v0",
    entry_point="whole_body_tracking.tasks.tracking.config.t1_hybrid.hybrid_env:HybridEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T1HybridEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Hybrid-T1-Eval-v0",
    entry_point="whole_body_tracking.tasks.tracking.config.t1_hybrid.hybrid_env:HybridEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.T1HybridEnvEvalCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1FlatPPORunnerCfg",
    },
)
