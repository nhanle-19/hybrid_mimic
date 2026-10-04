from isaaclab.utils import configclass
from ...t1_wbc_acc.agents.rsl_rl_ppo_cfg import T1WBCACCPPORunnerCfg


@configclass
class T1WBCAuthorityPPORunnerCfg(T1WBCACCPPORunnerCfg):
    experiment_name = 't1_wbc_authority'
    # Gaussian entropy in unconstrained logit space can grow while sigmoid
    # authorities saturate. Keep stochastic PPO exploration, but do not reward
    # that unbounded proxy for diversity in the physical action space.
    algorithm = T1WBCACCPPORunnerCfg().algorithm.replace(entropy_coef=0.0)
