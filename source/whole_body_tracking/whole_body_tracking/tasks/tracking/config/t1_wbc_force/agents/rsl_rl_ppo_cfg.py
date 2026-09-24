from isaaclab.utils import configclass
from ...t1_wbc_acc.agents.rsl_rl_ppo_cfg import T1WBCACCPPORunnerCfg


@configclass
class T1WBCForcePPORunnerCfg(T1WBCACCPPORunnerCfg):
    experiment_name = 't1_wbc_force'
