from isaaclab.utils import configclass
import rsl_rl.modules
from whole_body_tracking.utils.bounded_actor_critic import BoundedMeanActorCritic
from ...t1_wbc_acc.agents.rsl_rl_ppo_cfg import T1WBCACCPPORunnerCfg


# This RSL-RL version resolves policy names in its runner's rsl_rl namespace.
# Loading the task config registers the class for training and every play script.
rsl_rl.modules.BoundedMeanActorCritic = BoundedMeanActorCritic


@configclass
class T1WBCForcePPORunnerCfg(T1WBCACCPPORunnerCfg):
    experiment_name = 't1_wbc_force'
    policy = T1WBCACCPPORunnerCfg().policy.replace(class_name='rsl_rl.modules.BoundedMeanActorCritic')
