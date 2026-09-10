from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.t1.agents.rsl_rl_ppo_cfg import T1FlatPPORunnerCfg


@configclass
class T1AtlasPPORunnerCfg(T1FlatPPORunnerCfg):
    experiment_name = 't1_atlas'
