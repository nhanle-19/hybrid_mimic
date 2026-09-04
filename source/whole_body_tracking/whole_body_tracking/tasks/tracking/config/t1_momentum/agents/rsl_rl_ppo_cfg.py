"""RSL-RL configuration for the T1 momentum whole-body controller environments."""

from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.t1_hybrid.agents.rsl_rl_ppo_cfg import T1FlatPPORunnerCfg


@configclass
class T1MomentumPPORunnerCfg(T1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "t1_momentum"
