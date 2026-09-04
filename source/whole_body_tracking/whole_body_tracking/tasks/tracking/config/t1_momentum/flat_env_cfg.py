from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.t1_hybrid.flat_env_cfg import T1HybridEnvCfg, T1HybridEnvEvalCfg

from .controller_cfg import T1MomentumControllerCfg


@configclass
class T1MomentumEnvCfg(T1HybridEnvCfg):
    hybrid_controller: T1MomentumControllerCfg = T1MomentumControllerCfg()


@configclass
class T1MomentumEnvEvalCfg(T1HybridEnvEvalCfg):
    hybrid_controller: T1MomentumControllerCfg = T1MomentumControllerCfg()
