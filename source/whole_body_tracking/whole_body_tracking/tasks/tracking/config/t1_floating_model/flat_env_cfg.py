from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.t1_hybrid.flat_env_cfg import T1HybridEnvCfg, T1HybridEnvEvalCfg

from .controller_cfg import T1FloatingModelControllerCfg


@configclass
class T1FloatingModelEnvCfg(T1HybridEnvCfg):
    hybrid_controller: T1FloatingModelControllerCfg = T1FloatingModelControllerCfg()


@configclass
class T1FloatingModelEnvEvalCfg(T1HybridEnvEvalCfg):
    hybrid_controller: T1FloatingModelControllerCfg = T1FloatingModelControllerCfg()
