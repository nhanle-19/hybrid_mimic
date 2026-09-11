"""Atlas dynamics with the HybridMimic policy, rewards and PD/feedforward loop."""
from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.t1_hybrid.flat_env_cfg import T1HybridEnvCfg, T1HybridEnvEvalCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import EventEvalCfg
from .controller_cfg import T1AtlasControllerCfg
from whole_body_tracking.assets import ASSET_DIR
from .motion import AtlasEvaluationMotion, foot_collision


def configure_atlas(cfg):
    cfg.scene.num_envs = 128 if cfg.hybrid_controller.backend == 'batched' else 2
    cfg.scene.terrain.terrain_type = 'usd'
    cfg.scene.terrain.usd_path = f'{ASSET_DIR}/booster/t1/atlas_ground.usda'
    cfg.scene.terrain.visual_material = None
    cfg.events = EventEvalCfg()
    cfg.events.add_joint_default_pos = None
    cfg.events.base_com = None
    cfg.events.physics_material.params.update(static_friction_range=(.6, .6),
        dynamic_friction_range=(.6, .6), restitution_range=(0., 0.))
    # Fixed model/materials; retain the hybrid robot's PD gains and armature.
    cfg.scene.contact_forces.debug_vis = False
    cfg.commands.motion.debug_vis = False
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0., 0.)
    cfg.rewards.left_foot_right_foot_collision.func = foot_collision


@configclass
class T1AtlasEnvCfg(T1HybridEnvCfg):
    hybrid_controller: T1AtlasControllerCfg = T1AtlasControllerCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_atlas(self)


@configclass
class T1AtlasEnvEvalCfg(T1HybridEnvEvalCfg):
    hybrid_controller: T1AtlasControllerCfg = T1AtlasControllerCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_atlas(self)
        self.hybrid_controller.record_diagnostics = True
        self.commands.motion.class_type = AtlasEvaluationMotion
