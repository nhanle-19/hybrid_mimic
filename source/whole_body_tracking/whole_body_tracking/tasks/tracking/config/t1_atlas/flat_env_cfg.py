"""Separate direct-torque Atlas task; the floating_model baseline is untouched."""
from isaaclab.utils import configclass
from whole_body_tracking.tasks.tracking.config.t1.flat_env_cfg import T1FlatEnvCfg, T1FlatEnvEvalCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import EventEvalCfg
from .atlas_action import AtlasActionCfg
from whole_body_tracking.assets import ASSET_DIR
from .motion import AtlasEvaluationMotion, foot_collision


@configclass
class AtlasActionsCfg:
    atlas: AtlasActionCfg = AtlasActionCfg()


def configure_atlas(cfg):
    cfg.actions = AtlasActionsCfg()
    cfg.scene.num_envs = 2  # CPU OSQP is a correctness baseline, not a GPU batched solver.
    cfg.scene.terrain.terrain_type = 'usd'
    cfg.scene.terrain.usd_path = f'{ASSET_DIR}/booster/t1/atlas_ground.usda'
    cfg.scene.terrain.visual_material = None
    cfg.events = EventEvalCfg()
    cfg.events.add_joint_default_pos = None
    cfg.events.base_com = None
    cfg.events.physics_material.params.update(static_friction_range=(.6, .6),
        dynamic_friction_range=(.6, .6), restitution_range=(0., 0.))
    # Fixed inertial model, no gain/material/random CoM perturbations or unknown pushes.
    cfg.scene.robot = cfg.scene.robot.copy()
    for actuator in cfg.scene.robot.actuators.values():
        actuator.stiffness = 0.
        actuator.damping = 0.
        actuator.armature = 0.
        actuator.friction = 0.
        actuator.dynamic_friction = 0.
        actuator.viscous_friction = 0.
    cfg.scene.contact_forces.debug_vis = False
    cfg.commands.motion.debug_vis = False
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0., 0.)
    cfg.rewards.left_foot_right_foot_collision.func = foot_collision


@configclass
class T1AtlasEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_atlas(self)


@configclass
class T1AtlasEnvEvalCfg(T1FlatEnvEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_atlas(self)
        self.actions.atlas.record_diagnostics = True
        self.commands.motion.class_type = AtlasEvaluationMotion
