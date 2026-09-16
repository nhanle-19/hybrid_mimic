"""Separate direct-torque Atlas task; the floating_model baseline is untouched."""
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, SceneEntityCfg
from .observations import AtlasObservationsCfg
from .rewards import excessive_slip, mapped_action_rate, qp_failed, termination_penalty
from .contact_sensor import AtlasContactSensor
from whole_body_tracking.tasks.tracking.config.t1.flat_env_cfg import T1FlatEnvCfg, T1FlatEnvEvalCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import EventEvalCfg
from .atlas_action import AtlasActionCfg
from whole_body_tracking.assets import ASSET_DIR
from .motion import AtlasEvaluationMotion, AtlasStandingMotion, foot_collision


@configclass
class AtlasActionsCfg:
    atlas: AtlasActionCfg = AtlasActionCfg()


def configure_atlas(cfg):
    cfg.actions = AtlasActionsCfg()
    cfg.observations = AtlasObservationsCfg()
    cfg.scene.num_envs = 1024
    cfg.sim.device = 'cuda:0'
    cfg.scene.terrain.terrain_type = 'usd'
    cfg.scene.terrain.usd_path = f'{ASSET_DIR}/booster/t1/atlas_ground.usda'
    cfg.scene.terrain.visual_material = None
    # Filter each foot to the actual static floor collider, excluding self contacts.
    for side in ('left', 'right'):
        setattr(cfg.scene, f'{side}_foot_ground_contact', ContactSensorCfg(
            prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_foot_link',
            filter_prim_paths_expr=[f'{cfg.scene.terrain.prim_path}/terrain/Floor'],
            update_period=0., history_length=0, debug_vis=False, class_type=AtlasContactSensor))
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
    # Body poses are anchor-aligned: retain global trunk rewards intentionally
    # to constrain world position/heading, plus body pose/velocity and joint limits.
    cfg.rewards.action_rate_l2 = RewardTermCfg(func=mapped_action_rate, weight=-.1)
    # Only torso/head collisions are forbidden; limbs may provide support.
    cfg.rewards.undesired_contacts.params['sensor_cfg'] = SceneEntityCfg(
        'contact_forces', body_names=['Trunk', 'H1', 'H2'])
    cfg.rewards.termination = RewardTermCfg(func=termination_penalty, weight=-10.)
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed)
    cfg.rewards.excessive_slip = RewardTermCfg(func=excessive_slip, weight=-1.)
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


@configclass
class T1AtlasStandingEnvCfg(T1AtlasEnvEvalCfg):
    """Deterministic double-support standing test using the first reference pose."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = AtlasStandingMotion
        self.commands.motion.resampling_time_range = (1.e9, 1.e9)
        self.actions.atlas.standing_only = True
        self.observations.policy.enable_corruption = False
