"""Separate direct-torque WBCACC task; with reference-conditioned contact actions."""
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, SceneEntityCfg
from .observations import WBCACCObservationsCfg
from .rewards import alive_reward, excessive_slip, mapped_action_rate, qp_failed
from .contact_sensor import WBCACCContactSensor
from whole_body_tracking.tasks.tracking.config.t1.flat_env_cfg import T1FlatEnvCfg, T1FlatEnvEvalCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import EventEvalCfg
from .wbc_acc_action import WBCACCActionCfg
from whole_body_tracking.assets import ASSET_DIR
from .motion import WBCACCEvaluationMotion, WBCACCStandingMotion, foot_collision


@configclass
class WBCACCActionsCfg:
    wbc_acc: WBCACCActionCfg = WBCACCActionCfg()


def configure_wbc_acc(cfg):
    # One QP solve per policy action; hold torques for ten physics substeps.
    cfg.sim.dt = .002
    cfg.decimation = 10
    cfg.actions = WBCACCActionsCfg()
    cfg.observations = WBCACCObservationsCfg()
    cfg.scene.num_envs = 1024
    cfg.sim.device = 'cuda:0'
    cfg.scene.terrain.terrain_type = 'usd'
    cfg.scene.terrain.usd_path = f'{ASSET_DIR}/booster/t1/wbc_acc_ground.usda'
    cfg.scene.terrain.visual_material = None
    # Filter each foot to the actual static floor collider, excluding self contacts.
    for side in ('left', 'right'):
        setattr(cfg.scene, f'{side}_foot_ground_contact', ContactSensorCfg(
            prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_foot_link',
            filter_prim_paths_expr=[f'{cfg.scene.terrain.prim_path}/terrain/Floor'],
            update_period=0., history_length=0, debug_vis=False, class_type=WBCACCContactSensor))
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
    # RewardManager multiplies by dt: +1 per simulated second survived.
    cfg.rewards.alive = RewardTermCfg(func=alive_reward, weight=1.)
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed)
    cfg.rewards.excessive_slip = RewardTermCfg(func=excessive_slip, weight=-1.)
    cfg.rewards.left_foot_right_foot_collision.func = foot_collision


@configclass
class T1WBCACCEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_acc(self)


@configclass
class T1WBCACCEnvEvalCfg(T1FlatEnvEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_acc(self)
        self.actions.wbc_acc.record_diagnostics = True
        self.commands.motion.class_type = WBCACCEvaluationMotion


@configclass
class T1WBCACCStandingEnvCfg(T1WBCACCEnvEvalCfg):
    """Deterministic double-support standing test using the first reference pose."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = WBCACCStandingMotion
        self.commands.motion.resampling_time_range = (1.e9, 1.e9)
        self.actions.wbc_acc.standing_only = True
        self.observations.policy.enable_corruption = False
