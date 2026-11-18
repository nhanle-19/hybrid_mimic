"""Mild FORCE2 training randomization; the controller retains its nominal model."""
import math
import numpy as np
import torch

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg
from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg
from whole_body_tracking.utils.wbc_force_randomization import (
    perturb_inertials, require_match, state_observation_noise, uniform,
)


def randomize_physics(env, env_ids, mass_scale_range, inertia_scale_range, trunk_com_range,
                      joint_friction_range, armature_range):
    """Startup-only sampling, with a nominal-asset check before any changes."""
    if env_ids is not None:
        raise ValueError('FORCE2 inertial randomization must run at startup for all environments')
    action = env.action_manager.get_term('wbc_force')
    robot, model = action._asset, action.dynamics
    view = robot.root_physx_view
    mass, inertia, com = (view.get_masses().cpu().clone(), view.get_inertias().cpu().clone(),
                          view.get_coms().cpu().clone())
    order = [model.body_names.index(name) for name in robot.body_names]
    require_match(mass, model.mass.cpu()[order].expand_as(mass), 'nominal masses')
    require_match(inertia, model.inertia.cpu()[order].reshape(1, -1, 9).expand_as(inertia), 'nominal inertias')
    require_match(com[..., :3], model.com_local.cpu()[order].expand_as(com[..., :3]), 'nominal COM')
    mass, inertia, com, mass_scale, inertia_scale = perturb_inertials(
        mass, inertia, com, robot.body_names.index('Trunk'), mass_scale_range, inertia_scale_range, trunk_com_range)
    ids = torch.arange(env.num_envs, device='cpu')
    view.set_masses(mass, ids)
    view.set_inertias(inertia, ids)
    view.set_coms(com, ids)
    shape = robot.data.joint_pos.shape
    friction = uniform(shape, joint_friction_range, device=env.device)
    armature = uniform(shape, armature_range, device=env.device)
    robot.write_joint_friction_coefficient_to_sim(friction)
    robot.write_joint_armature_to_sim(armature)
    action._randomized_physics = dict(mass=mass, inertia=inertia, com=com,
                                      joint_friction=friction.cpu(), armature=armature.cpu())
    model_order = [robot.body_names.index(name) for name in model.body_names]
    action._physical_mass = mass[:, model_order].to(env.device, dtype=torch.float64)
    action._physical_inertia = inertia[:, model_order].reshape(env.num_envs, -1, 3, 3).to(env.device, dtype=torch.float64)
    action.validated = False
    metrics = env._force2_domain_randomization_metrics = {
        'mass_scale_min': float(mass_scale.min()), 'mass_scale_max': float(mass_scale.max()),
        'inertia_scale_min': float(inertia_scale.min()), 'inertia_scale_max': float(inertia_scale.max()),
        'joint_friction_max': float(friction.max()), 'armature_max': float(armature.max()),
        'velocity_pushes_total': 0., 'wrench_pulses_total': 0.,
    }
    print(f'[INFO] FORCE2 mild domain randomization active; nominal QP model retained. {metrics}', flush=True)


def validate_randomized_physics(action, states):
    """Allow exactly the registered perturbations; retain geometry and zero-PD checks."""
    sample = getattr(action, '_randomized_physics', None)
    if sample is None:
        raise RuntimeError('FORCE2 randomization enabled but startup sampling did not run')
    robot, view = action._asset, action._asset.root_physx_view
    for name, actual in (('mass', view.get_masses()), ('inertia', view.get_inertias()),
                        ('com', view.get_coms()), ('joint_friction', view.get_dof_friction_coefficients()),
                        ('armature', view.get_dof_armatures())):
        require_match(actual, sample[name], name)
    for name in ('joint_stiffness', 'joint_damping'):
        require_match(getattr(robot.data, name), torch.zeros_like(getattr(robot.data, name)), name, atol=0.)
    for name in action.dynamics.body_names:
        index = robot.body_names.index(name)
        actual = robot.data.body_link_pos_w[:, index] - action._env.scene.env_origins
        expected = torch.as_tensor(np.stack([s['frames'][name]['position'] for s in states]))
        require_match(actual, expected, f'{name} kinematic frame', atol=2e-4)
    # Momentum intentionally differs from the nominal model under inertial DR.
    # _states() computes measured momentum using the sampled physical inertias.
    action.validated = True


def push_velocity(env, env_ids, velocity_range):
    mdp.push_by_setting_velocity(env, env_ids, velocity_range)
    env._force2_domain_randomization_metrics['velocity_pushes_total'] += len(env_ids)


def start_wrench_pulse(env, env_ids, force_range, torque_range, duration_s):
    action = env.action_manager.get_term('wbc_force')
    shape = (len(env_ids), 1, 3)
    action._wrench_pulses.start(env_ids,
        uniform(shape, force_range, device=env.device), uniform(shape, torque_range, device=env.device),
        max(1, math.ceil(duration_s/env.physics_dt)))
    env._force2_domain_randomization_metrics['wrench_pulses_total'] += len(env_ids)


@configclass
class Force2StateNoiseCfg(NoiseCfg):
    func = state_observation_noise
    joint_position: float = .005  # rad
    joint_velocity: float = .05  # rad/s
    root_position: float = .002  # m
    orientation: float = .005  # rad, maximum axis-angle magnitude
    linear_velocity: float = .02  # m/s
    angular_velocity: float = .02  # rad/s


@configclass
class Force2RandomizationEventsCfg:
    physics_material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='startup', params={
        'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
        'static_friction_range': (.55, .65), 'dynamic_friction_range': (.55, .65),
        'restitution_range': (0., .02), 'num_buckets': 64, 'make_consistent': True,
    })
    model_parameters = EventTermCfg(func=randomize_physics, mode='startup', params={
        'mass_scale_range': (.97, 1.03), 'inertia_scale_range': (.95, 1.05), 'trunk_com_range': .003,
        'joint_friction_range': (0., .005), 'armature_range': (0., .0005),
    })
    push_robot = EventTermCfg(func=push_velocity, mode='interval', interval_range_s=(4., 8.), params={
        'velocity_range': {'x': (-.1, .1), 'y': (-.1, .1), 'yaw': (-.1, .1)},
    })
    external_wrench = EventTermCfg(func=start_wrench_pulse, mode='interval', interval_range_s=(6., 10.), params={
        'force_range': (-5., 5.), 'torque_range': (-.5, .5), 'duration_s': .1,
    })


def configure_randomization(cfg):
    cfg.scene.terrain.usd_path = f'{ASSET_DIR}/booster/t1/wbc_force_dr_ground.usda'
    cfg.events = Force2RandomizationEventsCfg()
    cfg.actions.wbc_force.domain_randomization = True
    cfg.commands.motion.pose_range = {
        'x': (-.005, .005), 'y': (-.005, .005), 'z': (0., .003),
        'roll': (-.01, .01), 'pitch': (-.01, .01), 'yaw': (-.02, .02),
    }
    cfg.commands.motion.velocity_range = {
        'x': (-.03, .03), 'y': (-.03, .03), 'z': (-.01, .01),
        'roll': (-.03, .03), 'pitch': (-.03, .03), 'yaw': (-.03, .03),
    }
    cfg.commands.motion.joint_position_range = (-.005, .005)
    cfg.observations.policy.actual_state.noise = Force2StateNoiseCfg()
    cfg.observations.policy.enable_corruption = True
    for name, actuator in cfg.scene.robot.actuators.items():
        values = dict(vars(actuator))
        values.pop('class_type')
        cfg.scene.robot.actuators[name] = DelayedImplicitActuatorCfg(**values, min_delay=0, max_delay=2)
