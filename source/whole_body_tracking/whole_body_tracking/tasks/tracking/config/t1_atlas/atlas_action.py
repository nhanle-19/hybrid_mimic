"""Isaac action term: analytical Atlas QP at every physics step, direct torque."""
import numpy as np
import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .controller_cfg import T1AtlasControllerCfg


class AtlasAction(ActionTerm):
    def __new__(cls, cfg, env):
        if cls is AtlasAction:
            if cfg.backend in ('batched', 'osqp'):
                from .atlas_gpu_action import AtlasGPUAction
                return object.__new__(AtlasGPUAction)
            if cfg.backend != 'osqp':
                raise ValueError('Atlas backend must be batched or osqp')
        return object.__new__(cls)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Optional dependencies are loaded only when this task is instantiated.
        from whole_body_tracking.utils.atlas_model import AtlasModel, FEET
        from whole_body_tracking.utils.atlas_qp import AtlasQP
        from whole_body_tracking.utils.atlas_contact import GroundContactEstimator
        if cfg.contact_source not in ('reference', 'ground_force'):
            raise ValueError('Atlas contact_source must be reference or ground_force')
        if cfg.contact_source == 'ground_force' and cfg.contact_schedule_file:
            raise ValueError('A reference contact schedule cannot drive ground_force mode')
        self.contact_estimator = GroundContactEstimator(env.num_envs, cfg.contact_on_force, cfg.contact_off_force)
        self.dynamics = AtlasModel()
        self.feet = FEET
        self.pin_to_sim = [self._asset.joint_names.index(n) for n in self.dynamics.joint_names]
        if set(self._asset.joint_names) != set(self.dynamics.joint_names):
            raise ValueError('Atlas model and simulation joint names differ')
        self._raw = torch.zeros((env.num_envs, 2), device=env.device)
        self._torques = torch.zeros((env.num_envs, 23), device=env.device)
        limits = self._asset.data.joint_effort_limits[:, self.pin_to_sim].cpu().numpy()
        self.solvers = [AtlasQP(lim, momentum_weights=cfg.controller.momentum_weights,
                               force_weight=cfg.controller.force_weight, acceleration_weight=cfg.controller.acceleration_weight,
                               tolerance=cfg.residual_tolerance) for lim in limits]
        self.references = None
        self.planned_contacts = None
        self.contact_override = None
        self.external = [[] for _ in range(env.num_envs)]
        self.pending = [None]*env.num_envs
        self.records = []
        self.validated = False
        self._clock = 0

    @property
    def action_dim(self):
        return 14

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._torques

    def process_actions(self, actions):
        if actions.shape != self._raw.shape or not torch.isfinite(actions).all():
            raise ValueError('Atlas expects 14 finite contact-policy actions per environment')
        self._raw[:] = actions

    def set_active_contacts(self, mask):
        """Explicit override, boolean (num_envs,2), or None for the configured source."""
        if mask is None:
            self.contact_override = None
            return
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (self._env.num_envs, 2):
            raise ValueError('Contact override must be boolean (num_envs,2)')
        self.contact_override = mask.copy()

    def set_external_wrenches(self, env_id, wrenches):
        """Known applied physical wrenches for the QP; caller must apply them to physics too."""
        self.external[env_id] = list(wrenches)

    def _reference_states(self):
        from whole_body_tracking.utils.atlas_control import reference_contact_schedule
        from scipy.spatial.transform import Rotation
        command = self._env.command_manager.get_term('motion')
        motion = command.motion
        root_id = self._asset.body_names.index('Trunk')
        jp = motion.joint_pos.cpu().numpy()[:, self.pin_to_sim]
        jv = motion.joint_vel.cpu().numpy()[:, self.pin_to_sim]
        pos = motion._body_pos_w[:, root_id].cpu().numpy()
        quat = motion._body_quat_w[:, root_id].cpu().numpy()[:, [1, 2, 3, 0]]
        lv = motion._body_lin_vel_w[:, root_id].cpu().numpy()
        av = motion._body_ang_vel_w[:, root_id].cpu().numpy()
        self.references = []
        for t in range(len(jp)):
            rotation = Rotation.from_quat(quat[t]).as_matrix()
            q = np.r_[pos[t], quat[t], jp[t]]
            v = np.r_[rotation.T@lv[t], rotation.T@av[t], jv[t]]
            self.references.append(self.dynamics.state(q, v))
        if self.cfg.contact_schedule_file:
            schedule = np.load(self.cfg.contact_schedule_file, allow_pickle=False)
            if schedule.dtype != bool or schedule.shape != (len(jp), 2):
                raise ValueError('Contact schedule must be a boolean NPY (motion_frames,2)')
            self.planned_contacts = schedule
        else:
            self.planned_contacts = reference_contact_schedule(self.references, self.cfg.contact_height_tolerance)
        if self.cfg.standing_only and self.cfg.contact_source == 'reference':
            inferred = reference_contact_schedule(self.references, self.cfg.contact_height_tolerance)
            if not inferred.all() or not self.planned_contacts.all():
                raise ValueError('Standing task requires a first reference pose with both feet on the ground '
                                 'and a double-support contact schedule')
            self.planned_contacts[:] = True

    def _ground_forces(self):
        from whole_body_tracking.utils.atlas_contact import read_ground_forces
        return read_ground_forces(self._env.scene.sensors, self.feet, self._env.num_envs)

    def _contact_masks(self, times, ground_forces):
        estimated = self.contact_estimator.update(ground_forces[:, :, 2])
        reference = self.planned_contacts[times]
        selected = estimated if self.cfg.contact_source == 'ground_force' else reference
        if self.contact_override is not None:
            selected = self.contact_override
        return selected, estimated, reference

    def _states(self):
        from scipy.spatial.transform import Rotation
        robot = self._asset
        pos = (robot.data.root_link_pos_w-self._env.scene.env_origins).cpu().numpy()
        quat = robot.data.root_link_quat_w.cpu().numpy()[:, [1, 2, 3, 0]]
        vel = robot.data.root_link_vel_w.cpu().numpy()
        jp = robot.data.joint_pos[:, self.pin_to_sim].cpu().numpy()
        jv = robot.data.joint_vel[:, self.pin_to_sim].cpu().numpy()
        masses = robot.root_physx_view.get_masses().cpu().numpy()
        inertia = robot.root_physx_view.get_inertias().cpu().numpy().reshape(self._env.num_envs, -1, 3, 3)
        # PhysX articulation get_inertias returns the full tensor about the CoM
        # in LINK-aligned axes (already rotated out of principal axes).
        body_rotation = Rotation.from_quat(robot.data.body_link_quat_w.cpu().numpy()[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(inertia.shape)
        world_inertia = body_rotation@inertia@body_rotation.transpose(0, 1, 3, 2)
        body_pos = robot.data.body_com_pos_w.cpu().numpy()
        body_vel = robot.data.body_com_vel_w.cpu().numpy()
        com = (masses[..., None]*body_pos).sum(axis=1)/masses.sum(axis=1)[:, None]
        linear = masses[..., None]*body_vel[..., :3]
        angular = (world_inertia@body_vel[..., 3:, None]).squeeze(-1)+np.cross(body_pos-com[:, None], linear)
        measured_momentum = np.concatenate([linear.sum(axis=1), angular.sum(axis=1)], axis=-1)
        states = []
        for i in range(self._env.num_envs):
            r = Rotation.from_quat(quat[i]).as_matrix()
            states.append(self.dynamics.state(np.r_[pos[i], quat[i], jp[i]], np.r_[r.T@vel[i, :3], r.T@vel[i, 3:], jv[i]]))
            states[-1]['measured_momentum'] = measured_momentum[i]
        return states

    def _validate_simulator_model(self, states):
        """Fail before applying torque if asset or randomization differs from the model."""
        robot = self._asset
        from scipy.spatial.transform import Rotation
        masses = robot.root_physx_view.get_masses().cpu().numpy()
        inertias = robot.root_physx_view.get_inertias().cpu().numpy().reshape(self._env.num_envs, -1, 3, 3)
        local_com_poses = robot.root_physx_view.get_coms().cpu().numpy()  # PhysX quaternion XYZW
        for name, body in self.dynamics.description['bodies'].items():
            index = robot.body_names.index(name)
            if not np.allclose(masses[:, index], body['mass'], atol=1e-5):
                raise RuntimeError(f'Atlas simulator mass mismatch for {name}')
            actual = (robot.data.body_link_pos_w[:, index]-self._env.scene.env_origins).cpu().numpy()
            expected = np.stack([s['frames'][name]['position'] for s in states])
            if not np.allclose(actual, expected, atol=2e-4):
                raise RuntimeError(f'Atlas simulator frame mismatch for {name}: max error {np.abs(actual-expected).max()}')
            com = local_com_poses[:, index, :3]
            if not np.allclose(com, body['com'], atol=1e-5):
                raise RuntimeError(f'Atlas simulator CoM mismatch for {name}')
            actual_inertia = inertias[:, index]
            expected_rotation = Rotation.from_quat(body['principal_axes_xyzw']).as_matrix()
            expected_inertia = expected_rotation@np.diag(body['inertia_diagonal'])@expected_rotation.T
            if not np.allclose(actual_inertia, expected_inertia, atol=1e-5):
                raise RuntimeError(f'Atlas simulator inertia mismatch for {name}')
        for state in states:
            if not np.allclose(state['measured_momentum'], state['momentum'], atol=1e-3):
                raise RuntimeError('Atlas simulator/analytical centroidal momentum mismatch')
        self.validated = True

    def _finish_pending(self, states):
        if not self.cfg.record_diagnostics:
            return
        ground_forces = self._ground_forces()
        for i, state in enumerate(states):
            record = self.pending[i]
            if record is None:
                continue
            record['actual_rate'] = (state['measured_momentum']-record.pop('_momentum'))/self._env.physics_dt
            previous_velocities = record.pop('_contact_velocities')
            residuals = [(state['frames'][foot]['velocity']-velocity)/self._env.physics_dt
                         for foot, velocity in previous_velocities.items()]
            record['actual_contact_acceleration_residual'] = float(np.max(np.abs(np.concatenate([r[2:] for r in residuals])), initial=0)) if residuals else 0.
            record['actual_contact_tangential_acceleration_residual'] = float(np.max(np.abs(np.concatenate([r[:2] for r in residuals])), initial=0)) if residuals else 0.
            record['applied_torque'] = self._asset.data.applied_torque[i, self.pin_to_sim].cpu().numpy().copy()
            sensor = self._env.scene.sensors['contact_forces']
            ids = [sensor.body_names.index(foot) for foot in self.feet]
            record['sensor_normal_force_w'] = sensor.data.net_forces_w[i, ids].cpu().numpy().copy()
            record['ground_normal_force_w'] = ground_forces[i].copy()
            self.records.append(record)
            self.pending[i] = None

    def apply_actions(self):
        from whole_body_tracking.utils.atlas_qp import Contact
        from whole_body_tracking.utils.atlas_control import build_reference_tasks, swing_tasks, force_diagnostics, contact_weights
        if self.references is None:
            self._reference_states()
        states = self._states()
        if not self.validated:
            self._validate_simulator_model(states)
        self._finish_pending(states)
        actions = self._raw.cpu().numpy()
        times = self._env.command_manager.get_term('motion').time_steps.cpu().numpy()
        ground_forces = self._ground_forces()
        masks, estimated, reference_contacts = self._contact_masks(times, ground_forces)
        for i, state in enumerate(states):
            ref = self.references[int(times[i])]
            mask = masks[i]
            active = [foot for foot, enabled in zip(self.feet, mask) if enabled]
            weights = contact_weights(actions[i], self.cfg.controller)
            rate, tasks = build_reference_tasks(state, ref, actions[i], self.cfg.controller)
            tasks += swing_tasks(state, ref, active, self.cfg.controller)
            result = self.solvers[i].solve(state, rate, [Contact(foot, friction=self.cfg.friction) for foot in active],
                                            tasks, self.external[i],
                                            contact_weights={foot: weights[self.feet.index(foot)] for foot in active})
            self._torques[i, self.pin_to_sim] = torch.as_tensor(result['torque'], device=self._env.device, dtype=self._torques.dtype)
            if self.cfg.record_diagnostics:
                record = force_diagnostics(result, state, self.solvers[i].torque_limits)
                record.update(env_id=i, time=self._clock*self._env.physics_dt,
                              foot_position_w=np.stack([state['frames'][foot]['position'] for foot in self.feet]),
                              reference_foot_position_w=np.stack([ref['frames'][foot]['position'] for foot in self.feet]),
                              foot_velocity_w=np.stack([state['frames'][foot]['velocity'] for foot in self.feet]),
                              reference_foot_velocity_w=np.stack([ref['frames'][foot]['velocity'] for foot in self.feet]),
                              com_position=state['com'].copy(), reference_com_position=ref['com'].copy(),
                              joint_position=state['q'][7:].copy(), reference_joint_position=ref['q'][7:].copy(),
                              contact_input_force_w=ground_forces[i].copy(),
                              estimated_contact=estimated[i].copy(),
                              reference_contact=reference_contacts[i].copy(),
                              contact_override_active=self.contact_override is not None,
                              contact_weights=weights.copy(), contact_weight_logits=actions[i].copy(),
                              contact_tangential_acceleration=np.stack([result['contact_acceleration'].get(foot, np.zeros(6))[:2] for foot in self.feet]),
                              desired_rate=rate, predicted_rate=result['rate'],
                              contact_acceleration_residual=result['metrics']['stance'],
                              inverse_dynamics_residual=result['inverse_residual'],
                              commanded_torque=result['torque'],
                              reference_frame=int(times[i]), variable_count=result['variable_count'],
                              momentum_identity_residual=result['metrics']['momentum_identity'],
                              simulator_momentum_identity_residual=np.max(np.abs(state['measured_momentum']-state['Ag']@state['v'])),
                              _momentum=state['measured_momentum'].copy(),
                              _contact_velocities={foot: state['frames'][foot]['velocity'].copy() for foot in active})
                self.pending[i] = record
        # No position target offset and no post-solve clamp. Actuator gains are zero.
        self._asset.set_joint_effort_target(self._torques)
        self._clock += 1

    def reset(self, env_ids=None):
        ids = list(range(self._env.num_envs)) if env_ids is None else list(env_ids)
        self.contact_estimator.reset([int(i) for i in ids])
        for i in ids:
            i = int(i)
            self.pending[i] = None  # Never differentiate momentum across a reset.
            self.external[i] = []
        self._raw[env_ids if env_ids is not None else slice(None)] = 0

    def save_diagnostics(self, path, finalize=True, completed=True, failure_reason=''):
        from pathlib import Path
        import json
        if finalize:
            self._finish_pending(self._states())
        if not self.records:
            raise RuntimeError('No Atlas diagnostic samples recorded')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: np.asarray([r[key] for r in self.records]) for key in self.records[0]},
                            joint_names=np.asarray(self.dynamics.joint_names), foot_names=np.asarray(self.feet),
                            physics_dt=self._env.physics_dt, planned_reference_contacts=(
                                self.planned_contacts.detach().cpu().numpy()
                                if isinstance(self.planned_contacts, torch.Tensor) else self.planned_contacts),
                            contact_source=np.asarray(self.cfg.contact_source),
                            contact_on_force_N=self.cfg.contact_on_force, contact_off_force_N=self.cfg.contact_off_force,
                            controller_config_json=np.asarray(json.dumps(self.cfg.controller.to_dict())),
                            evaluation_completed=completed, failure_reason=np.asarray(failure_reason))


@configclass
class AtlasActionCfg(ActionTermCfg):
    class_type: type = AtlasAction
    asset_name: str = 'robot'
    backend: str = 'batched'
    batched_max_iterations: int = 100
    batched_tolerance: float = 1e-9
    controller: T1AtlasControllerCfg = T1AtlasControllerCfg()
    friction: float = .6
    contact_height_tolerance: float = .025
    contact_schedule_file: str | None = None
    contact_source: str = 'reference'
    contact_on_force: float = 10.
    contact_off_force: float = 5.
    residual_tolerance: float = 2e-5
    record_diagnostics: bool = False
    standing_only: bool = False
