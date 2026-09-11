"""Atlas analytical QP behind the existing HybridMimic 57-action interface."""
import numpy as np
import torch

from .floating_model import FloatingModelController
from .hybrid import ctrl2components, highlvlPD
from .atlas_model import AtlasModel, FEET
from .atlas_qp import AtlasQP, Contact


class AtlasController(FloatingModelController):
    def __init__(self, articulation, cfg, env):
        super().__init__(articulation, cfg)
        self.cfg, self._asset, self._env = cfg, articulation, env
        self.dynamics = AtlasModel()
        self.feet = FEET
        self.pin_to_sim = [articulation.joint_names.index(n) for n in self.dynamics.joint_names]
        if set(articulation.joint_names) != set(self.dynamics.joint_names):
            raise ValueError('Atlas model and simulation joint names differ')
        self.solvers = [AtlasQP(limits, momentum_weights=cfg.momentum_weights,
                               force_weight=cfg.force_weight, acceleration_weight=cfg.acceleration_weight,
                               tolerance=cfg.residual_tolerance, failure_directory=cfg.failure_directory,
                               enforce_stance=cfg.enforce_stance)
                        for limits in self.torque_limits[:, self.pin_to_sim].cpu().numpy()]
        self.references = self.planned_contacts = self.contact_override = None
        self.external = [[] for _ in range(env.num_envs)]
        self.pending = [None]*env.num_envs
        self.records = []
        self.validated = False
        self._clock = 0

    def set_active_contacts(self, mask):
        """Explicit environment/gait override, boolean (num_envs,2), or None for reference."""
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
        for i, state in enumerate(states):
            record = self.pending[i]
            if record is None:
                continue
            record['actual_rate'] = (state['measured_momentum']-record.pop('_momentum'))/self._env.physics_dt
            previous_velocities = record.pop('_contact_velocities')
            residuals = [(state['frames'][foot]['velocity'][:len(velocity)]-velocity)/self._env.physics_dt
                         for foot, velocity in previous_velocities.items()]
            record['actual_contact_acceleration_residual'] = float(np.max(np.abs(np.concatenate(residuals)), initial=0)) if residuals else 0.
            record['applied_torque'] = self._asset.data.applied_torque[i, self.pin_to_sim].cpu().numpy().copy()
            sensor = self._env.scene.sensors['contact_forces']
            ids = [sensor.body_names.index(foot) for foot in self.feet]
            record['sensor_normal_force_w'] = sensor.data.net_forces_w[i, ids].cpu().numpy().copy()
            self.records.append(record)
            self.pending[i] = None

    def step(self, com_pos, com_vel, jacs, body_pos, base_quat, base_angvel, action, nle, lcc_rand):
        from .atlas_control import force_diagnostics, hybrid_balance_tasks
        if action.shape != (self._env.num_envs, self.action_dim) or not torch.isfinite(action).all():
            raise ValueError(f'Atlas expects {self.action_dim} finite HybridMimic actions per environment')
        if self.references is None:
            self._reference_states()  # Reference is used only to plan foot support.
        states = self._states()
        if not self.validated:
            self._validate_simulator_model(states)
        self._finish_pending(states)
        components = ctrl2components(action, self.joint_count, self.end_effector_count,
            self.torque_limits, self.torque_limits_cost, self.desired_linear_velocity_scale,
            self.desired_angular_velocity_scale, self.torque_action_scale,
            self.linear_velocity_gain, self.angular_velocity_gain)
        com_acc, ang_acc, global_vel, global_angvel = highlvlPD(base_quat, base_angvel,
            self.linear_velocity_gain, self.angular_velocity_gain,
            components['des_com_vel'], components['des_com_angvel'], com_vel)
        # Identical target convention to floating_model: simulator joint order,
        # configured default pose, no reference-pose residual addition.
        joint_action = self._env.action_manager.get_term('joint_pos')
        targets = components['des_pos']*joint_action._scale + joint_action._offset
        robot_data = self._asset.data
        pd_torque = robot_data.joint_stiffness*(targets-robot_data.joint_pos) \
            + robot_data.joint_damping*(robot_data.joint_vel_target-robot_data.joint_vel)
        torque = torch.zeros_like(self.torque_limits)
        total_torque = torch.zeros_like(self.torque_limits)
        force = torch.zeros((self._env.num_envs, self.end_effector_count, 6), device=action.device)
        times = self._env.command_manager.get_term('motion').time_steps.cpu().numpy()
        sensor = self._env.scene.sensors['contact_forces']
        measured_contacts = sensor.data.net_forces_w.cpu().numpy()
        weights = torch.exp(-components['w'].clamp(-self.force_logit_clip, self.force_logit_clip)).cpu().numpy()
        for i, state in enumerate(states):
            # Account for the hybrid robot's rotor inertia in joint dynamics.
            armature = self._asset.data.joint_armature[i, self.pin_to_sim].cpu().numpy()
            state['M'][6:, 6:] += np.diag(armature)
            mask = self.planned_contacts[int(times[i])] if self.contact_override is None else self.contact_override[i]
            contacts = [Contact(foot, friction=self.cfg.friction)
                        for foot, enabled in zip(self.feet, mask) if enabled]
            for name in self.end_effector_names:
                if name not in self.feet and np.linalg.norm(measured_contacts[i, sensor.body_names.index(name)]) > 10.:
                    # A measured hand contact is a point at the link origin on
                    # the flat ground; no fictitious hand support is planned.
                    contacts.append(Contact(name, points=np.zeros((1, 3)), friction=self.cfg.friction,
                                            constrain_rotation=False))
            # Posture belongs to joint PD. Neither a joint task nor the
            # momentum target independently pursues policy/reference posture.
            rate, tasks = hybrid_balance_tasks(state, com_acc[i].cpu().numpy(),
                ang_acc[i].cpu().numpy(), self.base_acceleration_weight)
            objective = dict(base_body='Trunk',
                base_weight=self.force_objective_weight*weights[i, 0],
                contact_weights={name: self.force_objective_weight*weights[i, k+1]
                                 for k, name in enumerate(self.end_effector_names)},
                angular_force_scale=self.angular_force_scale,
                torque_reference=components['torque'][i, self.pin_to_sim].cpu().numpy(),
                torque_weights=self.torque_objective_weight*components['torque_weight'][i, self.pin_to_sim].cpu().numpy())
            result = self.solvers[i].solve(state, rate, contacts, tasks, self.external[i], hybrid=objective,
                                          pd_torque=pd_torque[i, self.pin_to_sim].cpu().numpy())
            torque[i, self.pin_to_sim] = torch.as_tensor(result['torque'], device=action.device, dtype=torque.dtype)
            total_torque[i, self.pin_to_sim] = torch.as_tensor(result['total_torque'], device=action.device, dtype=torque.dtype)
            for k, name in enumerate(self.end_effector_names):
                wrench = np.zeros(6)
                for point, value in result['contact_forces'].get(name, []):
                    wrench += np.r_[value, np.cross(point-state['frames'][name]['position'], value)]
                force[i, k] = torch.as_tensor(wrench, device=action.device, dtype=force.dtype)
            if self.cfg.record_diagnostics:
                record = force_diagnostics(result, state, self.solvers[i].torque_limits)
                record.update(env_id=i, time=self._clock*self._env.physics_dt,
                    desired_rate=rate, predicted_rate=result['rate'],
                    contact_acceleration_residual=result['metrics']['stance'],
                    stance_constraint_enabled=result['stance_constraint_enabled'],
                    inverse_dynamics_residual=result['inverse_residual'],
                    commanded_torque=result['total_torque'], pd_torque=result['pd_torque'],
                    feedforward_torque=result['torque'], auxiliary_base_wrench=result['auxiliary_base_wrench'],
                    reference_frame=int(times[i]), variable_count=result['variable_count'],
                    momentum_identity_residual=result['metrics']['momentum_identity'],
                    simulator_momentum_identity_residual=np.max(np.abs(state['measured_momentum']-state['Ag']@state['v'])),
                    _momentum=state['measured_momentum'].copy(),
                    _contact_velocities={c.body: state['frames'][c.body]['velocity'][:6 if c.constrain_rotation else 3].copy()
                                         for c in contacts})
                self.pending[i] = record
        self._clock += 1
        # HybridEnv applies the returned feedforward torque through its existing
        # joint PD path, including the actuator's normal effort limit handling.
        # The shared torque-limit reward must also see the combined command.
        return components['des_pos'], torque, dict(f=force.flatten(1), candidate_tau=total_torque,
            w=components['w'], com_vel=global_vel, com_angvel=global_angvel,
            com_acc=com_acc, com_angacc=ang_acc)

    def reset(self, env_ids):
        for index in env_ids:
            self.pending[int(index)] = None
            self.external[int(index)] = []

    @torch.inference_mode()
    def save_diagnostics(self, path, finalize=True, completed=True, failure_reason=''):
        from pathlib import Path
        if finalize:
            self._finish_pending(self._states())
        if not self.records:
            raise RuntimeError('No Atlas diagnostic samples recorded')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: np.asarray([r[key] for r in self.records]) for key in self.records[0]},
                            joint_names=np.asarray(self.dynamics.joint_names), foot_names=np.asarray(self.feet),
                            physics_dt=self._env.physics_dt, planned_reference_contacts=self.planned_contacts,
                            evaluation_completed=completed, failure_reason=np.asarray(failure_reason))
