"""Device-resident Atlas training action; CPU transfers only for diagnostics."""
import numpy as np
import torch

from isaaclab.managers import ActionTerm
from .atlas_action import AtlasAction
from whole_body_tracking.utils.atlas_torch_model import AtlasTorchModel, rotation_xyzw, mv
from whole_body_tracking.utils.atlas_gpu_control import AtlasGPUQP, FEET, select_state
from whole_body_tracking.utils.atlas_contact_policy import available_vertices, finite_difference, validate_contact_asset


class AtlasGPUAction(AtlasAction):
    def __init__(self, cfg, env):
        ActionTerm.__init__(self, cfg, env)
        if cfg.backend == 'batched' and torch.device(env.device).type != 'cuda':
            raise ValueError('Atlas GPU training requires --device cuda:0 (no CPU fallback). '
                             'Use actions.atlas.backend=osqp explicitly for CPU evaluation.')
        if cfg.contact_source not in ('reference', 'ground_force'):
            raise ValueError('Atlas contact_source must be reference or ground_force')
        if cfg.contact_source == 'ground_force' and cfg.contact_schedule_file:
            raise ValueError('A reference contact schedule cannot drive ground_force mode')
        if not np.isfinite([cfg.contact_on_force, cfg.contact_off_force]).all() or not 0 <= cfg.contact_off_force < cfg.contact_on_force:
            raise ValueError('Contact thresholds must satisfy 0 <= off < on and be finite')
        validate_contact_asset()
        self.dynamics, self.feet = AtlasTorchModel(env.device), FEET
        if set(self._asset.joint_names) != set(self.dynamics.joint_names):
            raise ValueError('Atlas model and simulation joint names differ')
        self.pin_to_sim = [self._asset.joint_names.index(n) for n in self.dynamics.joint_names]
        self._raw = torch.zeros((env.num_envs, 14), device=env.device)
        self._torques = torch.zeros((env.num_envs, 23), device=env.device)
        self.limits = self._asset.data.joint_effort_limits[:, self.pin_to_sim].double()
        if bool((self._asset.data.joint_stiffness != 0).any() | (self._asset.data.joint_damping != 0).any()):
            raise ValueError('Atlas total torque bounds require zero actuator PD gains')
        self.solver = AtlasGPUQP(cfg.controller, cfg.friction, cfg.residual_tolerance,
                                 cfg.batched_max_iterations, cfg.batched_tolerance, backend=cfg.backend)
        self.references = self.planned_contacts = self.contact_override = None
        self.estimated_contacts = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.external = torch.zeros((env.num_envs, len(self.dynamics.body_names), 6), dtype=torch.float64, device=env.device)
        self.has_external = False
        self.pending, self.records = [None]*env.num_envs, []
        self.validated, self._clock = False, 0
        self.failure_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.qp_failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_prediction = None
        self.previous_wrench = None
        print(f'[INFO] Atlas backend={cfg.backend}: {env.num_envs} environments; dynamics and QP on {env.device}, float64')

    def set_active_contacts(self, mask):
        if mask is None:
            self.contact_override = None
            return
        mask = torch.as_tensor(mask, device=self._env.device)
        if mask.dtype != torch.bool or mask.shape != self.estimated_contacts.shape:
            raise ValueError('Contact override must be boolean (num_envs,2)')
        self.contact_override = mask.clone()

    def set_external_wrenches(self, env_id, wrenches):
        self.external[env_id] = 0
        for wrench in wrenches:
            self.external[env_id, self.dynamics.body_names.index(wrench.body)] += torch.as_tensor(
                wrench.wrench, device=self._env.device, dtype=torch.float64)
        self.has_external = True

    def _reference_states(self):
        motion = self._env.command_manager.get_term('motion').motion
        root = self._asset.body_names.index('Trunk')
        quat = motion._body_quat_w[:, root][:, [1, 2, 3, 0]].double()
        rotation = rotation_xyzw(quat)
        q = torch.cat((motion._body_pos_w[:, root].double(), quat, motion.joint_pos[:, self.pin_to_sim].double()), -1)
        v = torch.cat((mv(rotation.transpose(-1, -2), motion._body_lin_vel_w[:, root].double()),
                       mv(rotation.transpose(-1, -2), motion._body_ang_vel_w[:, root].double()),
                       motion.joint_vel[:, self.pin_to_sim].double()), -1)
        # Cache only reference quantities consumed by feedback (not M or all
        # link Jacobians); chunk temporary dynamics memory for long motions.
        chunks = []
        for start in range(0, len(q), 256):
            state = self.dynamics.state(q[start:start+256], v[start:start+256])
            chunks.append({k: state[k] for k in ('q', 'v', 'com', 'momentum')})
            chunks[-1]['frames'] = {name: {k: state['frames'][name][k] for k in ('position', 'rotation', 'velocity')}
                                     for name in ('Trunk', *FEET)}
        self.references = {k: torch.cat([c[k] for c in chunks]) for k in ('q', 'v', 'com', 'momentum')}
        self.references['frames'] = {name: {k: torch.cat([c['frames'][name][k] for c in chunks])
                                             for k in ('position', 'rotation', 'velocity')} for name in ('Trunk', *FEET)}
        for frame in self.references['frames'].values():
            frame['acceleration'] = finite_difference(frame['velocity'], 1./float(motion.fps))
        self.planned_contacts = torch.zeros((len(q), 2), dtype=torch.bool, device=q.device)

    def _states(self):
        data = self._asset.data
        quat = data.root_link_quat_w[:, [1, 2, 3, 0]].double()
        rotation = rotation_xyzw(quat)
        q = torch.cat(((data.root_link_pos_w-self._env.scene.env_origins).double(), quat,
                       data.joint_pos[:, self.pin_to_sim].double()), -1)
        vel = data.root_link_vel_w.double()
        v = torch.cat((mv(rotation.transpose(-1, -2), vel[:, :3]), mv(rotation.transpose(-1, -2), vel[:, 3:]),
                       data.joint_vel[:, self.pin_to_sim].double()), -1)
        state = self.dynamics.state(q, v)
        if self.cfg.record_diagnostics or not self.validated:
            ids = [self._asset.body_names.index(name) for name in self.dynamics.body_names]
            pos, vel = data.body_com_pos_w[:, ids].double(), data.body_com_vel_w[:, ids].double()
            rotation = rotation_xyzw(data.body_link_quat_w[:, ids][:, :, [1, 2, 3, 0]].double())
            inertia = rotation@self.dynamics.inertia@rotation.transpose(-1, -2)
            mass = self.dynamics.mass[None, :, None]
            com = (mass*pos).sum(1)/self.dynamics.mass.sum()
            linear = mass*vel[:, :, :3]
            angular = mv(inertia, vel[:, :, 3:])+torch.cross(pos-com[:, None], linear, dim=-1)
            state['measured_momentum'] = torch.cat((linear.sum(1), angular.sum(1)), -1)
        return state

    @staticmethod
    def _cpu_states(state):
        def cpu(value):
            return {k: cpu(v) for k, v in value.items()} if isinstance(value, dict) else value.detach().cpu().numpy()
        arrays = cpu(state)
        def item(value, index):
            return {k: (v if k in ('mass', 'gravity') else item(v, index)) for k, v in value.items()} if isinstance(value, dict) else value[index]
        return [item(arrays, i) for i in range(len(arrays['q']))]

    def _finish_pending(self, states):
        if not self.cfg.record_diagnostics or self.last_prediction is None:
            return
        prediction = self.last_prediction
        actual = self._actual_wrenches()
        valid = prediction.pop('_valid')
        for i in torch.where(valid)[0].tolist():
            record = {k: v[i].detach().cpu().numpy().copy() for k, v in prediction.items()}
            record['actual_contact_wrench'] = actual[i].detach().cpu().numpy().copy()
            record['applied_torque'] = self._asset.data.applied_torque[i, self.pin_to_sim].cpu().numpy().copy()
            self.records.append(record)
        self.last_prediction = None

    def _actual_wrenches(self):
        result = self._raw.new_zeros((self._env.num_envs, 2, 6))
        for i, name in enumerate(('left_foot_ground_contact', 'right_foot_ground_contact')):
            view = self._env.scene.sensors[name].contact_physx_view
            normal, points, normals, _, counts, starts = view.get_contact_data(dt=self._env.physics_dt)
            friction, friction_points, friction_counts, friction_starts = view.get_friction_data(dt=self._env.physics_dt)
            origin = self._asset.data.body_link_pos_w[:, self._asset.body_names.index(FEET[i])]
            for forces, locations, count, start in ((normal.reshape(-1, 1)*normals, points, counts, starts),
                                                    (friction, friction_points, friction_counts, friction_starts)):
                count, start = count.reshape(-1).long(), start.reshape(-1).long()
                width = int(count.max())
                if width == 0:
                    continue
                offsets = torch.arange(width, device=self._env.device)[None]
                ids = (start[:, None]+offsets).clamp(0, len(forces)-1)
                selected = torch.where((offsets < count[:, None])[..., None], forces[ids], 0.)
                moment = torch.cross(locations[ids]-origin[:, None], selected, dim=-1)
                result[:, i, :3] += selected.sum(1)
                result[:, i, 3:] += moment.sum(1)
        return result

    def _gpu_ground_forces(self):
        forces = []
        for foot, name in zip(FEET, ('left_foot_ground_contact', 'right_foot_ground_contact')):
            sensor = self._env.scene.sensors[name]
            matrix = sensor.data.force_matrix_w
            if list(sensor.body_names) != [foot] or matrix is None or tuple(matrix.shape) != (self._env.num_envs, 1, 1, 3):
                raise RuntimeError(f'{name} must resolve one foot and one ground filter')
            forces.append(matrix[:, 0, 0])
        result = torch.stack(forces, 1)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Non-finite foot-ground contact forces')
        return result

    def apply_actions(self):
        if self.references is None:
            self._reference_states()
        state = self._states()
        if not self.validated:
            self._validate_simulator_model(self._cpu_states(state))
        self._finish_pending(state)
        times = self._env.command_manager.get_term('motion').time_steps
        ref = select_state(self.references, times)
        ground = self._gpu_ground_forces()
        self.estimated_contacts[:] = ground[..., 2] > 1.
        masks = available_vertices(state, self.cfg.controller, self._env.scene.env_origins.double())
        if self.contact_override is not None:
            masks &= self.contact_override[:, :, None]  # May remove support, never create it.
        result = self.solver.solve(state, ref, self._raw.double(), masks, self.limits,
                                    self.external if self.has_external else None)
        failed = result['failed']
        self.qp_failed |= failed
        self.failure_count += failed.long()
        fallback = (-self.cfg.controller.fallback_joint_damping*state['v'][:, 6:]).clamp(-self.limits, self.limits)
        # Actuator PD, friction and armature are zero in configure_atlas: these
        # are the complete applied torque commands, bounded including fallback.
        torque = torch.where(self.qp_failed[:, None], fallback, result['torque'])
        self._torques[:, self.pin_to_sim] = torque.to(self._torques.dtype)
        tracking_error = (state['q'][:, 7:]-ref['q'][:, 7:]).square().mean(-1).sqrt()
        saturation = (torque.abs() >= .99*self.limits).double().mean(-1)
        logs = self._env.extras.setdefault('log', {})
        logs['Atlas/solver_failure_fraction'] = failed.float().mean()
        logs['Atlas/solver_failures_total'] = self.failure_count.sum().float()
        logs['Atlas/joint_tracking_rmse'] = tracking_error.mean()
        logs['Atlas/torque_saturation_fraction'] = saturation.mean()
        logs['Atlas/predicted_normal_force'] = result['wrenches'][..., 2].mean()
        logs['Atlas/measured_normal_force'] = ground[..., 2].mean()
        if self.previous_wrench is not None and self._clock % self._env.cfg.decimation == 0:
            measured = self._actual_wrenches()
            for foot in range(2):
                for axis, label in enumerate(('fx', 'fy', 'fz', 'mx', 'my', 'mz')):
                    logs[f'Atlas/{FEET[foot]}_predicted_{label}'] = self.previous_wrench[:, foot, axis].mean()
                    logs[f'Atlas/{FEET[foot]}_actual_{label}'] = measured[:, foot, axis].mean()
            logs['Atlas/contact_wrench_prediction_rmse'] = (measured-self.previous_wrench).square().mean().sqrt()
        self.previous_wrench = result['wrenches'].detach().clone()
        if self.cfg.record_diagnostics:
            self.last_prediction = dict(_valid=torch.ones_like(failed), env_id=torch.arange(self._env.num_envs, device=self._env.device),
                time=torch.full_like(tracking_error, self._clock*self._env.physics_dt),
                predicted_contact_wrench=result['wrenches'], support_activation=result['activation'],
                contact_weights=result['contact_weights'], available_vertices=masks,
                contact_acceleration=result['contact_acceleration'], solver_failed=failed,
                joint_tracking_rmse=tracking_error, torque_saturation=saturation, commanded_torque=torque,
                qp_residual=result['residual'], foot_position_w=torch.stack([state['frames'][f]['position'] for f in FEET], 1),
                reference_foot_position_w=torch.stack([ref['frames'][f]['position'] for f in FEET], 1))
        self._asset.set_joint_effort_target(self._torques)
        self._clock += 1

    def save_diagnostics(self, path, finalize=True, completed=True, failure_reason=''):
        from pathlib import Path
        import json
        if finalize:
            self._finish_pending(None)
        if not self.records:
            raise RuntimeError('No Atlas diagnostics recorded')
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.asarray([r[k] for r in self.records]) for k in self.records[0]},
            foot_names=np.asarray(FEET), joint_names=np.asarray(self.dynamics.joint_names),
            physics_dt=self._env.physics_dt, controller_config_json=np.asarray(json.dumps(self.cfg.controller.to_dict())),
            evaluation_completed=completed, failure_reason=np.asarray(failure_reason))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.estimated_contacts[ids] = False
        self.external[ids] = 0
        self._raw[ids] = 0
        self.qp_failed[ids] = False
        if self.last_prediction is not None:
            self.last_prediction['_valid'][ids] = False
        if self.cfg.record_diagnostics:
            for i in (range(self._env.num_envs) if env_ids is None else env_ids.tolist()):
                self.pending[i] = None
