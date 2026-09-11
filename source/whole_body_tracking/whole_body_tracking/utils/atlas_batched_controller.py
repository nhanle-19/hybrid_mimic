"""GPU-batched Atlas controller with the shared HybridMimic action interface."""
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch

from .floating_model import FloatingModelController
from .hybrid import ctrl2components, highlvlPD
from .atlas_torch_model import AtlasTorchModel, rotation_xyzw, mv
from .atlas_batched_control import BatchedAtlasQP


class AtlasBatchedController(FloatingModelController):
    def __init__(self, articulation, cfg, env):
        super().__init__(articulation,cfg)
        if cfg.enforce_stance:
            raise ValueError('Batched Atlas currently requires enforce_stance=False; use backend=osqp to enforce stance')
        self.cfg,self._asset,self._env = cfg,articulation,env
        self.dynamics = AtlasTorchModel(env.device)
        if set(articulation.joint_names)!=set(self.dynamics.joint_names):
            raise ValueError('Atlas model and simulator joints differ')
        self.pin_to_sim = torch.tensor([articulation.joint_names.index(n) for n in self.dynamics.joint_names],device=env.device)
        self.body_ids = [articulation.body_names.index(n) for n in self.dynamics.body_names]
        self.feet = ('left_foot_link','right_foot_link')
        self.contact_ids = [self.end_effector_names.index(n) for n in self.feet]
        self.qp = BatchedAtlasQP(self.end_effector_names,cfg,env.device)
        if self.torque_limits.device != self.dynamics.mass.device:
            raise ValueError('Atlas articulation, dynamics and solver must use the same device')
        print(f'[INFO] Atlas backend=batched: dynamics and QP tensors on {self.dynamics.mass.device}, float64')
        self.planned_contacts = self.contact_override = None
        self.validated = False
        self._clock = 0
        self.records = []
        self.pending = None
        self.diagnostic_count = min(env.num_envs,cfg.diagnostic_env_count)
        self.external_wrenches = torch.zeros((env.num_envs,len(self.body_ids),6),dtype=torch.float64,device=env.device)
        self.has_external = False

    def _states(self):
        d = self._asset.data
        rotation = rotation_xyzw(d.root_link_quat_w[:,[1,2,3,0]].double())
        q = torch.cat(((d.root_link_pos_w-self._env.scene.env_origins).double(),
                       d.root_link_quat_w[:,[1,2,3,0]].double(),d.joint_pos[:,self.pin_to_sim].double()),-1)
        root_velocity = d.root_link_vel_w.double()
        v = torch.cat((mv(rotation.transpose(-1,-2),root_velocity[:,:3]),
                       mv(rotation.transpose(-1,-2),root_velocity[:,3:]),d.joint_vel[:,self.pin_to_sim].double()),-1)
        state = self.dynamics.state(q,v)
        state['M'][:,6:,6:] += torch.diag_embed(d.joint_armature[:,self.pin_to_sim].double())
        if self.cfg.record_diagnostics or not self.validated:
            pos = d.body_com_pos_w[:,self.body_ids].double()
            vel = d.body_com_vel_w[:,self.body_ids].double()
            body_rotation = rotation_xyzw(d.body_link_quat_w[:,self.body_ids][:,:,[1,2,3,0]].double())
            inertia = body_rotation@self.dynamics.inertia@body_rotation.transpose(-1,-2)
            mass = self.dynamics.mass[None,:,None]
            com = (mass*pos).sum(1)/mass.sum()
            linear = mass*vel[:,:,:3]
            angular = mv(inertia,vel[:,:,3:])+torch.cross(pos-com[:,None],linear,dim=-1)
            state['measured_momentum'] = torch.cat((linear.sum(1),angular.sum(1)),-1)
        return state

    def _validate(self,state):
        robot=self._asset
        masses=robot.root_physx_view.get_masses().to(self._env.device,dtype=torch.float64)[:,self.body_ids]
        if not torch.allclose(masses,self.dynamics.mass.expand_as(masses),atol=1e-5):
            raise RuntimeError('Batched Atlas simulator/model mass mismatch')
        inertias=robot.root_physx_view.get_inertias().to(self._env.device,dtype=torch.float64).reshape(self._env.num_envs,-1,3,3)[:,self.body_ids]
        if not torch.allclose(inertias,self.dynamics.inertia.expand_as(inertias),atol=1e-5):
            raise RuntimeError('Batched Atlas simulator/model inertia mismatch')
        com=robot.root_physx_view.get_coms().to(self._env.device,dtype=torch.float64)[:,self.body_ids,:3]
        if not torch.allclose(com,self.dynamics.com_local.expand_as(com),atol=1e-5):
            raise RuntimeError('Batched Atlas simulator/model CoM offset mismatch')
        expected=torch.stack([state['frames'][name]['position'] for name in self.dynamics.body_names],1)
        actual=(robot.data.body_link_pos_w[:,self.body_ids]-self._env.scene.env_origins[:,None]).double()
        if not torch.allclose(actual,expected,atol=2e-4):
            raise RuntimeError('Batched Atlas simulator/model link position mismatch')
        if not torch.allclose(state['momentum'],state['measured_momentum'],atol=1e-3):
            raise RuntimeError('Batched Atlas simulator/model momentum mismatch')
        self.validated=True

    def _reference_contacts(self):
        motion=self._env.command_manager.get_term('motion').motion
        if self.cfg.contact_schedule_file:
            schedule=np.load(self.cfg.contact_schedule_file,allow_pickle=False)
            if schedule.dtype!=bool or schedule.shape!=(len(motion.joint_pos),2):
                raise ValueError('Contact schedule must be boolean (motion_frames,2)')
            self.planned_contacts=torch.as_tensor(schedule,device=self._env.device)
            return
        root=self._asset.body_names.index('Trunk')
        quat=motion._body_quat_w[:,root][:,[1,2,3,0]].double()
        rot=rotation_xyzw(quat)
        q=torch.cat((motion._body_pos_w[:,root].double(),quat,motion.joint_pos[:,self.pin_to_sim].double()),-1)
        v=torch.cat((mv(rot.transpose(-1,-2),motion._body_lin_vel_w[:,root].double()),
                     mv(rot.transpose(-1,-2),motion._body_ang_vel_w[:,root].double()),
                     motion.joint_vel[:,self.pin_to_sim].double()),-1)
        center=self.qp.points.mean(0)
        # Bound temporary dynamics memory for long training motions. Each chunk
        # stays on the simulation device; this runs once before the first solve.
        height_chunks=[]
        for start in range(0,len(q),256):
            reference=self.dynamics.state(q[start:start+256],v[start:start+256])
            count=reference['q'].shape[0]
            height_chunks.append(torch.stack([(reference['frames'][f]['position']+mv(
                reference['frames'][f]['rotation'],center.expand(count,-1)))[:,2] for f in self.feet],-1))
        heights=torch.cat(height_chunks)
        floor=torch.quantile(heights.flatten(),.1)
        self.planned_contacts=heights<=floor+self.cfg.contact_height_tolerance

    def set_active_contacts(self,mask):
        if mask is None:
            self.contact_override=None
            return
        value=torch.as_tensor(mask,device=self._env.device)
        if value.dtype!=torch.bool or value.shape!=(self._env.num_envs,2):
            raise ValueError('Contact override must be boolean (num_envs,2)')
        self.contact_override=value.clone()

    def set_external_wrenches(self,env_id,wrenches):
        self.external_wrenches[env_id]=0
        for wrench in wrenches:
            index=self.dynamics.body_names.index(wrench.body)
            value=torch.as_tensor(wrench.wrench,device=self._env.device,dtype=torch.float64)
            if value.shape!=(6,) or not torch.isfinite(value).all():
                raise ValueError('External wrench must contain six finite values')
            self.external_wrenches[env_id,index]+=value
        self.has_external=True

    def _external(self,state):
        if not self.has_external:
            return None,None
        positions=torch.stack([state['frames'][name]['position'] for name in self.dynamics.body_names],1)
        jac=torch.stack([state['frames'][name]['J'] for name in self.dynamics.body_names],1)
        w=self.external_wrenches
        centroidal=torch.cat((w[:,:,:3].sum(1),(w[:,:,3:]+torch.cross(positions-state['com'][:,None],w[:,:,:3],dim=-1)).sum(1)),-1)
        generalized=mv(jac.transpose(-1,-2),w).sum(1)
        return centroidal,generalized

    def _save_failure(self,result,state,pd):
        index=int((~result['valid']).nonzero()[0,0])
        problem=result['problem']
        folder=Path(self.cfg.failure_directory);folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'batched_qp_failure_{uuid4().hex}.npz'
        def array(x):return x[index].detach().cpu().numpy()
        equality,inequality=problem['equality'],problem['inequality']
        matrix=torch.cat((equality,inequality),1)
        lower=torch.cat((problem['target'],torch.full_like(problem['bound'],-torch.inf)),1)
        upper=torch.cat((problem['target'],problem['bound']),1)
        np.savez_compressed(path,hessian=array(problem['hessian']),linear=array(problem['linear']),
            constraint_matrix=array(matrix),lower=array(lower),upper=array(upper),q=array(state['q']),v=array(state['v']),
            pd_torque=array(pd),candidate=array(result['x']),env_id=index,
            solver_residual=array(result['solver_info']['residual']),complementarity=array(result['solver_info']['complementarity']),
            factorization_failed=array(result['solver_info']['factorization_failed']))
        raise RuntimeError(f'Batched Atlas QP did not meet convergence/physical checks in environment {index}; saved {path}; no fallback torque applied')

    @torch.no_grad()
    def step(self,com_pos,com_vel,jacs,body_pos,base_quat,base_angvel,action,nle,lcc_rand):
        if action.shape!=(self._env.num_envs,self.action_dim) or not torch.isfinite(action).all():
            raise ValueError('Atlas expects finite batched HybridMimic actions')
        if self.planned_contacts is None:self._reference_contacts()
        state=self._states()
        if not self.validated:self._validate(state)
        self._finish_pending(state)
        c=ctrl2components(action,self.joint_count,self.end_effector_count,self.torque_limits,self.torque_limits_cost,
            self.desired_linear_velocity_scale,self.desired_angular_velocity_scale,self.torque_action_scale,
            self.linear_velocity_gain,self.angular_velocity_gain)
        linear,angular,global_vel,global_angvel=highlvlPD(base_quat,base_angvel,self.linear_velocity_gain,
            self.angular_velocity_gain,c['des_com_vel'],c['des_com_angvel'],com_vel)
        term=self._env.action_manager.get_term('joint_pos');d=self._asset.data
        targets=c['des_pos']*term._scale+term._offset
        pd=d.joint_stiffness*(targets-d.joint_pos)+d.joint_damping*(d.joint_vel_target-d.joint_vel)
        sensor=self._env.scene.sensors['contact_forces']
        sensor_ids=[sensor.body_names.index(name) for name in self.end_effector_names]
        active=torch.linalg.vector_norm(sensor.data.net_forces_w[:,sensor_ids],dim=-1)>10.
        times=self._env.command_manager.get_term('motion').time_steps
        planned=self.planned_contacts[times] if self.contact_override is None else self.contact_override
        active[:,self.contact_ids]=planned
        wext,gext=self._external(state)
        result=self.qp.solve(state,pd[:,self.pin_to_sim].double(),self.torque_limits[:,self.pin_to_sim].double(),
            torch.cat((linear,angular),-1).double(),c['w'].double(),c['torque'][:,self.pin_to_sim].double(),
            self.torque_objective_weight*c['torque_weight'][:,self.pin_to_sim].double(),active,wext,gext)
        if not bool(result['valid'].all()):self._save_failure(result,state,pd[:,self.pin_to_sim])
        ff=torch.zeros_like(pd);total=torch.zeros_like(pd)
        ff[:,self.pin_to_sim]=result['torque'].to(ff.dtype)
        total[:,self.pin_to_sim]=result['total_torque'].to(total.dtype)
        if self.cfg.record_diagnostics:self._record(result,state,active,times)
        self._clock+=1
        return c['des_pos'],ff,dict(f=result['forces'].flatten(1).to(action.dtype),candidate_tau=total,
            w=c['w'],com_vel=global_vel,com_angvel=global_angvel,com_acc=linear,com_angacc=angular)

    def _finish_pending(self,state):
        if self.pending is None:return
        d=self.diagnostic_count
        record=self.pending
        record['actual_rate']=(state['measured_momentum'][:d]-record.pop('_momentum'))/self._env.physics_dt
        velocity=torch.stack([state['frames'][f]['velocity'][:d] for f in self.feet],1)
        measured=(velocity-record.pop('_velocity'))/self._env.physics_dt
        record['actual_contact_acceleration_residual']=(measured.abs()*record['active_contact'][:,:,None]).flatten(1).amax(-1)
        record['applied_torque']=self._asset.data.applied_torque[:d,self.pin_to_sim].double().clone()
        sensor=self._env.scene.sensors['contact_forces'];ids=[sensor.body_names.index(f) for f in self.feet]
        record['sensor_normal_force_w']=sensor.data.net_forces_w[:d,ids].double().clone()
        self.records.append(record);self.pending=None

    def _record(self,result,state,active,times):
        d=self.diagnostic_count
        rho=result['rho'][:d]
        point_forces=[];cursor=0
        for name,count in zip(self.end_effector_names,self.qp.counts):
            if name in self.feet:
                k=self.end_effector_names.index(name)
                point_forces.append((rho[:,cursor:cursor+16].reshape(d,4,4)@self.qp.rays.T)*active[:d,k,None,None])
            cursor+=count*4
        force=torch.stack(point_forces,1);normal=force[:,:,:,2];tangent=torch.linalg.vector_norm(force[:,:,:,:2],dim=-1)
        utilization=torch.where(normal>1.,tangent/(self.cfg.friction*normal),torch.nan)
        sums=normal.sum(-1)
        cop=(normal[:,:,:,None]*self.qp.points[None,None,:,:2]).sum(2)/sums[:,:,None].clamp_min(1e-30)
        cop=torch.where((sums>1e-6)[:,:,None],cop,torch.nan)
        stance=torch.stack([state['frames'][f]['J'][:d]@result['acceleration'][:d,:,None]+state['frames'][f]['bias'][:d,:,None] for f in self.feet],1).squeeze(-1)
        contact=active[:d,self.contact_ids]
        record=dict(point_forces=force,normal_forces=normal,tangential_forces=tangent,friction_utilization=utilization,cop=cop,
            active_contact=contact.clone(),torque_utilization=result['total_torque'][:d].abs()/self.torque_limits[:d,self.pin_to_sim],
            env_id=torch.arange(d,device=self._env.device),time=torch.full((d,),self._clock*self._env.physics_dt,device=self._env.device,dtype=torch.float64),
            desired_rate=result['desired_rate'][:d],predicted_rate=result['rate'][:d],
            contact_acceleration_residual=(stance.abs()*contact[:,:,None]).flatten(1).amax(-1),
            stance_constraint_enabled=torch.zeros(d,device=self._env.device,dtype=torch.bool),
            inverse_dynamics_residual=result['inverse_residual'][:d],commanded_torque=result['total_torque'][:d],
            pd_torque=result['pd_torque'][:d],feedforward_torque=result['torque'][:d],
            auxiliary_base_wrench=result['auxiliary_base_wrench'][:d],reference_frame=times[:d].clone(),
            variable_count=torch.full((d,),result['x'].shape[-1],device=self._env.device),
            momentum_identity_residual=(state['momentum'][:d]-mv(state['Ag'][:d],state['v'][:d])).abs().amax(-1),
            simulator_momentum_identity_residual=(state['measured_momentum'][:d]-state['momentum'][:d]).abs().amax(-1),
            _momentum=state['measured_momentum'][:d].clone(),
            _velocity=torch.stack([state['frames'][f]['velocity'][:d] for f in self.feet],1).clone(),
            _valid=torch.ones(d,device=self._env.device,dtype=torch.bool))
        self.pending=record

    def reset(self,env_ids):
        self.external_wrenches[env_ids]=0
        if self.qp.warm_start is not None and 'valid' in self.qp.warm_start:
            self.qp.warm_start['valid'][env_ids] = False
        if self.pending is not None:
            ids=env_ids[env_ids<self.diagnostic_count]
            self.pending['_valid'][ids]=False

    @torch.inference_mode()
    def save_diagnostics(self,path,finalize=True,completed=True,failure_reason=''):
        if finalize:self._finish_pending(self._states())
        if not self.records:raise RuntimeError('No Atlas diagnostics recorded')
        valid=torch.cat([r['_valid'] for r in self.records])
        values={key:torch.cat([r[key] for r in self.records])[valid].cpu().numpy()
                for key in self.records[0] if key!='_valid'}
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**values,joint_names=np.asarray(self.dynamics.joint_names),foot_names=np.asarray(self.feet),
            physics_dt=self._env.physics_dt,planned_reference_contacts=self.planned_contacts.cpu().numpy(),
            evaluation_completed=completed,failure_reason=np.asarray(failure_reason))
