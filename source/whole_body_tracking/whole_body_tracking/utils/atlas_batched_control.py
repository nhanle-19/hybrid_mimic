"""Batched assembly of the same Atlas hybrid objective and physical constraints."""
import torch
try:
    from .atlas_torch_model import skew, mv
    from .atlas_batched_qp import bmv, eliminate_equalities, solve_batched_qp
except ImportError:
    from atlas_torch_model import skew, mv
    from atlas_batched_qp import bmv, eliminate_equalities, solve_batched_qp


class BatchedAtlasQP:
    def __init__(self, contact_names, cfg, device='cpu', dtype=torch.float64):
        self.names, self.cfg = list(contact_names), cfg
        self.device, self.dtype = torch.device(device), dtype
        self.tensor = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
        self.points = self.tensor([[-.09,-.04,-.03],[-.09,.04,-.03],[.10,-.04,-.03],[.10,.04,-.03]])
        # Same ordering as friction_rays([0,0,1], mu) in the OSQP reference.
        mu = cfg.friction
        self.rays = self.tensor([[0.,0.,-mu,mu],[mu,-mu,0.,0.],[1.,1.,1.,1.]])
        self.counts = [4 if name in ('left_foot_link','right_foot_link') else 1 for name in self.names]
        self.nr = 4*sum(self.counts)

    def assemble(self, state, pd, torque_limits, acceleration, logits, torque_reference, torque_weights, active,
                 external_centroidal=None, external_generalized=None):
        cfg = self.cfg
        n, nv, _ = state['M'].shape
        nr, size = self.nr, nv+self.nr+6
        qmap = torch.zeros((n, 6, nr), device=self.device, dtype=self.dtype)
        gmap = torch.zeros((n, nv, nr), device=self.device, dtype=self.dtype)
        contact_maps = []
        cursor = 0
        for k, (name, count) in enumerate(zip(self.names, self.counts)):
            frame = state['frames'][name]
            points = self.points if count == 4 else self.tensor([[0.,0.,0.]])
            offsets = (frame['rotation'][:, None]@points[None, :, :, None]).squeeze(-1)
            locations = frame['position'][:, None]+offsets
            jp = frame['J'][:, None, :3]-skew(offsets)@frame['J'][:, None, 3:]
            rays = self.rays[None, None]*active[:, k, None, None, None]
            qblock = torch.cat((rays.expand(n,count,-1,-1), skew(locations-state['com'][:, None])@rays), 2)
            gblock = jp.transpose(-1,-2)@rays
            width = count*4
            qmap[:, :, cursor:cursor+width] = qblock.permute(0,2,1,3).reshape(n,6,width)
            gmap[:, :, cursor:cursor+width] = gblock.permute(0,2,1,3).reshape(n,nv,width)
            local_wrench = torch.cat((rays.expand(n,count,-1,-1), skew(offsets)@rays), 2)
            wmap = torch.zeros((n,6,size), device=self.device, dtype=self.dtype)
            wmap[:, :, nv+cursor:nv+cursor+width] = local_wrench.permute(0,2,1,3).reshape(n,6,width)
            contact_maps.append(wmap)
            cursor += width
        base = state['frames']['Trunk']
        eye = torch.eye(3, device=self.device, dtype=self.dtype).expand(n,-1,-1)
        zeros = torch.zeros_like(eye)
        base_q = torch.cat((torch.cat((eye,zeros),-1), torch.cat((skew(base['position']-state['com']),eye),-1)),1)
        force_g = torch.cat((gmap, base['J'].transpose(-1,-2)), -1)
        force_q = torch.cat((qmap, base_q), -1)
        wg = torch.cat((state['mass']*state['gravity'], self.tensor([0.,0.,0.])), -1).expand(n,-1)
        wext = torch.zeros_like(wg) if external_centroidal is None else external_centroidal
        gext = torch.zeros((n,nv), device=self.device, dtype=self.dtype) if external_generalized is None else external_generalized
        torque_map = torch.cat((state['M'][:,6:], -force_g[:,6:]), -1)
        total_bias = state['bias'][:,6:]-gext[:,6:]
        ff_bias = total_bias-pd
        ag = torch.cat((state['Ag'], torch.zeros((n,6,nr+6),device=self.device,dtype=self.dtype)),-1)
        nominal = torch.zeros((n,nv),device=self.device,dtype=self.dtype)
        nominal[:,:6] = torch.linalg.solve(base['J'][:,:,:6], (acceleration-base['bias']).unsqueeze(-1)).squeeze(-1)
        rate = bmv(state['Ag'],nominal)+state['Ag_bias']
        wh = self.tensor(cfg.momentum_weights)
        hessian = ag.transpose(-1,-2)@(wh[None,:,None]*ag)
        regularizer = self.tensor([cfg.acceleration_weight]*nv+[cfg.force_weight]*(nr+6))
        hessian = hessian+torch.diag_embed(regularizer.expand(n,-1))
        linear = -bmv(ag.transpose(-1,-2),wh*(rate-state['Ag_bias']))
        weights = torch.exp(-logits.clamp(-cfg.force_logit_clip,cfg.force_logit_clip))*cfg.force_objective_weight
        axes = self.tensor([1.,1.,1.]+[cfg.angular_force_scale]*3)
        for k, wmap in enumerate(contact_maps):
            hessian = hessian+wmap.transpose(-1,-2)@(weights[:,k+1,None,None]*axes[None,:,None]*wmap)
        hessian[:,-6:,-6:] += torch.diag_embed(weights[:,0,None]*axes)
        hessian = hessian+torque_map.transpose(-1,-2)@(torque_weights[:,:,None]*torque_map)
        linear = linear-bmv(torque_map.transpose(-1,-2),torque_weights*(torque_reference-ff_bias))
        task_j = torch.cat((base['J'],torch.zeros((n,6,nr+6),device=self.device,dtype=self.dtype)),-1)
        hessian = hessian+cfg.base_acceleration_weight*(task_j.transpose(-1,-2)@task_j)
        linear = linear-cfg.base_acceleration_weight*bmv(task_j.transpose(-1,-2),acceleration-base['bias'])
        equality = torch.cat((state['Ag'],-force_q),-1)
        target = wg+wext-state['Ag_bias']
        rho_map = torch.zeros((n,nr,size),device=self.device,dtype=self.dtype)
        rho_map[:,:,nv:nv+nr] = -torch.eye(nr,device=self.device,dtype=self.dtype)
        inequality = torch.cat((torque_map,-torque_map,rho_map),1)
        bound = torch.cat((torque_limits-total_bias,torque_limits+total_bias,torch.zeros((n,nr),device=self.device,dtype=self.dtype)),-1)
        return dict(hessian=hessian,linear=linear,equality=equality,target=target,inequality=inequality,bound=bound,
                    torque_map=torque_map,total_bias=total_bias,ff_bias=ff_bias,force_g=force_g,force_q=force_q,
                    contact_maps=torch.stack(contact_maps,1),rate=rate,gravity_wrench=wg,external=wext,gext=gext)

    def solve(self, state, pd, torque_limits, acceleration, logits, torque_reference, torque_weights, active,
              external_centroidal=None, external_generalized=None):
        if self.cfg.enforce_stance:
            raise ValueError('Batched Atlas currently implements the stance-disabled experiment; select backend=osqp for hard stance constraints')
        problem = self.assemble(state,pd,torque_limits,acceleration,logits,torque_reference,torque_weights,active,
                                external_centroidal,external_generalized)
        p,g,t,c = eliminate_equalities(problem['hessian'],problem['linear'],problem['equality'],problem['target'])
        inequalities = problem['inequality']@t
        bounds = problem['bound']-bmv(problem['inequality'],c)
        reduced, info = solve_batched_qp(p,g,inequalities,bounds,max_iterations=self.cfg.batched_max_iterations,
                                        tolerance=self.cfg.batched_tolerance)
        x = bmv(t,reduced)+c
        nv = state['M'].shape[-1]
        qdd, rho = x[:,:nv],x[:,nv:-6]
        total = bmv(problem['torque_map'],x)+problem['total_bias']
        torque = total-pd
        inverse = bmv(state['M'],qdd)+state['bias']-bmv(problem['force_g'],x[:,nv:])-problem['gext']
        inverse = inverse-torch.cat((torch.zeros_like(total[:,:6]),total),-1)
        forces = (problem['contact_maps']@x[:,None,:,None]).squeeze(-1)
        predicted_rate = bmv(state['Ag'],qdd)+state['Ag_bias']
        equality_error = (bmv(problem['equality'],x)-problem['target']).abs().amax(-1)
        inequality_error = (bmv(problem['inequality'],x)-problem['bound']).clamp_min(0).amax(-1)
        valid = info['converged'] & torch.isfinite(x).all(-1) & (equality_error<=self.cfg.residual_tolerance) & (inequality_error<=self.cfg.residual_tolerance) & (inverse.abs().amax(-1)<=self.cfg.residual_tolerance)
        return dict(torque=torque,total_torque=total,pd_torque=pd,acceleration=qdd,rho=rho,forces=forces,
                    auxiliary_base_wrench=x[:,-6:],desired_rate=problem['rate'],rate=predicted_rate,
                    inverse_residual=inverse,equality_error=equality_error,inequality_error=inequality_error,
                    valid=valid,solver_info=info,problem=problem,x=x)
