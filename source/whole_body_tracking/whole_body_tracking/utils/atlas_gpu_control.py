"""Batched version of the reference-conditioned contact Atlas reference controller and physical QP.

All dynamics, objective assembly, equality elimination and solves stay on the
input torch device. The four contact patterns are batched separately so absent
feet contribute neither forces nor stance constraints.
"""
import torch

try:
    from .atlas_torch_model import mv, skew
    from .atlas_batched_qp import solve_batched_qp
    from .atlas_contact_policy import decode_actions, sole_vertices
except ImportError:
    from atlas_torch_model import mv, skew
    from atlas_batched_qp import solve_batched_qp
    from atlas_contact_policy import decode_actions, sole_vertices

FEET = ('left_foot_link', 'right_foot_link')


def select_state(state, ids):
    return {k: ({name: {key: value[ids] for key, value in frame.items()}
                 for name, frame in v.items()} if k == 'frames' else
                v if k in ('mass', 'gravity') else v[ids]) for k, v in state.items()}


def rotation_log(r):
    """SO(3) logarithm, including the neighborhood of a half turn."""
    vector = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                          r[..., 1, 0]-r[..., 0, 1]), -1) / 2
    sine = vector.norm(dim=-1)
    cosine = ((r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1., 1.)
    angle = torch.atan2(sine, cosine)
    regular = vector * (angle/sine.clamp_min(1e-15))[..., None]
    # Near pi, the antisymmetric part vanishes; recover the axis from R+I.
    symmetric = (r+r.transpose(-1, -2))/2
    diagonal = ((symmetric.diagonal(dim1=-2, dim2=-1)+1)/2).clamp_min(0).sqrt()
    index = diagonal.argmax(-1)
    axis_rows = (symmetric+torch.eye(3, device=r.device, dtype=r.dtype))/2
    axis = axis_rows.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    axis = torch.where((axis*vector).sum(-1, keepdim=True) < 0, -axis, axis)
    return torch.where((cosine < -0.999999)[..., None], axis*angle[..., None], regular)


def reference_tasks(state, ref, action, cfg, active):
    nj = state['M'].shape[-1]-6
    desired_linear = ref['momentum'][:, :3]
    desired_angular = ref['momentum'][:, 3:]
    rate = torch.cat((state['mass']*cfg.com_position_gain*(ref['com']-state['com'])+
                      cfg.linear_momentum_gain*(desired_linear-state['momentum'][:, :3]),
                      cfg.angular_momentum_gain*(desired_angular-state['momentum'][:, 3:])), -1)
    target = cfg.posture_position_gain*(ref['q'][:, 7:]-state['q'][:, 7:])
    target += cfg.posture_velocity_gain*(ref['v'][:, 6:]-state['v'][:, 6:])
    identity = torch.eye(nj+6, device=action.device, dtype=action.dtype)[6:].expand(len(action), -1, -1)
    tasks = [(identity, target, cfg.posture_weight)]
    pelvis, desired = state['frames']['Trunk'], ref['frames']['Trunk']
    angular = cfg.pelvis_position_gain*rotation_log(desired['rotation']@pelvis['rotation'].transpose(-1, -2))
    angular += cfg.pelvis_velocity_gain*(desired['velocity'][:, 3:]-pelvis['velocity'][:, 3:])
    tasks.append((pelvis['J'][:, 3:], angular-pelvis['bias'][:, 3:], cfg.pelvis_weight))
    if cfg.swing_weight:
        for foot in FEET:
            if foot in active:
                continue
            current, desired = state['frames'][foot], ref['frames'][foot]
            error = torch.cat((desired['position']-current['position'],
                               rotation_log(desired['rotation']@current['rotation'].transpose(-1, -2))), -1)
            acceleration = cfg.swing_position_gain*error+cfg.swing_velocity_gain*(desired['velocity']-current['velocity'])
            tasks.append((current['J'], acceleration-current['bias'], cfg.swing_weight))
    return rate, tasks


class AtlasGPUQP:
    def __init__(self, cfg, friction=.6, tolerance=2e-5, max_iterations=100, solver_tolerance=1e-9, backend="batched"):
        self.cfg, self.friction, self.tolerance = cfg, friction, tolerance
        self.backend = backend
        self.max_iterations, self.solver_tolerance = max_iterations, solver_tolerance

    def solve(self, state, reference, actions, masks, limits, external=None):
        n, nv = state['bias'].shape
        activation, weights = decode_actions(actions, self.cfg)
        if masks.shape == (n, 2):
            masks = masks[:, :, None].expand(-1, -1, 4)
        if masks.shape != (n, 2, 4) or masks.dtype != torch.bool:
            raise ValueError('Actual support geometry must be boolean (N,2,4)')
        point_masks = masks & (activation > 0)[:, :, None]
        masks = point_masks.any(-1)
        failed = torch.zeros(n, dtype=torch.bool, device=actions.device)
        wrenches = actions.new_zeros((n, 2, 6))
        contact_acceleration = actions.new_zeros((n, 2, 6))
        torque = torch.empty_like(limits)
        acceleration = torch.empty_like(state['bias'])
        rho = actions.new_zeros((n, 32))
        rates, desired_rates = actions.new_empty((n, 6)), actions.new_empty((n, 6))
        residuals = actions.new_empty(n)
        # Constant four-pattern loop, independent of the number of environments.
        for pattern in range(4):
            ids = torch.where((masks[:, 0].long()+2*masks[:, 1].long()) == pattern)[0]
            if ids.numel() == 0:
                continue
            s, ref = select_state(state, ids), select_state(reference, ids)
            active = [foot for i, foot in enumerate(FEET) if pattern & (1 << i)]
            rate, tasks = reference_tasks(s, ref, actions[ids], self.cfg, active)
            result = self._solve_group(s, rate, tasks, active, limits[ids],
                                       None if external is None else external[ids], weights[ids], activation[ids], point_masks[ids], ref)
            torque[ids], acceleration[ids], rates[ids] = result['torque'], result['acceleration'], result['rate']
            desired_rates[ids], residuals[ids] = rate, result['residual']
            failed[ids] = result['failed']
            for j, foot in enumerate(FEET):
                contact_acceleration[ids, j] = result['contact_acceleration'][foot]
            for i, foot in enumerate(active):
                start = FEET.index(foot)*16
                rho[ids, start:start+16] = result['rho'][:, i*16:(i+1)*16]
                wrenches[ids, FEET.index(foot)] = result['wrenches'][foot]
        return dict(torque=torque, acceleration=acceleration, rho=rho, rate=rates,
                    desired_rate=desired_rates, residual=residuals, contact_weights=weights,
                    contact_acceleration=contact_acceleration, activation=activation, available=point_masks,
                    failed=failed, wrenches=wrenches)

    def _solve_group(self, s, rate, tasks, active, limits, external, weights, activation, point_masks, ref):
        n, nv = s['bias'].shape
        nr, nx = 16*len(active), nv+16*len(active)
        zeros = lambda *shape: s['M'].new_zeros(shape)
        tensor = lambda value: torch.as_tensor(value, device=s['M'].device, dtype=s['M'].dtype)
        points = sole_vertices(s['q'].device, s['q'].dtype)
        mu = self.friction
        rays = tensor([[0., 0., -mu, mu], [mu, -mu, 0., 0.], [1., 1., 1., 1.]])
        qmap, gmap = zeros(n, 6, nr), zeros(n, nv, nr)
        for i, foot in enumerate(active):
            frame = s['frames'][foot]
            offsets = points[FEET.index(foot)]@frame['rotation'].transpose(-1, -2)
            for j in range(4):
                offset = offsets[:, j]
                location = frame['position']+offset
                jp = frame['J'][:, :3]-skew(offset)@frame['J'][:, 3:]
                sl = slice(i*16+j*4, i*16+j*4+4)
                qmap[:, :, sl] = torch.cat((rays.expand(n, -1, -1), skew(location-s['com'])@rays), 1)
                gmap[:, :, sl] = jp.transpose(-1, -2)@rays
        wg = torch.cat((s['mass']*s['gravity'], zeros(3))).expand(n, -1)
        wext, gext = zeros(n, 6), zeros(n, nv)
        if external is not None:
            for i, frame in enumerate(s['frames'].values()):
                wrench = external[:, i]
                wext += torch.cat((wrench[:, :3], wrench[:, 3:]+torch.cross(frame['position']-s['com'], wrench[:, :3], dim=-1)), -1)
                gext += mv(frame['J'].transpose(-1, -2), wrench)
        torque_map = torch.cat((s['M'][:, 6:], -gmap[:, 6:]), -1)
        torque_bias = s['bias'][:, 6:]-gext[:, 6:]
        objective = torch.cat((s['Ag'], zeros(n, 6, nr)), -1)
        wh = tensor(self.cfg.momentum_weights)
        hessian = objective.transpose(-1, -2)@(wh[None, :, None]*objective)
        hessian += torch.diag(torch.cat((tensor(self.cfg.acceleration_weight).expand(nv), tensor(self.cfg.force_weight).expand(nr))))
        linear = -mv(objective.transpose(-1, -2), wh*(rate-s['Ag_bias']))
        eq, rhs = [torch.cat((s['Ag'], -qmap), -1)], [wg+wext-s['Ag_bias']]
        # All six body-frame motion requests are soft, including swing bodies.
        # Spatial vectors and Jacobians are world-aligned [linear, angular] at
        # each body origin. a_ref differentiates that same spatial velocity.
        for i, foot in enumerate(FEET):
            frame, desired = s['frames'][foot], ref['frames'][foot]
            target_acceleration = desired['acceleration']+tensor(self.cfg.contact_velocity_gain)*(desired['velocity']-frame['velocity'])
            j = torch.cat((frame['J'], zeros(n, 6, nr)), -1)
            weight = weights[:, i]
            hessian += j.transpose(-1, -2)@(weight[:, :, None]*j)
            linear += mv(j.transpose(-1, -2), weight*(frame['bias']-target_acceleration))
        # Unsupported collider corners have exactly zero rho. Group-independent
        # elimination below drops those variables rather than using infeasible
        # pairs of strict barrier inequalities at zero capacity.
        for jacobian, target, weight in tasks:
            j = torch.cat((jacobian, zeros(n, jacobian.shape[1], nr)), -1)
            hessian += weight*j.transpose(-1, -2)@j
            linear -= weight*mv(j.transpose(-1, -2), target)
        equality, target = torch.cat(eq, 1), torch.cat(rhs, 1)
        inequality = torch.cat((torque_map, -torque_map, torch.cat((zeros(n, nr, nv), -torch.eye(nr, device=s['q'].device, dtype=s['q'].dtype).expand(n, -1, -1)), -1)), 1)
        bound = torch.cat((limits-torque_bias, limits+torque_bias, zeros(n, nr)), -1)
        if active:
            normal_rows = zeros(n, len(active), nx)
            capacities = []
            for i, foot in enumerate(active):
                # Every friction ray has unit world-normal component, so Fn=sum rho.
                normal_rows[:, i, nv+i*16:nv+(i+1)*16] = 1.
                capacities.append(activation[:, FEET.index(foot)]*self.cfg.contact_force_max[FEET.index(foot)])
            inequality = torch.cat((inequality, normal_rows), 1)
            bound = torch.cat((bound, torch.stack(capacities, -1)), -1)
        enabled = torch.cat([point_masks[:, FEET.index(f)] for f in active], -1) if active else torch.ones((n, 0), dtype=torch.bool, device=s['q'].device)
        # At most 256 geometry patterns for the existing two four-corner boxes.
        codes = (enabled.long()*(2**torch.arange(enabled.shape[1], device=s['q'].device))).sum(-1)
        x, converged = zeros(n, nx), torch.zeros(n, dtype=torch.bool, device=s['q'].device)
        for code in torch.unique(codes):
            ids = torch.where(codes == code)[0]
            keep_rho = enabled[ids[0]].repeat_interleave(4)
            keep = torch.cat((torch.ones(nv, dtype=torch.bool, device=s['q'].device), keep_rho))
            # Drop nonnegativity rows belonging to eliminated zero coefficients.
            rows = torch.cat((torch.ones(2*(nv-6), dtype=torch.bool, device=s['q'].device), keep_rho,
                              torch.ones(len(active), dtype=torch.bool, device=s['q'].device)))
            h0, g0 = hessian[ids][:, keep][:, :, keep], linear[ids][:, keep]
            e, target0 = equality[ids][:, :, keep], target[ids]
            gmat, bounds = inequality[ids][:, rows][:, :, keep], bound[ids][:, rows]
            q, r = torch.linalg.qr(e.transpose(-1, -2), mode='complete')
            ne = e.shape[1]
            offset = mv(q[:, :, :ne], torch.linalg.solve_triangular(r[:, :ne].transpose(-1, -2), target0[..., None], upper=False).squeeze(-1))
            basis = q[:, :, ne:]
            h = basis.transpose(-1, -2)@h0@basis
            g = mv(basis.transpose(-1, -2), g0+mv(h0, offset))
            if self.backend == 'osqp':
                reduced, ok = self._osqp(h, g, gmat@basis, bounds-mv(gmat, offset))
            else:
                reduced, info = solve_batched_qp(h, g, gmat@basis, bounds-mv(gmat, offset),
                    max_iterations=self.max_iterations, tolerance=self.solver_tolerance)
                ok = info['converged']
            full = zeros(len(ids), nx)
            full[:, keep] = offset+mv(basis, reduced)
            x[ids], converged[ids] = full, ok
        torque = mv(torque_map, x)+torque_bias
        acceleration, rho = x[:, :nv], x[:, nv:]
        inverse = mv(s['M'], acceleration)+s['bias']-mv(gmap, rho)-gext-torch.cat((zeros(n, 6), torque), -1)
        residual = torch.stack(((mv(equality, x)-target).abs().amax(-1),
                                (mv(inequality, x)-bound).clamp_min(0).amax(-1),
                                inverse.abs().amax(-1),
                                (s['momentum']-mv(s['Ag'], s['v'])).abs().amax(-1)), -1).amax(-1)
        valid = converged & torch.isfinite(x).all(-1) & (residual <= self.tolerance)
        # Failure is explicit; the action term applies bounded damping torque.
        torque = torch.where(valid[:, None], torque, 0.)
        acceleration = torch.where(valid[:, None], acceleration, 0.)
        rho = torch.where(valid[:, None], rho, 0.)
        wrenches = {}
        for i, foot in enumerate(active):
            frame = s['frames'][foot]
            offsets = points[FEET.index(foot)]@frame['rotation'].transpose(-1, -2)
            forces = rho[:, i*16:(i+1)*16].reshape(n, 4, 4)@rays.T
            wrenches[foot] = torch.cat((forces.sum(1), torch.cross(offsets, forces, dim=-1).sum(1)), -1)
        return dict(torque=torque, acceleration=acceleration, rho=rho,
                    rate=mv(s['Ag'], acceleration)+s['Ag_bias'], residual=residual,
                    contact_acceleration={foot: mv(s['frames'][foot]['J'], acceleration)+s['frames'][foot]['bias'] for foot in FEET}, failed=~valid, wrenches=wrenches)

    @staticmethod
    def _osqp(hessian, linear, inequality, bound):
        import numpy as np
        import osqp
        from scipy import sparse
        out = torch.zeros_like(linear)
        valid = torch.zeros(len(linear), dtype=torch.bool, device=linear.device)
        for i in range(len(linear)):
            solver = osqp.OSQP()
            scale = hessian[i].diagonal().clamp_min(1e-12).rsqrt()
            h = hessian[i]*scale[:, None]*scale[None, :]
            g = inequality[i]*scale[None, :]
            row = g.abs().amax(-1).clamp_min(1e-12).reciprocal()
            solver.setup(P=sparse.csc_matrix(np.triu(h.cpu().numpy())), q=(linear[i]*scale).cpu().numpy(),
                         A=sparse.csc_matrix((g*row[:, None]).cpu().numpy()), l=np.full(bound.shape[1], -np.inf),
                         u=(bound[i]*row).cpu().numpy(), eps_abs=1e-10, eps_rel=1e-10, max_iter=100000, verbose=False, polish=True)
            result = solver.solve()
            if result.info.status_val == 1 and result.x is not None:
                out[i] = torch.as_tensor(result.x, device=linear.device)*scale
                valid[i] = True
        return out, valid
