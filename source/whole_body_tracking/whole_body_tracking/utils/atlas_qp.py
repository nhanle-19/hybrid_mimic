"""Atlas-style centroidal QP with physical point contacts and hard torque bounds.

No simulator imports: usable for analytical residual tests and offline evaluation.
"""
from dataclasses import dataclass, field

import numpy as np
import osqp
from scipy import sparse


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@dataclass
class Contact:
    body: str
    # Conservative sole rectangle in the foot-link frame, at the collision sole.
    points: np.ndarray = field(default_factory=lambda: np.array([
        [-.09, -.04, -.03], [-.09, .04, -.03], [.10, -.04, -.03], [.10, .04, -.03]]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    friction: float = .6
    constrain_rotation: bool = True


@dataclass
class MotionTask:
    jacobian: np.ndarray
    target_minus_bias: np.ndarray
    weight: float
    name: str = 'motion'
    hard: bool = False


@dataclass
class ExternalWrench:
    """Known world-aligned [force,moment] acting at the named body-frame origin."""
    body: str
    wrench: np.ndarray


def friction_rays(normal, mu):
    normal = np.asarray(normal, dtype=float)
    if mu <= 0 or not np.isfinite(mu) or normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise ValueError('Invalid contact normal or friction')
    n = normal/np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    t1 = np.cross(n, axis); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    # Diamond inscribed in the circular Coulomb cone: never exceeds mu*Fn.
    return np.stack([n+mu*t1, n-mu*t1, n+mu*t2, n-mu*t2], axis=1)


class AtlasQP:
    def __init__(self, torque_limits, momentum_weights=(1., 1., 1., 10., 10., 10.),
                 force_weight=1e-5, acceleration_weight=1e-5, tolerance=2e-5,
                 failure_directory=None):
        self.failure_directory = failure_directory
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        if not np.all(np.isfinite(self.torque_limits)&(self.torque_limits > 0)):
            raise ValueError('Torque limits must be finite and positive')
        self.wh = np.asarray(momentum_weights, dtype=float)
        self.wrho, self.wa, self.tolerance = force_weight, acceleration_weight, tolerance
        if self.wh.shape != (6,) or np.any(self.wh <= 0) or force_weight <= 0 or acceleration_weight <= 0:
            raise ValueError('Objective weights must be positive')

    def solve(self, state, desired_rate, contacts, tasks=(), external=(), *, hybrid=None, pd_torque=None):
        nv = state['M'].shape[0]
        if self.torque_limits.shape != (nv-6,):
            raise ValueError('Torque-limit dimension mismatch')
        pd_torque = np.zeros(nv-6) if pd_torque is None else np.asarray(pd_torque, dtype=float)
        if pd_torque.shape != (nv-6,) or not np.isfinite(pd_torque).all():
            raise ValueError('PD torque must be a finite vector in model joint order')
        if len({c.body for c in contacts}) != len(contacts):
            raise ValueError('Duplicate load-bearing body')
        nr = sum(4*len(c.points) for c in contacts)
        # HybridMimic's first wrench logit weights an explicit auxiliary base
        # wrench. Keep it out of the strict physical-QP mode used by offline tests.
        nb = 6 if hybrid is not None else 0
        nx = nv+nr+nb
        qmap, gmap = np.zeros((6, nr)), np.zeros((nv, nr))
        point_maps, stance = [], []
        cursor = 0
        for contact in contacts:
            frame = state['frames'][contact.body]
            stance.append((frame, 6 if contact.constrain_rotation else 3))
            rays = friction_rays(contact.normal, contact.friction)
            for point in np.asarray(contact.points):
                offset = frame['rotation']@point
                location = frame['position']+offset
                jp = frame['J'][:3]-skew(offset)@frame['J'][3:]
                sl = slice(cursor, cursor+4)
                qmap[:, sl] = np.vstack([rays, skew(location-state['com'])@rays])
                gmap[:, sl] = jp.T@rays
                point_maps.append((contact, location, rays, sl))
                cursor += 4
        base_g = np.zeros((nv, nb))
        base_q = np.zeros((6, nb))
        if nb:
            frame = state['frames'][hybrid['base_body']]
            base_g = frame['J'].T
            base_q = np.block([[np.eye(3), np.zeros((3, 3))],
                               [skew(frame['position']-state['com']), np.eye(3)]])
        force_g = np.hstack([gmap, base_g])
        force_q = np.hstack([qmap, base_q])
        wg = np.r_[state['mass']*state['gravity'], np.zeros(3)]
        wext, gext = np.zeros(6), np.zeros(nv)
        for ext in external:
            frame = state['frames'][ext.body]
            wrench = np.asarray(ext.wrench, dtype=float)
            wext += np.r_[wrench[:3], wrench[3:]+np.cross(frame['position']-state['com'], wrench[:3])]
            gext += frame['J'].T@wrench
        torque_map = np.hstack([state['M'][6:], -force_g[6:]])
        # Inverse dynamics determines TOTAL actuator torque. Return only the
        # feedforward correction; the runtime adds its known PD contribution.
        torque_bias = state['bias'][6:]-gext[6:]-pd_torque
        objective = np.hstack([state['Ag'], np.zeros((6, nr+nb))])
        target = np.asarray(desired_rate)-state['Ag_bias']
        hessian = objective.T@np.diag(self.wh)@objective+np.diag(np.r_[np.full(nv, self.wa), np.full(nr+nb, self.wrho)])
        linear = -objective.T@(self.wh*target)
        if hybrid is not None:
            # Same exp(-logit) wrench penalties and normalized torque reference
            # objective as HybridMimic; contact moments arise from point forces.
            axis_weights = np.r_[np.ones(3), np.full(3, hybrid['angular_force_scale'])]
            for body, weight in hybrid['contact_weights'].items():
                wrench_map = np.zeros((6, nx))
                for contact, location, rays, sl in point_maps:
                    if contact.body == body:
                        wrench_map[:, nv+sl.start:nv+sl.stop] = np.vstack([
                            rays, skew(location-state['frames'][body]['position'])@rays])
                hessian += wrench_map.T@np.diag(weight*axis_weights)@wrench_map
            hessian[-6:, -6:] += np.diag(hybrid['base_weight']*axis_weights)
            weights = np.asarray(hybrid['torque_weights'])
            target_tau = np.asarray(hybrid['torque_reference'])-torque_bias
            hessian += torque_map.T@(weights[:, None]*torque_map)
            linear -= torque_map.T@(weights*target_tau)
        eq = [np.hstack([state['Ag'], -force_q])]
        rhs = [wg+wext-state['Ag_bias']]
        for frame, rows in stance:
            eq.append(np.hstack([frame['J'][:rows], np.zeros((rows, nr+nb))]))
            rhs.append(-frame['bias'][:rows])
        for task in tasks:
            j = np.hstack([task.jacobian, np.zeros((len(task.target_minus_bias), nr+nb))])
            if task.hard:
                eq.append(j); rhs.append(task.target_minus_bias)
            else:
                hessian += task.weight*j.T@j
                linear -= task.weight*j.T@task.target_minus_bias
        aeq, beq = np.vstack(eq), np.concatenate(rhs)
        rows = [aeq, torque_map]
        lower = [beq, -self.torque_limits-pd_torque-torque_bias]
        upper = [beq, self.torque_limits-pd_torque-torque_bias]
        if nr:
            rows.append(np.hstack([np.zeros((nr, nv)), np.eye(nr), np.zeros((nr, nb))]))
            lower.append(np.zeros(nr)); upper.append(np.full(nr, np.inf))
        constraint_matrix = np.vstack(rows)
        lower_bounds, upper_bounds = np.concatenate(lower), np.concatenate(upper)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian)), q=linear,
                     A=sparse.csc_matrix(constraint_matrix), l=lower_bounds, u=upper_bounds,
                     eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True, verbose=False)
        result = solver.solve()
        initial_status = result.info.status
        # Numeric status codes differ between OSQP 0.6 and 1.x.
        retried = initial_status.lower() in ('solved inaccurate', 'maximum iterations reached')
        objective_scale = 1.
        if retried:
            import warnings
            warnings.warn(f'Atlas QP: {initial_status}; retrying with normalized objective. '
                          'All torque/contact constraints and residual checks remain active.', RuntimeWarning)
            # Multiplication by a positive scalar preserves the QP minimizer
            # and relative objective weights. Constraint rows stay in physical
            # units; no slack, torque clipping, or inaccurate solution is accepted.
            objective_scale = max(1., float(np.max(np.abs(np.diag(hessian)))))
            retry_solver = osqp.OSQP()
            retry_solver.setup(P=sparse.csc_matrix(np.triu(hessian/objective_scale)), q=linear/objective_scale,
                A=sparse.csc_matrix(constraint_matrix), l=lower_bounds, u=upper_bounds,
                eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True, verbose=False,
                adaptive_rho_interval=25)
            if result.x is not None and np.isfinite(result.x).all():
                retry_solver.warm_start(x=result.x)
            result = retry_solver.solve()
        def fail(reason):
            dump = ''
            if self.failure_directory is not None:
                from pathlib import Path
                from uuid import uuid4
                directory = Path(self.failure_directory)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory/f'qp_failure_{uuid4().hex}.npz'
                np.savez_compressed(path, hessian=hessian, linear=linear,
                    constraint_matrix=constraint_matrix, lower=lower_bounds, upper=upper_bounds,
                    q=state['q'], v=state['v'], pd_torque=pd_torque,
                    status=np.asarray(result.info.status), initial_status=np.asarray(initial_status),
                    objective_scale=objective_scale, iterations=result.info.iter, failure_reason=np.asarray(reason))
                dump = f'; problem saved to {path}'
            raise RuntimeError(f'Atlas QP failed: {reason} after {result.info.iter} iterations'
                               f'{dump}; no clipped/fallback torque applied')

        if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
            fail(result.info.status)
        x = result.x
        acceleration, rho = x[:nv], x[nv:nv+nr]
        base_wrench = x[nv+nr:]
        torque = torque_map@x+torque_bias
        total_torque = pd_torque+torque
        rate = state['Ag']@acceleration+state['Ag_bias']
        inverse = state['M']@acceleration+state['bias']-force_g@x[nv:]-gext-np.r_[np.zeros(6), total_torque]
        contact_forces = {}
        normal_violation, friction_violation = 0., 0.
        for contact, location, rays, sl in point_maps:
            force = rays@rho[sl]
            normal = np.asarray(contact.normal)/np.linalg.norm(contact.normal)
            fn = normal@force
            ft = np.linalg.norm(force-fn*normal)
            normal_violation = max(normal_violation, -fn)
            friction_violation = max(friction_violation, ft-contact.friction*fn)
            contact_forces.setdefault(contact.body, []).append((location, force))
        stance_residual = np.concatenate([f['J'][:rows]@acceleration+f['bias'][:rows]
                                         for f, rows in stance]) if stance else np.zeros(0)
        metrics = dict(momentum_identity=float(np.max(np.abs(state['momentum']-state['Ag']@state['v']))),
                       momentum_rate=float(np.max(np.abs(rate-wg-force_q@x[nv:]-wext))),
                       stance=float(np.max(np.abs(stance_residual), initial=0)),
                       base_inverse_dynamics=float(np.max(np.abs(inverse[:6]))),
                       inverse_dynamics=float(np.max(np.abs(inverse))),
                       torque_violation=float(np.max(np.maximum(np.abs(total_torque)-self.torque_limits, 0))),
                       rho_violation=float(np.max(np.maximum(-rho, 0), initial=0)),
                       normal_force_violation=float(normal_violation),
                       friction_violation=float(friction_violation),
                       equality=float(np.max(np.abs(aeq@x-beq))))
        if max(metrics.values()) > self.tolerance:
            fail(f'constraint residual check: {metrics}')
        return dict(acceleration=acceleration, rho=rho, torque=torque, rate=rate,
                    solver_retried=retried,
                    pd_torque=pd_torque.copy(), total_torque=total_torque,
                    auxiliary_base_wrench=base_wrench,
                    desired_rate=np.asarray(desired_rate), contact_forces=contact_forces,
                    contacts=contacts, metrics=metrics, inverse_residual=inverse,
                    external_centroidal_wrench=wext, gravity_wrench=wg,
                    stance_residual=stance_residual, iterations=result.info.iter, variable_count=nx)
