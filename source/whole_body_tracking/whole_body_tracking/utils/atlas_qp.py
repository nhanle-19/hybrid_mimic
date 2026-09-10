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
                 force_weight=1e-5, acceleration_weight=1e-5, tolerance=2e-5):
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        if not np.all(np.isfinite(self.torque_limits)&(self.torque_limits > 0)):
            raise ValueError('Torque limits must be finite and positive')
        self.wh = np.asarray(momentum_weights, dtype=float)
        self.wrho, self.wa, self.tolerance = force_weight, acceleration_weight, tolerance
        if self.wh.shape != (6,) or np.any(self.wh <= 0) or force_weight <= 0 or acceleration_weight <= 0:
            raise ValueError('Objective weights must be positive')

    def solve(self, state, desired_rate, contacts, tasks=(), external=()):
        nv = state['M'].shape[0]
        if self.torque_limits.shape != (nv-6,):
            raise ValueError('Torque-limit dimension mismatch')
        if len({c.body for c in contacts}) != len(contacts):
            raise ValueError('Duplicate load-bearing body')
        nr = sum(4*len(c.points) for c in contacts)
        nx = nv+nr
        qmap, gmap = np.zeros((6, nr)), np.zeros((nv, nr))
        point_maps, stance = [], []
        cursor = 0
        for contact in contacts:
            frame = state['frames'][contact.body]
            stance.append(frame)
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
        wg = np.r_[state['mass']*state['gravity'], np.zeros(3)]
        wext, gext = np.zeros(6), np.zeros(nv)
        for ext in external:
            frame = state['frames'][ext.body]
            wrench = np.asarray(ext.wrench, dtype=float)
            wext += np.r_[wrench[:3], wrench[3:]+np.cross(frame['position']-state['com'], wrench[:3])]
            gext += frame['J'].T@wrench
        torque_map = np.hstack([state['M'][6:], -gmap[6:]])
        torque_bias = state['bias'][6:]-gext[6:]
        objective = np.hstack([state['Ag'], np.zeros((6, nr))])
        target = np.asarray(desired_rate)-state['Ag_bias']
        hessian = objective.T@np.diag(self.wh)@objective+np.diag(np.r_[np.full(nv, self.wa), np.full(nr, self.wrho)])
        linear = -objective.T@(self.wh*target)
        eq = [np.hstack([state['Ag'], -qmap])]
        rhs = [wg+wext-state['Ag_bias']]
        for frame in stance:
            eq.append(np.hstack([frame['J'], np.zeros((6, nr))]))
            rhs.append(-frame['bias'])
        for task in tasks:
            j = np.hstack([task.jacobian, np.zeros((len(task.target_minus_bias), nr))])
            if task.hard:
                eq.append(j); rhs.append(task.target_minus_bias)
            else:
                hessian += task.weight*j.T@j
                linear -= task.weight*j.T@task.target_minus_bias
        aeq, beq = np.vstack(eq), np.concatenate(rhs)
        rows = [aeq, torque_map]
        lower = [beq, -self.torque_limits-torque_bias]
        upper = [beq, self.torque_limits-torque_bias]
        if nr:
            rows.append(np.hstack([np.zeros((nr, nv)), np.eye(nr)]))
            lower.append(np.zeros(nr)); upper.append(np.full(nr, np.inf))
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(hessian)), q=linear,
                     A=sparse.csc_matrix(np.vstack(rows)), l=np.concatenate(lower), u=np.concatenate(upper),
                     eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polish=True, verbose=False)
        result = solver.solve()
        if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
            raise RuntimeError(f'Atlas QP failed: {result.info.status}; no clipped/fallback torque applied')
        x = result.x
        acceleration, rho = x[:nv], x[nv:]
        torque = torque_map@x+torque_bias
        rate = state['Ag']@acceleration+state['Ag_bias']
        inverse = state['M']@acceleration+state['bias']-gmap@rho-gext-np.r_[np.zeros(6), torque]
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
        stance_residual = np.concatenate([f['J']@acceleration+f['bias'] for f in stance]) if stance else np.zeros(0)
        metrics = dict(momentum_identity=float(np.max(np.abs(state['momentum']-state['Ag']@state['v']))),
                       momentum_rate=float(np.max(np.abs(rate-wg-qmap@rho-wext))),
                       stance=float(np.max(np.abs(stance_residual), initial=0)),
                       base_inverse_dynamics=float(np.max(np.abs(inverse[:6]))),
                       inverse_dynamics=float(np.max(np.abs(inverse))),
                       torque_violation=float(np.max(np.maximum(np.abs(torque)-self.torque_limits, 0))),
                       rho_violation=float(np.max(np.maximum(-rho, 0), initial=0)),
                       normal_force_violation=float(normal_violation),
                       friction_violation=float(friction_violation),
                       equality=float(np.max(np.abs(aeq@x-beq))))
        if max(metrics.values()) > self.tolerance:
            raise RuntimeError(f'Atlas QP physical residual check failed: {metrics}')
        return dict(acceleration=acceleration, rho=rho, torque=torque, rate=rate,
                    desired_rate=np.asarray(desired_rate), contact_forces=contact_forces,
                    contacts=contacts, metrics=metrics, inverse_residual=inverse,
                    external_centroidal_wrench=wext, gravity_wrench=wg,
                    stance_residual=stance_residual, iterations=result.info.iter, variable_count=nx)
