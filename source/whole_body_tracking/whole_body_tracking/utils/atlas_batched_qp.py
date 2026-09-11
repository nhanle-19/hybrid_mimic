"""Device-resident batched convex QP solver; no environment loop or CPU fallback.

Mehrotra predictor/corrector interior point method for strictly convex objectives
and linear inequalities. Atlas eliminates its six auxiliary-base equality rows
analytically before calling this kernel.
"""
import torch


def bmv(a, x):
    return (a@x.unsqueeze(-1)).squeeze(-1)


def _step_length(value, direction, fraction=1.):
    ratio = torch.where(direction < 0, -value/direction.clamp_max(-1e-30), torch.full_like(value, torch.inf))
    return (fraction*ratio.amin(-1)).clamp(max=1.)


def solve_batched_qp(hessian, linear, inequality, bound, *, max_iterations=40, tolerance=1e-8):
    """Solve all batch members together; return unscaled x and convergence data.

    x minimizes .5*x.T*H*x+g.T*x subject to G*x <= h. All tensors are float64
    on one device. Failed members are reported, never replaced/clipped.
    """
    if any(t.device != hessian.device or t.dtype != torch.float64
           for t in (hessian, linear, inequality, bound)):
        raise ValueError('Batched QP inputs must be float64 on the same device')
    n, size, _ = hessian.shape
    # Variable and row equilibration, preserving the optimization problem.
    scale = hessian.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).rsqrt()
    p = hessian*scale[:, :, None]*scale[:, None, :]
    q = linear*scale
    g = inequality*scale[:, None, :]
    row_scale = g.abs().amax(-1).clamp_min(1e-12).reciprocal()
    g = g*row_scale[:, :, None]
    h = bound*row_scale
    x = torch.zeros_like(q)
    slack = h.abs().clamp_min(1.)
    dual = torch.ones_like(slack)
    eye = torch.eye(size, dtype=p.dtype, device=p.device)[None]
    converged = torch.zeros(n, dtype=torch.bool, device=p.device)
    factorization_failed = torch.zeros_like(converged)
    iteration = 0
    for iteration in range(max_iterations):
        rd = bmv(p, x)+q+bmv(g.transpose(-1, -2), dual)
        ri = bmv(g, x)+slack-h
        mu = (slack*dual).mean(-1)
        residual = torch.maximum(rd.abs().amax(-1)/(1+q.abs().amax(-1)), ri.abs().amax(-1)/(1+h.abs().amax(-1)))
        converged |= (residual <= tolerance) & (mu <= tolerance)
        # One batch synchronization per check, never one per environment.
        if iteration % 5 == 0 and bool(converged.all()):
            break
        ratio = dual/slack
        k = p+g.transpose(-1, -2)@(ratio[:, :, None]*g)
        # Barrier curvature grows near active bounds. Equilibrate the Newton
        # system as well as the original QP before its GPU factorization.
        newton_scale = k.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30).rsqrt()
        equilibrated = k*newton_scale[:, :, None]*newton_scale[:, None, :]
        chol, info = torch.linalg.cholesky_ex(equilibrated+1e-14*eye)
        if bool((info != 0).any()):
            factorization_failed = info != 0
            break  # Return failure status so the controller can dump the QP.
        def direction(centering):
            rhs = -rd+bmv(g.transpose(-1, -2), (centering-dual*ri)/slack)
            dx = torch.cholesky_solve((rhs*newton_scale).unsqueeze(-1), chol).squeeze(-1)*newton_scale
            ds = -ri-bmv(g, dx)
            dz = (-centering-dual*ds)/slack
            return dx, ds, dz
        dx, ds, dz = direction(slack*dual)
        ap, ad = _step_length(slack, ds), _step_length(dual, dz)
        mu_aff = ((slack+ap[:, None]*ds)*(dual+ad[:, None]*dz)).mean(-1)
        # Keep sufficient centering to avoid collapsing the barrier before
        # stationarity converges on heavily saturated training actions.
        sigma = ((mu_aff/mu.clamp_min(1e-30)).clamp(0., 1.)**3).clamp_min(.1)
        dx, ds, dz = direction(slack*dual+ds*dz-sigma[:, None]*mu[:, None])
        ap, ad = _step_length(slack, ds, .995), _step_length(dual, dz, .995)
        ap, ad = ap.masked_fill(converged, 0.), ad.masked_fill(converged, 0.)
        x = x+ap[:, None]*dx
        slack = (slack+ap[:, None]*ds).clamp_min(1e-18)
        dual = (dual+ad[:, None]*dz).clamp_min(1e-18)
    rd = bmv(p, x)+q+bmv(g.transpose(-1, -2), dual)
    ri = bmv(g, x)+slack-h
    mu = (slack*dual).mean(-1)
    residual = torch.maximum(rd.abs().amax(-1)/(1+q.abs().amax(-1)), ri.abs().amax(-1)/(1+h.abs().amax(-1)))
    converged = (residual <= tolerance) & (mu <= tolerance) & ~factorization_failed
    return x*scale, dict(converged=converged, iterations=iteration+1, residual=residual,
                        complementarity=mu, factorization_failed=factorization_failed)


def eliminate_equalities(hessian, linear, equality, target, eliminated=6):
    """Eliminate final variables whose equality block is square/invertible."""
    n, size, _ = hessian.shape
    kept = size-eliminated
    e = equality[:, :, -eliminated:]
    bottom = torch.linalg.solve(e, -equality[:, :, :kept])
    offset_bottom = torch.linalg.solve(e, target.unsqueeze(-1)).squeeze(-1)
    transform = torch.cat((torch.eye(kept, device=hessian.device, dtype=hessian.dtype).expand(n, -1, -1), bottom), 1)
    offset = torch.cat((torch.zeros((n, kept), device=hessian.device, dtype=hessian.dtype), offset_bottom), -1)
    reduced_h = transform.transpose(-1, -2)@hessian@transform
    reduced_g = bmv(transform.transpose(-1, -2), linear+bmv(hessian, offset))
    return reduced_h, reduced_g, transform, offset
