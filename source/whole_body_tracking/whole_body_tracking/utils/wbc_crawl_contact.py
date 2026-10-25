"""Eight-region crawl FORCE QP using actual mesh contact points.

This is a solver/geometry interface, independent of Isaac task registration.
Regions can span several rigid bodies (notably knees). Every force therefore
uses its own body's point Jacobian; no knee/hand force is applied at a foot.
"""
from pathlib import Path

import numpy as np
import torch

try:
    from .wbc_acc_torch_model import mv, skew
    from .wbc_force_gpu_control import WBCForceGPUQP
except ImportError:
    from wbc_acc_torch_model import mv, skew
    from wbc_force_gpu_control import WBCForceGPUQP


CRAWL_REGIONS = ('left_hand', 'right_hand', 'left_knee', 'right_knee',
                 'left_forearm', 'right_forearm', 'right_foot', 'left_upper_arm')


class CrawlContactGeometry:
    """Re-evaluate annotated body-local mesh vertices at the current pose.

    Up to four distinct vertices per region form a conservative support patch:
    the lowest eligible vertex, followed by farthest-point samples in XY.
    Sampling only removes admissible moment arms; it never invents new points.
    This supports sliding: there are no stationary-contact equalities.
    """

    def __init__(self, annotation_path, device='cpu', dtype=torch.float64):
        self.path = Path(annotation_path)
        with np.load(self.path, allow_pickle=False) as data:
            if int(data['annotation_version']) != 3:
                raise ValueError('Crawl contacts require version 3 full-mesh annotations')
            names = data['surface_names'].tolist()
            self.body_names = data['body_names'].tolist()
            self.regions = []
            for name in CRAWL_REGIONS:
                ids = np.flatnonzero(data['point_surface_index'] == names.index(name))
                if not len(ids):
                    raise ValueError(f'Missing mesh vertices for {name}')
                local = data['point_position_body'][ids]
                bodies = data['point_body_index'][ids]
                if not np.isfinite(local).all() or (bodies <= 0).any():
                    raise ValueError(f'Invalid body-local mesh vertices for {name}')
                self.regions.append((torch.as_tensor(local, device=device, dtype=dtype),
                                     torch.as_tensor(bodies, device=device, dtype=torch.long),
                                     torch.as_tensor(ids, device=device, dtype=torch.long)))

    def at(self, state, cfg):
        n, nv = state['bias'].shape
        sample_count = 4
        reference = state['q']
        zeros = lambda *shape: reference.new_zeros(shape)
        positions, jacobians = zeros(n, 8, sample_count, 3), zeros(n, 8, sample_count, 3, nv)
        biases = zeros(n, 8, sample_count, 3)
        masks = torch.zeros((n, 8, sample_count), device=reference.device, dtype=torch.bool)
        point_ids = torch.full((n, 8, sample_count), -1, device=reference.device, dtype=torch.long)
        clearances = zeros(n, 8)
        batch = torch.arange(n, device=reference.device)
        gap_tolerance = float(cfg.contact_gap_tolerance)
        patch_tolerance = float(cfg.contact_patch_height_tolerance)
        penetration = float(cfg.max_contact_penetration)
        if not np.isfinite([gap_tolerance, patch_tolerance, penetration, cfg.ground_height]).all() or min(gap_tolerance, patch_tolerance, penetration) < 0:
            raise ValueError('Invalid crawl floor tolerances')
        for region, (local, bodies, ids) in enumerate(self.regions):
            if local.device != reference.device or local.dtype != reference.dtype:
                raise ValueError('Contact geometry and dynamics must use the same device/dtype')
            world = zeros(n, len(local), 3)
            for body in bodies.unique().tolist():
                members = bodies == body
                frame = state['frames'][self.body_names[body]]
                world[:, members] = local[members]@frame['rotation'].transpose(-1, -2)+frame['position'][:, None]
            gaps = world[..., 2]-cfg.ground_height
            clearances[:, region] = gaps.amin(-1)
            # A deeply penetrating region is invalid rather than a license to
            # take forces from other vertices farther up its surface.
            eligible = ((gaps <= gap_tolerance) & (gaps >= -penetration)
                        & (gaps <= gaps.amin(-1, keepdim=True)+patch_tolerance)
                        & (world[..., :2].abs() < 100.).all(-1)
                        & (gaps.amin(-1, keepdim=True) >= -penetration))
            remaining = eligible.clone()
            nearest = torch.full_like(gaps, torch.inf)
            for slot in range(sample_count):
                if slot == 0:
                    selected = torch.where(remaining, gaps, torch.inf).argmin(-1)
                else:
                    selected = torch.where(remaining, nearest, -torch.inf).argmax(-1)
                valid = remaining.any(-1)
                chosen = world[batch, selected]
                positions[:, region, slot] = chosen
                masks[:, region, slot] = valid
                point_ids[:, region, slot] = torch.where(valid, ids[selected], -1)
                selected_bodies = bodies[selected]
                for body in bodies.unique().tolist():
                    frame = state['frames'][self.body_names[body]]
                    offset = mv(frame['rotation'], local[selected])
                    jp = frame['J'][:, :3]-skew(offset)@frame['J'][:, 3:]
                    omega, alpha = frame['velocity'][:, 3:], frame['bias'][:, 3:]
                    bias = (frame['bias'][:, :3]+torch.cross(alpha, offset, dim=-1)
                            +torch.cross(omega, torch.cross(omega, offset, dim=-1), dim=-1))
                    choose = selected_bodies == body
                    jacobians[choose, region, slot] = jp[choose]
                    biases[choose, region, slot] = bias[choose]
                remaining[batch, selected] = False
                distance = (world[..., :2]-chosen[:, None, :2]).square().sum(-1)
                nearest = torch.minimum(nearest, distance)
        geometry = dict(position=positions, J=jacobians, bias=biases,
                        origin=state['com'][:, None].expand(-1, 8, -1),
                        point_ids=point_ids, clearance=clearances)
        return geometry, masks


class WBCCrawlForceQP(WBCForceGPUQP):
    """Eight floor-normal bounds from decoded NN outputs and live geometry."""

    def __init__(self, cfg, geometry, **kwargs):
        # Many nearly coincident patch vertices make force sharing weakly
        # conditioned. Tight stationarity is needed for OSQP torque parity,
        # even when looser solves already satisfy the physical constraints.
        kwargs.setdefault('solver_tolerance', 1e-12)
        kwargs.setdefault('max_iterations', 150)
        super().__init__(cfg, **kwargs)
        if self.contact_count != len(CRAWL_REGIONS):
            raise ValueError('Crawl FORCE requires eight force limits in CRAWL_REGIONS order')
        self.geometry = geometry

    def solve(self, state, reference, actions, limits, external=None):
        if not bool(torch.isfinite(state['q']).all() & torch.isfinite(state['v']).all()):
            raise ValueError('Non-finite crawl state')
        geometry, actual_mask = self.geometry.at(state, self.cfg)
        near_floor_mask = actual_mask.clone()
        point_velocity = torch.einsum('ncpij,nj->ncpi', geometry['J'], state['v'])
        separating_velocity = getattr(self.cfg, 'contact_separation_velocity', None)
        if separating_velocity is not None:
            if not np.isfinite(separating_velocity) or separating_velocity <= 0:
                raise ValueError('Contact separation velocity must be finite and positive')
            separating = ((geometry['position'][..., 2] > self.cfg.ground_height)
                          & (point_velocity[..., 2] > separating_velocity))
            actual_mask = actual_mask & ~separating
        result = super().solve(state, reference, actions, actual_mask, limits, external,
                               contact_geometry=geometry)
        # The eight decoded NN outputs are the only requested force ceilings.
        # Live geometry can remove unavailable support; reference annotations
        # are deliberately absent from the solver API.
        result['requested_normal_force_limits'] = result['normal_force_limits']
        result['normal_force_limits'] = result['normal_force_limits']*actual_mask.any(-1)
        result.update(actual_available=actual_mask, near_floor_available=near_floor_mask,
                      point_position_w=geometry['position'], point_ids=geometry['point_ids'],
                      point_velocity_w=point_velocity,
                      clearance=geometry['clearance'], wrench_origin_w=geometry['origin'])
        return result
