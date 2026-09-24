"""Ground-only contact measurements and hysteresis for simulator QP tests."""
import numpy as np


class GroundContactEstimator:
    """Per-foot Schmitt trigger: activate above on_N, release below off_N.

    Each update is one physics step. Reset starts with no measured support;
    there is no reference- or geometry-based fallback before the first force.
    """

    def __init__(self, num_envs, on_N=10., off_N=5.):
        if not np.isfinite([on_N, off_N]).all() or not 0 <= off_N < on_N:
            raise ValueError('Contact thresholds must satisfy 0 <= off_N < on_N and be finite')
        self.on_N, self.off_N = on_N, off_N
        self.active = np.zeros((num_envs, 2), dtype=bool)

    def update(self, normal_force_z):
        forces = np.asarray(normal_force_z)
        if forces.shape != self.active.shape or not np.isfinite(forces).all():
            raise ValueError('Ground normal forces must be finite (num_envs, 2)')
        self.active[:] = np.where(self.active, forces > self.off_N, forces >= self.on_N)
        return self.active.copy()

    def reset(self, env_ids=None):
        self.active[slice(None) if env_ids is None else env_ids] = False


def read_ground_forces(sensors, feet, num_envs):
    """Read ground-filtered world normal forces, never net/all-object forces.

    One single-body sensor per foot avoids PhysX many-to-many filtering.
    The WBCACC floor has one collision shape and must resolve to one filter.
    Returned shape is (num_envs, 2, 3), left then right.
    """
    forces = []
    for foot, name in zip(feet, ('left_foot_ground_contact', 'right_foot_ground_contact')):
        sensor = sensors[name]
        matrix = sensor.data.force_matrix_w
        if list(sensor.body_names) != [foot] or matrix is None or tuple(matrix.shape) != (num_envs, 1, 1, 3):
            raise RuntimeError(f'{name} must resolve one foot and one ground filter; check the WBCACC floor path')
        forces.append(matrix[:, 0, 0].detach().cpu().numpy())
    result = np.stack(forces, axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite foot-ground contact forces')
    return result
