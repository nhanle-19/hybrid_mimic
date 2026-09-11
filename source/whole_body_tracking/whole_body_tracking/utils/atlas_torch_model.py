"""Batched analytical T1 dynamics on a torch device, using the exported USD tree.

The only Python loop traverses the fixed robot tree, never the environment batch.
Coordinates match AtlasModel: root translation/quaternion XYZW, local root twist.
"""
import json
from pathlib import Path
import torch

MODEL_PATH = Path(__file__).resolve().parents[1]/'assets/booster/t1/atlas_dynamics.json'


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def rotation_xyzw(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-15)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def mv(a, v):
    return (a @ v.unsqueeze(-1)).squeeze(-1)


class AtlasTorchModel:
    def __init__(self, device='cpu', dtype=torch.float64, path=MODEL_PATH):
        self.device, self.dtype = torch.device(device), dtype
        self.description = json.loads(Path(path).read_text())
        self.body_names = [self.description['root']]
        self.joint_names, self.tree = [], []
        def tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)
        self.tensor = tensor
        def children(parent):
            for j in self.description['joints']:
                if j['parent'] != parent:
                    continue
                parent_index = self.body_names.index(parent)
                self.body_names.append(j['child'])
                self.joint_names.append(j['name'])
                axis = torch.eye(3, device=device, dtype=dtype)['XYZ'.index(j['axis'])]
                self.tree.append((parent_index, tensor(j['pos0']), rotation_xyzw(tensor(j['quat0_xyzw'])),
                                  tensor(j['pos1']), rotation_xyzw(tensor(j['quat1_xyzw'])), axis))
                children(j['child'])
        children(self.body_names[0])
        self.nv = len(self.joint_names)+6
        self.mass = tensor([self.description['bodies'][b]['mass'] for b in self.body_names])
        self.com_local = tensor([self.description['bodies'][b]['com'] for b in self.body_names])
        principal = rotation_xyzw(tensor([self.description['bodies'][b]['principal_axes_xyzw'] for b in self.body_names]))
        diagonal = tensor([self.description['bodies'][b]['inertia_diagonal'] for b in self.body_names])
        self.inertia = principal @ torch.diag_embed(diagonal) @ principal.transpose(-1, -2)
        self.gravity = tensor([0., 0., -9.81])
        self.eye3 = torch.eye(3, device=device, dtype=dtype)

    def state(self, q, v):
        q, v = q.to(dtype=self.dtype), v.to(dtype=self.dtype)
        n = q.shape[0]
        root_r = rotation_xyzw(q[:, 3:7])
        root_v, root_w = mv(root_r, v[:, :3]), mv(root_r, v[:, 3:6])
        root_j = torch.zeros((n, 6, self.nv), device=q.device, dtype=q.dtype)
        root_j[:, :3, :3], root_j[:, 3:, 3:6] = root_r, root_r
        positions, rotations, velocities, omegas = [q[:, :3]], [root_r], [root_v], [root_w]
        accelerations, alphas, jacobians = [torch.cross(root_w, root_v, dim=-1)], [torch.zeros_like(root_w)], [root_j]
        for index, (parent, p0, r0, p1, r1, axis) in enumerate(self.tree):
            rp, pp, wp = rotations[parent], positions[parent], omegas[parent]
            offset0 = mv(rp, p0.expand(n, -1))
            pivot = pp+offset0
            axis_world = mv(rp@r0, axis.expand(n, -1))
            angle, speed = q[:, 7+index], v[:, 6+index]
            ax = skew(axis)
            joint_r = self.eye3+torch.sin(angle)[:, None, None]*ax+(1-torch.cos(angle))[:, None, None]*(ax@ax)
            rc = rp@r0@joint_r@r1.T
            offset1 = mv(rc, p1.expand(n, -1))
            pc = pivot-offset1
            wc = wp+axis_world*speed[:, None]
            alphac = alphas[parent]+torch.cross(wp, axis_world*speed[:, None], dim=-1)
            vp = velocities[parent]+torch.cross(wp, offset0, dim=-1)
            ap = accelerations[parent]+torch.cross(alphas[parent], offset0, dim=-1)+torch.cross(wp, torch.cross(wp, offset0, dim=-1), dim=-1)
            vc = vp-torch.cross(wc, offset1, dim=-1)
            ac = ap-torch.cross(alphac, offset1, dim=-1)-torch.cross(wc, torch.cross(wc, offset1, dim=-1), dim=-1)
            jc = jacobians[parent].clone()
            jc[:, :3] -= skew(pc-pp)@jc[:, 3:]
            jc[:, :3, 6+index] = torch.cross(axis_world, pc-pivot, dim=-1)
            jc[:, 3:, 6+index] = axis_world
            positions.append(pc); rotations.append(rc); velocities.append(vc); omegas.append(wc)
            accelerations.append(ac); alphas.append(alphac); jacobians.append(jc)
        pos, rot = torch.stack(positions, 1), torch.stack(rotations, 1)
        vel, omega = torch.stack(velocities, 1), torch.stack(omegas, 1)
        acc, alpha, jac = torch.stack(accelerations, 1), torch.stack(alphas, 1), torch.stack(jacobians, 1)
        local_offset = mv(rot, self.com_local.expand(n, -1, -1))
        body_com = pos+local_offset
        vc = vel+torch.cross(omega, local_offset, dim=-1)
        ac = acc+torch.cross(alpha, local_offset, dim=-1)+torch.cross(omega, torch.cross(omega, local_offset, dim=-1), dim=-1)
        jv, jw = jac[:, :, :3]-skew(local_offset)@jac[:, :, 3:], jac[:, :, 3:]
        inertia = rot@self.inertia@rot.transpose(-1, -2)
        linear_j = self.mass[None, :, None, None]*jv
        angular_j = inertia@jw
        mass_matrix = (jv.transpose(-1, -2)@linear_j+jw.transpose(-1, -2)@angular_j).sum(1)
        force = self.mass[None, :, None]*(ac-self.gravity)
        moment = mv(inertia, alpha)+torch.cross(omega, mv(inertia, omega), dim=-1)
        bias = (mv(jv.transpose(-1, -2), force)+mv(jw.transpose(-1, -2), moment)).sum(1)
        mass = self.mass.sum()
        com = (self.mass[None, :, None]*body_com).sum(1)/mass
        offset = body_com-com[:, None]
        ag = torch.cat((linear_j.sum(1), (angular_j+skew(offset)@linear_j).sum(1)), 1)
        linear_rate = self.mass[None, :, None]*ac
        ag_bias = torch.cat((linear_rate.sum(1), (moment+torch.cross(offset, linear_rate, dim=-1)).sum(1)), -1)
        linear = self.mass[None, :, None]*vc
        momentum = torch.cat((linear.sum(1), (mv(inertia, omega)+torch.cross(offset, linear, dim=-1)).sum(1)), -1)
        frames = {name: dict(position=pos[:, i], rotation=rot[:, i], J=jac[:, i],
                            bias=torch.cat((acc[:, i], alpha[:, i]), -1),
                            velocity=torch.cat((vel[:, i], omega[:, i]), -1))
                  for i, name in enumerate(self.body_names)}
        return dict(q=q, v=v, M=mass_matrix, bias=bias, Ag=ag, Ag_bias=ag_bias,
                    com=com, momentum=momentum, mass=mass, gravity=self.gravity, frames=frames)
