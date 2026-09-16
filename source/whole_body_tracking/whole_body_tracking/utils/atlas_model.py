"""Analytical floating-base dynamics from the simulator's exported USD tree.

Pinocchio tangent coordinates use LOCAL root linear/angular velocity followed by
scalar joint velocities. Spatial vectors are [linear, angular]; centroidal vectors
are world aligned and centered at the whole-robot CoM.
"""
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

MODEL_PATH = Path(__file__).resolve().parents[1]/'assets/booster/t1/atlas_dynamics.json'
FEET = ('left_foot_link', 'right_foot_link')


class AtlasModel:
    def __init__(self, path=MODEL_PATH):
        self.description = json.loads(Path(path).read_text())
        self.model = pin.Model()
        m = self.model
        root_id = m.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), 'root_joint')
        locations = {self.description['root']: (root_id, pin.SE3.Identity())}
        def pose(pos, quat):
            q = pin.Quaternion(np.asarray(quat, dtype=float)); q.normalize()
            return pin.SE3(q.matrix(), np.asarray(pos, dtype=float))
        def add_body(name, jid, placement):
            b = self.description['bodies'][name]
            r = pose([0, 0, 0], b['principal_axes_xyzw']).rotation
            inertia = pin.Inertia(b['mass'], np.asarray(b['com']), r@np.diag(b['inertia_diagonal'])@r.T)
            m.appendBodyToJoint(jid, inertia, placement)
            m.addFrame(pin.Frame(name, jid, 0, placement, pin.FrameType.BODY))
        add_body(self.description['root'], root_id, pin.SE3.Identity())
        # CRBA requires contiguous subtrees: insert joints in depth-first order.
        visited = set()
        def add_children(parent_name):
            for j in self.description['joints']:
                if j['parent'] != parent_name:
                    continue
                if j['child'] in visited:
                    raise ValueError('Cyclic rigid-body tree')
                visited.add(j['child'])
                parent, parent_link = locations[j['parent']]
                axis = np.eye(3)['XYZ'.index(j['axis'])]
                jid = m.addJoint(parent, pin.JointModelRevoluteUnaligned(axis),
                                parent_link*pose(j['pos0'], j['quat0_xyzw']), j['name'])
                placement = pose(j['pos1'], j['quat1_xyzw']).inverse()
                locations[j['child']] = (jid, placement)
                add_body(j['child'], jid, placement)
                add_children(j['child'])
        add_children(self.description['root'])
        if len(visited) != len(self.description['joints']):
            raise ValueError('Disconnected rigid-body tree')
        self.data = m.createData()
        self.joint_names = list(m.names)[2:]
        self.mass = sum(i.mass for i in m.inertias)
        self.frame_ids = {name: m.getFrameId(name) for name in self.description['bodies']}

    def state(self, q, v):
        """Refresh every matrix at this state; no stale or finite-differenced Jacobians."""
        m, d = self.model, self.data
        q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        if q.shape != (m.nq,) or v.shape != (m.nv,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('Invalid generalized state')
        if np.linalg.norm(q[3:7]) < 1e-12:
            raise ValueError('Invalid zero root quaternion')
        # Isaac states are float32: normalize before analytical algorithms, which
        # assume a unit free-flyer quaternion even when the error is only 1e-8.
        q = pin.normalize(m, q)
        pin.computeAllTerms(m, d, q, v)
        mass = np.asarray(d.M).copy()
        mass = np.triu(mass)+np.triu(mass, 1).T
        bias = np.asarray(d.nle).copy()
        ag = pin.computeCentroidalMap(m, d, q).copy()
        dag = pin.computeCentroidalMapTimeVariation(m, d, q, v).copy()
        pin.forwardKinematics(m, d, q, v, np.zeros(m.nv))
        pin.computeJointJacobiansTimeVariation(m, d, q, v)
        pin.updateFramePlacements(m, d)
        pin.centerOfMass(m, d, q, v)
        frames = {}
        for name, fid in self.frame_ids.items():
            jac = pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED).copy()
            djac = pin.getFrameJacobianTimeVariation(m, d, fid, pin.LOCAL_WORLD_ALIGNED).copy()
            frames[name] = dict(position=d.oMf[fid].translation.copy(), rotation=d.oMf[fid].rotation.copy(),
                                J=jac, bias=djac@v, velocity=jac@v)
        # Independent body-sum momentum, rather than defining actual h as Ag@v.
        momentum = np.zeros(6)
        for jid in range(1, m.njoints):
            inertia, placement, velocity = m.inertias[jid], d.oMi[jid], d.v[jid]
            omega = placement.rotation@velocity.angular
            vc = placement.rotation@(velocity.linear+np.cross(velocity.angular, inertia.lever))
            position = placement.act(inertia.lever)
            linear = inertia.mass*vc
            momentum[:3] += linear
            momentum[3:] += placement.rotation@inertia.inertia@placement.rotation.T@omega + np.cross(position-d.com[0], linear)
        return dict(q=q.copy(), v=v.copy(), M=mass, bias=bias, Ag=ag, Ag_bias=dag@v,
                    com=d.com[0].copy(), com_velocity=d.vcom[0].copy(), momentum=momentum,
                    frames=frames, mass=self.mass, gravity=np.asarray(m.gravity.linear).copy())

    def reference(self, q, v):
        return self.state(q, v)
