"""Build a MuJoCo T1 from the same USD physics export used by FORCE2."""
import ast
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/booster/t1'
GEOMETRY = ROOT / 'eval_data/reference_walk_20261016/hdm_bk_settled_env_geometry.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(x):
    return ' '.join(format(float(v), '.17g') for v in np.atleast_1d(x))


def wxyz(r):
    return r.as_quat()[[3, 0, 1, 2]]


def actuator_limits():
    path = ROOT / 'source/whole_body_tracking/whole_body_tracking/robots/t1.py'
    tree = ast.parse(path.read_text())
    cfg = next(n.value for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'BOOSTER_T1_CFG' for t in n.targets))
    actuators = next(k.value for k in cfg.keywords if k.arg == 'actuators')
    result = {}
    names = json.loads((ASSET / 'wbc_acc_dynamics.json').read_text())['joints']
    for actuator in actuators.values:
        args = {k.arg: k.value for k in actuator.keywords}
        patterns = ast.literal_eval(args['joint_names_expr'])
        effort, speed = (ast.literal_eval(args[k]) for k in ('effort_limit_sim', 'velocity_limit_sim'))
        for joint in names:
            name = joint['name']
            if any(re.fullmatch(p, name) for p in patterns):
                def match(value):
                    return next(v for p, v in value.items() if re.fullmatch(p, name)) if isinstance(value, dict) else value
                result[name] = dict(effort=match(effort), speed=match(speed))
    assert len(result) == 23
    return result


def build_model(output):
    from pxr import Usd, UsdGeom
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    description = json.loads((ASSET / 'wbc_acc_dynamics.json').read_text())
    for name, digest in description['source_layers_sha256'].items():
        matches = list(ASSET.rglob(name))
        assert len(matches) == 1 and sha(matches[0]) == digest, f'Stale USD export: {name}'
    geometry = dict(np.load(GEOMETRY, allow_pickle=False))
    limits = dict(zip(geometry['joint_names'].tolist(), geometry['joint_limits']))
    motors = actuator_limits()
    root = ET.Element('mujoco', model='T1_FORCE2_DSMS')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', fusestatic='false')
    option = ET.SubElement(root, 'option', timestep='.01', gravity='0 0 -9.81',
                          integrator='implicitfast', cone='elliptic', iterations='100', tolerance='1e-10')
    ET.SubElement(option, 'flag', warmstart='disable')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'joint', damping='0', armature='0', frictionloss='0')
    ET.SubElement(default, 'geom', friction='.6 0 0', condim='3', solref='.02 1',
                  solimp='0 .95 .001 .5 2', margin='0', density='0')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .1', rgba='.85 .88 .9 1')
    elements = {}
    joint_order = []

    def add_body(name, parent, joint=None):
        attrs = dict(name=name)
        if joint:
            r0, r1 = (Rotation.from_quat(joint[k]) for k in ('quat0_xyzw', 'quat1_xyzw'))
            rest = r0 * r1.inv()
            attrs.update(pos=numbers(np.array(joint['pos0']) - rest.apply(joint['pos1'])), quat=numbers(wxyz(rest)))
        body = ET.SubElement(parent, 'body', **attrs)
        elements[name] = body
        b = description['bodies'][name]
        ET.SubElement(body, 'inertial', mass=numbers(b['mass']), pos=numbers(b['com']),
                      quat=numbers(wxyz(Rotation.from_quat(b['principal_axes_xyzw']))), diaginertia=numbers(b['inertia_diagonal']))
        if joint:
            name_j = joint['name']
            axis = Rotation.from_quat(joint['quat1_xyzw']).apply(np.eye(3)['XYZ'.index(joint['axis'])])
            ET.SubElement(body, 'joint', name=name_j, type='hinge', pos=numbers(joint['pos1']),
                          axis=numbers(axis), range=numbers(limits[name_j]))
            joint_order.append(name_j)
        else:
            ET.SubElement(body, 'freejoint', name='root_joint')
        for child in description['joints']:
            if child['parent'] == name:
                add_body(child['child'], body, child)

    add_body(description['root'], world)
    stage = Usd.Stage.Open(str(ASSET / 't1.usd'))
    cache = UsdGeom.XformCache()
    colliders = []
    for name, element in elements.items():
        body = stage.GetPrimAtPath('/T1/' + name)
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if '/collisions/' not in str(prim.GetPath()) or prim.GetTypeName() == 'Xform':
                continue
            matrix = np.array(cache.GetLocalToWorldTransform(prim) * inverse).T
            scale = np.linalg.norm(matrix[:3, :3], axis=0)
            rotation = matrix[:3, :3] / scale
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            attrs = dict(name=name + '_collision', pos=numbers(matrix[:3, 3]),
                         quat=numbers(wxyz(Rotation.from_matrix(rotation))), rgba='.28 .34 .40 1')
            if prim.IsA(UsdGeom.Cube):
                attrs.update(type='box', size=numbers(scale * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2))
            elif prim.IsA(UsdGeom.Sphere):
                np.testing.assert_allclose(scale, scale[0], atol=1e-7)
                attrs.update(type='sphere', size=numbers(scale[0] * UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
            elif prim.IsA(UsdGeom.Cylinder):
                c = UsdGeom.Cylinder(prim)
                assert c.GetAxisAttr().Get() == 'Z'
                np.testing.assert_allclose(scale[0], scale[1], atol=1e-7)
                attrs.update(type='cylinder', size=numbers([c.GetRadiusAttr().Get() * scale[0], c.GetHeightAttr().Get() * scale[2] / 2]))
            else:
                raise ValueError(f'Unsupported collider: {prim.GetPath()} {prim.GetTypeName()}')
            ET.SubElement(element, 'geom', **attrs)
            colliders.append(dict(source=str(prim.GetPath()), body=name, **attrs))
    actuator = ET.SubElement(root, 'actuator')
    for name in joint_order:
        effort = motors[name]['effort']
        kp = min(80., effort * 2.)
        kd = 2. * np.sqrt(kp * .05)
        # Native servos follow the upstream position-control example. Its torque
        # mode converts these to bounded normalized motors before optimization.
        ET.SubElement(actuator, 'general', name=name, joint=name, gainprm=numbers(kp),
                      biastype='affine', biasprm=numbers([0., -kp, -kd]),
                      ctrlrange=numbers(limits[name]), forcerange=numbers([-effort, effort]))
    ET.indent(root)
    path = output / 't1_force2.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    m = mujoco.MjModel.from_xml_path(str(path))
    assert (m.nq, m.nv, m.nu) == (30, 29, 23)
    np.testing.assert_allclose(m.body_mass.sum(), sum(b['mass'] for b in description['bodies'].values()), atol=1e-12)
    report = dict(model=str(path.resolve()), model_sha256=sha(path), mass_kg=float(m.body_mass.sum()),
                  source_dynamics_sha256=sha(ASSET / 'wbc_acc_dynamics.json'), source_usd_layers=description['source_layers_sha256'],
                  geometry_sha256=sha(GEOMETRY), colliders=colliders, motors=motors,
                  friction=.6, zero_passive_actuator_dynamics=True,
                  limitation='MuJoCo contact dynamics differ from PhysX; this is not a PhysX or hardware feasibility certificate.')
    (output / 'model.json').write_text(json.dumps(report, indent=2) + '\n')
    return path


def load_reference(path, model):
    z = dict(np.load(path, allow_pickle=False))
    fps = float(z['fps'].item())
    root = z['body_names'].tolist().index('Trunk')
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    order = [z['joint_names'].tolist().index(n) for n in joints]
    q = np.c_[z['body_pos_w'][:, root], z['body_quat_w'][:, root], z['joint_pos'][:, order]]
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    velocity = np.zeros((len(q), model.nv))
    for i in range(len(q)):
        a, b = max(0, i - 1), min(len(q) - 1, i + 1)
        mujoco.mj_differentiatePos(model, velocity[i], (b - a) / fps, q[a], q[b])
    return z, np.c_[q, velocity], fps


def validate_references(directory, model_path):
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    manifest = json.loads((Path(directory) / 'manifest.json').read_text())
    max_pos = max_rot = 0.
    frame_count = 0
    for row in manifest['motions']:
        path = Path(directory) / row['file']
        assert sha(path) == row['sha256']
        z, x, fps = load_reference(path, m)
        assert fps == 50 and len(x) == row['frames']
        bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in z['body_names'].tolist()]
        assert min(bids) > 0
        for i, state in enumerate(x):
            d.qpos[:] = state[:m.nq]
            mujoco.mj_kinematics(m, d)
            max_pos = max(max_pos, float(np.abs(d.xpos[bids] - z['body_pos_w'][i]).max()))
            actual = Rotation.from_quat(d.xquat[bids][:, [1, 2, 3, 0]]).as_matrix()
            reference = Rotation.from_quat(z['body_quat_w'][i][:, [1, 2, 3, 0]]).as_matrix()
            max_rot = max(max_rot, float(np.abs(actual - reference).max()))
        frame_count += len(x)
    assert max_pos < 2e-6 and max_rot < 2e-6, (max_pos, max_rot)
    return dict(clips=len(manifest['motions']), frames=frame_count, max_fk_position_error_m=max_pos,
                max_fk_rotation_matrix_error=max_rot, manifest_sha256=sha(Path(directory) / 'manifest.json'), passed=True)
