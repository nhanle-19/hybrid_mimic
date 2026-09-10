"""Export the T1 USD rigid-body tree for the independent Atlas dynamics model.

Run with Isaac Sim's pxr modules available. No simulator launch is required.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from pxr import Usd, UsdPhysics
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usd', type=Path, default=root/'source/whole_body_tracking/whole_body_tracking/assets/booster/t1/t1.usd')
    parser.add_argument('--output', type=Path, default=root/'source/whole_body_tracking/whole_body_tracking/assets/booster/t1/atlas_dynamics.json')
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.usd))
    def quat(q):
        return [*map(float, q.GetImaginary()), float(q.GetReal())]
    bodies, joints = {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            bodies[prim.GetName()] = dict(mass=float(mass.GetMassAttr().Get()),
                com=list(mass.GetCenterOfMassAttr().Get()),
                inertia_diagonal=list(mass.GetDiagonalInertiaAttr().Get()),
                principal_axes_xyzw=quat(mass.GetPrincipalAxesAttr().Get()))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            joints.append(dict(name=prim.GetName(), parent=j.GetBody0Rel().GetTargets()[0].name,
                child=j.GetBody1Rel().GetTargets()[0].name, axis=j.GetAxisAttr().Get(),
                pos0=list(j.GetLocalPos0Attr().Get()), quat0_xyzw=quat(j.GetLocalRot0Attr().Get()),
                pos1=list(j.GetLocalPos1Attr().Get()), quat1_xyzw=quat(j.GetLocalRot1Attr().Get())))
    roots = set(bodies)-{j['child'] for j in joints}
    if roots != {'Trunk'} or len(joints) != 23 or len(bodies) != 24:
        raise ValueError('Expected the 23-DOF T1 rigid-body tree')
    # Hash every contributing layer, not only the small top-level composition file.
    layers = {Path(l.realPath).name: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
              for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}
    args.output.write_text(json.dumps(dict(root='Trunk', source_layers_sha256=layers,
                                         bodies=bodies, joints=joints), indent=2)+'\n')
    print(f'Exported {len(bodies)} bodies and {len(joints)} joints to {args.output}')


if __name__ == '__main__':
    main()
