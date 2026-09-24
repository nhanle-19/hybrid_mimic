"""Export existing foot box colliders; no hand-placed support locations."""
import hashlib
import json
from pathlib import Path
from itertools import product
from pxr import Usd, UsdGeom, Gf

root = Path(__file__).resolve().parents[1]
asset = root/'source/whole_body_tracking/whole_body_tracking/assets/booster/t1'
stage = Usd.Stage.Open(str(asset/'t1.usd'))
cache = UsdGeom.XformCache()
result = {'source_layers_sha256': {Path(l.realPath).name: hashlib.sha256(Path(l.realPath).read_bytes()).hexdigest()
          for l in stage.GetUsedLayers() if l.realPath and Path(l.realPath).is_file()}, 'bodies': {}}
for foot in ('left_foot_link', 'right_foot_link'):
    body = stage.GetPrimAtPath('/T1/'+foot)
    cubes = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
             if '/collisions/' in str(p.GetPath()) and p.IsA(UsdGeom.Cube)]
    if len(cubes) != 1:
        raise ValueError('Expected one box collider per foot; revalidate geometry for this asset')
    cube = cubes[0]
    transform = cache.GetLocalToWorldTransform(cube)*cache.GetLocalToWorldTransform(body).GetInverse()
    half = UsdGeom.Cube(cube).GetSizeAttr().Get()/2
    vertices = [list(transform.Transform(Gf.Vec3d(*(half*x for x in signs)))) for signs in product((-1, 1), repeat=3)]
    # Existing four-ray-per-corner representation, using the actual bottom face.
    sole = [v for v in vertices if abs(v[2]-min(p[2] for p in vertices)) < 1e-7]
    if len(sole) != 4:
        raise ValueError('Collider bottom face is not aligned with the foot frame')
    result['bodies'][foot] = {'collider_path': str(cube.GetPath()), 'vertices': vertices, 'sole_vertices': sole}
path = asset/'wbc_acc_contacts.json'
path.write_text(json.dumps(result, indent=2)+'\n')
print(path)
