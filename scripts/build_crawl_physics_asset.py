"""Build a crawl USD overlay with contact colliders matching the reviewed mesh.

The original robot asset is read-only. Each body/region gets a <=64-vertex
convex collider. All reference-frame floor-extreme vertices are retained, so
the 599-frame contact schedule is unchanged by the conservative simplification.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'source/whole_body_tracking/whole_body_tracking/assets/booster/t1/t1.usd'
ANNOTATIONS = ROOT/'eval_data/reference_crawl_forward/contact_geometry.npz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collider_vertices(points, floor_directions, limit=64):
    required = set(np.argmin(points@floor_directions.T, axis=0).tolist())
    required.update(np.argmin(points, axis=0).tolist())
    required.update(np.argmax(points, axis=0).tolist())
    hull = ConvexHull(points)
    if len(required) > limit:
        raise ValueError(f'{len(required)} reference extrema exceed the collider limit {limit}')
    selected = sorted(required)
    candidates = hull.vertices
    while len(selected) < min(limit, len(candidates)):
        distances = ((points[candidates, None]-points[selected][None])**2).sum(-1).min(-1)
        index = int(candidates[np.argmax(distances)])
        if index in selected:
            break
        selected.append(index)
    return np.asarray(selected)


def build(output, annotations=ANNOTATIONS):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(annotations, allow_pickle=False) as data:
        labels = dict(data)
    asset = output/'t1_crawl.usda'
    layer = Sdf.Layer.FindOrOpen(str(asset)) if asset.exists() else Sdf.Layer.CreateNew(str(asset))
    layer.Clear()
    layer.subLayerPaths = [str(SOURCE)]
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath('/T1'))
    original_colliders = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    for path in original_colliders:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    material = UsdShade.Material.Define(stage, '/T1/crawlMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.6)
    api.CreateDynamicFrictionAttr(.6)
    api.CreateRestitutionAttr(0.)
    saved, entries = [], []
    for body in np.unique(labels['point_body_index']):
        name = str(labels['body_names'][body])
        for region in np.unique(labels['point_surface_index'][labels['point_body_index'] == body]):
            ids = np.flatnonzero((labels['point_body_index'] == body) & (labels['point_surface_index'] == region))
            local = labels['point_position_body'][ids]
            selected = collider_vertices(local, labels['body_rotation_w'][:, body, 2, :])
            ids = ids[selected]
            points = labels['point_position_body'][ids].astype(np.float32)
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            # Outward winding, matching the corresponding hull plane normal.
            normals = np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]])
            flip = (normals*hull.equations[:, :3]).sum(-1) < 0
            faces[flip] = faces[flip][:, [0, 2, 1]]
            region_name = str(labels['surface_names'][region])
            path = f'/T1/{name}/crawl_{region_name}'
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(points.tolist())
            mesh.CreateFaceVertexCountsAttr([3]*len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            mesh.GetPrim().AddAppliedSchema('PhysxConvexHullCollisionAPI')
            mesh.GetPrim().CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int).Set(64)
            mesh.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
            mesh.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
            mesh.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose='physics')
            # Keep explicit inertias: collision approximation must not alter dynamics.
            assert stage.GetPrimAtPath(f'/T1/{name}').HasAPI(UsdPhysics.MassAPI)
            saved.extend(ids.tolist())
            entries.append(dict(body=name, region=region_name, path=path, vertex_count=len(points),
                                points_body=points.tolist(), planes_body=hull.equations.tolist()))
    # Joint housings up to two edges apart overlap in the original visual model.
    # Preserve collisions for all non-neighbor pairs, as in the geometry audit.
    dynamics = json.loads((SOURCE.parent/'wbc_acc_dynamics.json').read_text())
    adjacency = {body: set() for body in dynamics['bodies']}
    for joint in dynamics['joints']:
        adjacency[joint['parent']].add(joint['child'])
        adjacency[joint['child']].add(joint['parent'])
    for body, neighbors in adjacency.items():
        excluded = set(neighbors)
        for neighbor in neighbors:
            excluded.update(adjacency[neighbor])
        excluded.discard(body)
        relation = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath('/T1/'+body)).CreateFilteredPairsRel()
        relation.SetTargets(['/T1/'+name for name in sorted(excluded)])
    stage.GetRootLayer().Save()
    ids = np.asarray(saved)
    point_keys = ('point_position_body', 'point_body_index', 'point_surface_index', 'point_geom_index', 'point_mesh_vertex_index')
    geometry = {key: labels[key][ids] for key in point_keys}
    geometry['point_position_body'] = geometry['point_position_body'].astype(np.float32).astype(float)
    geometry.update(annotation_version=np.asarray(3), body_names=labels['body_names'], surface_names=labels['surface_names'],
                    geometry_basis=np.asarray('Authored convex collision vertices'), point_source_index=ids)
    gap = np.full_like(labels['surface_clearance_m'], np.inf)
    for entry in entries:
        body = labels['body_names'].tolist().index(entry['body'])
        region = labels['surface_names'].tolist().index(entry['region'])
        low = (np.asarray(entry['points_body'])@labels['body_rotation_w'][:, body, 2, :].T).min(0)
        gap[:, region] = np.minimum(gap[:, region], low+labels['body_position_w'][:, body, 2])
    error = float(np.max(np.abs(gap-labels['surface_clearance_m'])))
    if error > 1e-6 or not np.array_equal(gap <= .001, labels['surface_active']):
        raise ValueError(f'Physics collider simplification changes reference contact labels: {error}')
    geometry_path = output/'collision_geometry.npz'
    np.savez_compressed(geometry_path, **geometry)
    manifest = dict(source_asset=str(SOURCE), source_asset_sha256=digest(SOURCE), source_geometry_sha256=digest(annotations),
                    asset=str(asset), asset_sha256=digest(asset), geometry=str(geometry_path), geometry_sha256=digest(geometry_path),
                    collider_count=len(entries), entries=entries, reference_clearance_error_m=error,
                    reference_label_mismatches=0, friction=.6, contact_offset_m=.001, rest_offset_m=0.,
                    approximation='Conservative convex subsets with every reference floor-extreme vertex retained; <=64 vertices per collider.',
                    self_collision='Enabled except body pairs separated by <=2 joints.')
    (output/'collision_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'entries'}, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'eval_data/reference_crawl_forward/physics_asset')
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    args = parser.parse_args()
    build(args.output, args.annotations)
