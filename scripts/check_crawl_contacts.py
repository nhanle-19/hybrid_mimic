"""Independently audit saved crawl labels against the composed T1 USD asset.

Requires numpy, scipy, mujoco and Isaac Sim's pxr modules (no simulator launch).
Reads poses/labels only; writes label_audit.json, label_audit_frames.csv and
label_audit_geometry.npz beside them. No motion, labels or video are changed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "source/whole_body_tracking/whole_body_tracking/assets/booster/t1"
MESH_REGIONS = {
    "Trunk": "torso", "Waist": "torso", "H1": "head", "H2": "head",
    "AL1": "left_upper_arm", "AL2": "left_upper_arm", "AL3": "left_upper_arm",
    "left_hand_link": "left_forearm_hand",
    "AR1": "right_upper_arm", "AR2": "right_upper_arm", "AR3": "right_upper_arm",
    "right_hand_link": "right_forearm_hand",
    "Hip_Pitch_Left": "left_thigh", "Hip_Roll_Left": "left_thigh", "Hip_Yaw_Left": "left_thigh",
    "Shank_Left": "left_shin", "Ankle_Cross_Left": "left_foot", "left_foot_link": "left_foot",
    "Hip_Pitch_Right": "right_thigh", "Hip_Roll_Right": "right_thigh", "Hip_Yaw_Right": "right_thigh",
    "Shank_Right": "right_shin", "Ankle_Cross_Right": "right_foot", "right_foot_link": "right_foot",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def count_histogram(values):
    return {str(int(v)): int((values == v).sum()) for v in np.unique(values)}


def mismatch_count(first, second):
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} != {second.shape}")
    return int(np.count_nonzero(first != second))


def usd_geometry_and_fk(motion, geometry):
    """Read live USD primitives/joints and use SciPy FK, not the label generator."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(ASSET / "t1.usd"))
    cache = UsdGeom.XformCache()
    layers = {Path(layer.realPath).name: sha256(Path(layer.realPath))
              for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    if layers != geometry["source_layers_sha256"]:
        raise ValueError("Contact-site asset hashes do not match the composed USD asset")
    enabled = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            enabled.append(prim)
    actual_paths = set()
    for parent in enabled:
        for prim in Usd.PrimRange(parent, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Gprim):
                actual_paths.add(str(prim.GetPath()))
    expected_paths = {s["collider_path"] for s in geometry["surfaces"]}
    if actual_paths != expected_paths:
        raise ValueError(f"Collider coverage mismatch: {actual_paths ^ expected_paths}")

    n = len(motion["root_pos"])
    positions = {"Trunk": motion["root_pos"].copy()}
    rotations = {"Trunk": Rotation.from_quat(motion["root_rot"]).as_matrix()}
    joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    joint_names = list(motion["joint_names"])
    if set(j.GetPrim().GetName() for j in joints) != set(joint_names):
        raise ValueError("Motion joint names disagree with the USD")

    def quat(value):
        return Rotation.from_quat([*value.GetImaginary(), value.GetReal()]).as_matrix()

    remaining = joints.copy()
    while remaining:
        progress = False
        for joint in remaining.copy():
            parent, child = joint.GetBody0Rel().GetTargets()[0].name, joint.GetBody1Rel().GetTargets()[0].name
            if parent not in positions:
                continue
            angle = motion["dof_pos"][:, joint_names.index(joint.GetPrim().GetName())]
            axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
            rot = rotations[parent] @ quat(joint.GetLocalRot0Attr().Get())
            rot = rot @ Rotation.from_rotvec(angle[:, None] * axis).as_matrix() @ quat(joint.GetLocalRot1Attr().Get()).T
            pos = positions[parent] + np.einsum("nij,j->ni", rotations[parent], joint.GetLocalPos0Attr().Get())
            pos -= np.einsum("nij,j->ni", rot, joint.GetLocalPos1Attr().Get())
            positions[child], rotations[child] = pos, rot
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ValueError("USD joint tree is disconnected")

    gaps, worlds, locals_, surface_ids, names, bodies = [], [], [], [], [], []
    surface_residual = 0.0
    for s, saved in enumerate(geometry["surfaces"]):
        prim = stage.GetPrimAtPath(saved["collider_path"])
        body = stage.GetPrimAtPath("/T1/" + saved["body"])
        transform = np.array(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()).T
        np.testing.assert_allclose(transform, saved["transform_to_body"], rtol=0, atol=1e-12)
        rot = rotations[saved["body"]] @ transform[:3, :3]
        centre = positions[saved["body"]] + np.einsum("nij,j->ni", rotations[saved["body"]], transform[:3, 3])
        row = rot[:, 2]
        shape = prim.GetTypeName()
        if shape != saved["shape"]:
            raise ValueError("Contact shape type mismatch")
        points = np.asarray(saved["points_body"])
        canonical = (points - transform[:3, 3]) @ np.linalg.inv(transform[:3, :3]).T
        if shape == "Cube":
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            np.testing.assert_allclose(half, saved["half_size"], rtol=0, atol=1e-12)
            gap = centre[:, 2] - half * np.abs(row).sum(1)
            residual = np.abs(np.abs(canonical).max(1) - half).max()
        elif shape == "Cylinder":
            cylinder = UsdGeom.Cylinder(prim)
            radius, height = float(cylinder.GetRadiusAttr().Get()), float(cylinder.GetHeightAttr().Get())
            np.testing.assert_allclose([radius, height], [saved["radius"], saved["height"]], rtol=0, atol=1e-12)
            axis = "XYZ".index(cylinder.GetAxisAttr().Get())
            radial_axes = [k for k in range(3) if k != axis]
            gap = centre[:, 2] - radius * np.linalg.norm(row[:, radial_axes], axis=1) - height / 2 * np.abs(row[:, axis])
            radial = np.linalg.norm(canonical[:, radial_axes], axis=1)
            axial = np.abs(canonical[:, axis])
            residual = max(float(np.maximum(radial - radius, 0).max()),
                           float(np.maximum(axial - height / 2, 0).max()),
                           float(np.minimum(np.abs(radial - radius), np.abs(axial - height / 2)).max()))
        elif shape == "Sphere":
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            np.testing.assert_allclose(radius, saved["radius"], rtol=0, atol=1e-12)
            gap = centre[:, 2] - radius * np.linalg.norm(row, axis=1)
            gaps.append(gap)
            continue  # The head is checked analytically, not a support-site sample.
        else:
            raise ValueError(f"Unsupported collider: {shape}")
        surface_residual = max(surface_residual, float(residual))
        gaps.append(gap)
        worlds.append(np.einsum("nij,pj->npi", rotations[saved["body"]], points) + positions[saved["body"]][:, None])
        locals_.extend(points)
        surface_ids.extend([s] * len(points))
        names.extend(saved["name"] + "/" + name for name in saved["point_names"])
        bodies.extend([saved["body"]] * len(points))
    return dict(gaps=np.stack(gaps, 1), world=np.concatenate(worlds, 1), local=np.asarray(locals_),
                surface=np.asarray(surface_ids), names=np.asarray(names), bodies=np.asarray(bodies),
                positions=positions, rotations=rotations, point_surface_error_m=surface_residual,
                asset_layers_sha256=layers, frames=n)


def audit(case, motion_path):
    motion, labels = read_npz(motion_path), read_npz(case / "contact_geometry.npz")
    if int(labels.get("annotation_version", 1)) >= 3:
        # The repaired reference labels the full visible surface. Recompute all
        # labels and independently check the USD kinematics/clearance as well.
        from finalize_crawl_repair import RAW, validate_and_label
        provenance_path = case / "reference_validation.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        baseline = ROOT / provenance.get("source_motion", str(RAW))
        if provenance.get("source_motion_sha256") and sha256(baseline) != provenance["source_motion_sha256"]:
            raise ValueError("Reference validation baseline hash mismatch")
        _, expected, validation = validate_and_label(
            motion_path, baseline, allow_path_smoothing=provenance.get("allow_path_smoothing", False))
        failures = {}
        for key, value in expected.items():
            if key not in labels or labels[key].shape != value.shape:
                failures[key] = "missing or wrong shape"
            elif np.issubdtype(value.dtype, np.floating):
                if not np.allclose(labels[key], value, rtol=0, atol=1e-10):
                    failures[key] = "values disagree"
            elif not np.array_equal(labels[key], value):
                failures[key] = "values disagree"
        with (case / "contact_labels.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        csv_errors = int(len(rows) != len(expected["time_s"]))
        for i, row in enumerate(rows[:len(expected["time_s"])]):
            csv_errors += int(int(row["frame"]) != i or abs(float(row["time_s"]) - expected["time_s"][i]) > 5.1e-7)
            for j, name in enumerate(expected["surface_names"]):
                csv_errors += int(row[str(name) + "_mode"] != expected["surface_mode"][i, j])
                csv_errors += int(int(row[str(name) + "_points"]) != expected["surface_point_count"][i, j])
                csv_errors += int(abs(float(row[str(name) + "_clearance_mm"]) / 1000 - expected["surface_clearance_m"][i, j]) > 1e-9)
        passed = not failures and csv_errors == 0
        report = dict(geometric_contact_labels_pass=passed, annotation_version=3,
                      full_body_geometric_contact_labels_verified=passed,
                      physical_execution_verified=False, frames=len(expected["time_s"]),
                      mismatched_arrays=failures, csv_mismatched_fields=csv_errors,
                      contact_frames_by_region=validation["contact_frames_by_region"],
                      required_support_gap_range_mm=validation["required_support_gap_range_mm"],
                      independent_usd_fk_max_error=validation["independent_usd_fk_max_error"],
                      baseline_motion=display_path(baseline), baseline_sha256=sha256(baseline),
                      validation_scope="Full visible-mesh contact labels, frame alignment, all-point coverage and sliding; separate USD floor clearance and kinematics. Not a dynamics rollout.",
                      provenance={display_path(p): sha256(p) for p in (motion_path, case / "contact_geometry.npz", case / "contact_labels.csv")})
        (case / "label_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    geometry = json.loads((case / "contact_sites.json").read_text())
    if geometry["ground_height_m"] != 0:
        raise ValueError("This audit expects the reference floor at world z=0")
    verified = usd_geometry_and_fk(motion, geometry)
    n, fps = verified["frames"], float(motion["fps"])
    names = [s["name"] for s in geometry["surfaces"]]
    np.testing.assert_array_equal(labels["surface_names"], names)
    np.testing.assert_allclose(labels["time_s"], np.arange(n) / fps, atol=1e-12, rtol=0)
    np.testing.assert_allclose(labels["fps"], fps, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(labels["point_names"], verified["names"])
    np.testing.assert_array_equal(labels["point_surface_index"], verified["surface"])
    np.testing.assert_array_equal(labels["body_names"][labels["point_body_index"]], verified["bodies"])
    np.testing.assert_allclose(labels["point_position_body"], verified["local"], atol=1e-12, rtol=0)
    body_pos = np.stack([verified["positions"][str(b)] for b in labels["body_names"]], 1)
    body_rot = np.stack([verified["rotations"][str(b)] for b in labels["body_names"]], 1)
    errors = dict(
        stored_body_position_max_error_m=float(np.max(abs(labels["body_position_w"] - body_pos))),
        stored_body_rotation_max_error=float(np.max(abs(labels["body_rotation_w"] - body_rot))),
        stored_surface_gap_max_error_m=float(np.max(abs(labels["surface_clearance_m"] - verified["gaps"]))),
        sampled_point_surface_max_error_m=verified["point_surface_error_m"],
    )
    tolerance, patch = geometry["contact_tolerance_m"], geometry["patch_tolerance_m"]
    world, surface = verified["world"], verified["surface"]
    expected = (world[:, :, 2] <= tolerance) & (world[:, :, 2] <= verified["gaps"][:, surface] + patch)
    frames, points = np.nonzero(expected)
    actual_frame, actual_point = labels["contact_frame"], labels["contact_point_index"]
    recorded = np.zeros(expected.shape, dtype=np.int32)
    np.add.at(recorded, (actual_frame, actual_point), 1)
    expected_counts = np.stack([expected[:, surface == s].sum(1) for s in range(len(names))], 1)
    expected_active = expected_counts > 0
    continuous_active = verified["gaps"] <= tolerance
    sparse_values_aligned = np.array_equal(frames, actual_frame) and np.array_equal(points, actual_point)
    if not sparse_values_aligned:
        # Still report omissions/false positives, but don't compare misaligned coordinates.
        coordinate_error = None
    else:
        coordinate_error = float(np.max(abs(world[frames, points] - labels["contact_position_w"])))
        errors["contact_position_max_error_m"] = coordinate_error
        errors["contact_gap_max_error_m"] = float(np.max(abs(world[frames, points, 2] - labels["contact_gap_m"])))
    errors["contact_normal_max_error"] = float(np.max(abs(labels["contact_normal_w"] - [0., 0., 1.])))

    # Fixed material-site velocities; the identity of the lowest site can change.
    velocity = np.gradient(world, 1 / fps, axis=0)
    speed = np.linalg.norm(velocity[actual_frame, actual_point, :2], axis=1)
    point_mode = np.where(speed > .015, "sliding_contact", "contact")
    surface_speed = np.zeros(expected_counts.shape)
    np.maximum.at(surface_speed, (actual_frame, surface[actual_point]), speed)
    surface_mode = np.where(expected_active, np.where(surface_speed > .015, "sliding_contact", "contact"), "non_contact")
    errors["material_velocity_max_error_m_s"] = float(np.max(abs(velocity[actual_frame, actual_point] - labels["contact_material_velocity_w"])))
    errors["tangent_speed_max_error_m_s"] = float(np.max(abs(speed - labels["contact_tangent_speed_m_s"])))
    errors["surface_speed_max_error_m_s"] = float(np.max(abs(surface_speed - labels["surface_tangent_speed_m_s"])))

    checks = dict(
        extra_labeled_points=int(((recorded > 0) & ~expected).sum()),
        missing_labeled_points=int((expected & (recorded == 0)).sum()),
        duplicate_point_records=int((recorded > 1).sum()),
        wrong_point_surface_index=mismatch_count(labels["contact_surface_index"], surface[actual_point]),
        wrong_surface_point_count=mismatch_count(labels["surface_point_count"], expected_counts),
        wrong_surface_active=mismatch_count(labels["surface_active"], expected_active),
        continuous_contacts_missing_from_labels=int((continuous_active & ~labels["surface_active"]).sum()),
        false_surface_contacts=int((labels["surface_active"] & ~continuous_active).sum()),
        wrong_frame_offsets=mismatch_count(labels["frame_offsets"], np.r_[0, np.cumsum(expected.sum(1))]),
        wrong_contact_modes=mismatch_count(labels["contact_mode"], point_mode),
        wrong_surface_modes=mismatch_count(labels["surface_mode"], surface_mode),
        sparse_records_out_of_order=int(not sparse_values_aligned),
    )
    # Review CSV is another artifact and can drift even when the NPZ is correct.
    csv_errors = 0
    used = np.flatnonzero(expected_active.any(0))
    with (case / "contact_labels.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n:
        raise ValueError("Contact CSV does not cover the whole motion")
    for i, row in enumerate(rows):
        csv_errors += int(int(row["frame"]) != i or abs(float(row["time_s"]) - i / fps) > 5.1e-7)
        for s in used:
            prefix = names[s]
            if any(prefix + suffix not in row for suffix in ("_mode", "_points", "_clearance_mm")):
                csv_errors += 1
                continue
            csv_errors += int(row[prefix + "_mode"] != surface_mode[i, s])
            csv_errors += int(int(row[prefix + "_points"]) != expected_counts[i, s])
            csv_errors += int(abs(float(row[prefix + "_clearance_mm"]) / 1000 - verified["gaps"][i, s]) > 1e-9)
    checks["csv_mismatched_fields"] = csv_errors

    # A second engine checks every body transform and the visible mesh, not just
    # the points selected by the labels. All mesh vertices are considered.
    model = mj.MjModel.from_xml_path(str(ROOT / "GMR/assets/booster_t1/T1_serial.xml"))
    data = mj.MjData(model)
    joint_order = [list(motion["joint_names"]).index(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)) for j in range(1, model.njnt)]
    qpos = np.c_[motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"][:, joint_order]]
    mj_body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, str(b)) for b in labels["body_names"]]
    if min(mj_body_ids) < 0:
        raise ValueError("MuJoCo model is missing a USD body")
    meshes = []
    for g in range(model.ngeom):
        if not model.geom_bodyid[g] or model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
        group = names.index(MESH_REGIONS[body])
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        meshes.append((g, group, model.mesh_vert[start:start + count].astype(float)))
    mesh_gaps = np.full((n, len(names)), np.inf)
    max_pos_error = max_rot_error = 0.0
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        max_pos_error = max(max_pos_error, float(abs(data.xpos[mj_body_ids] - body_pos[i]).max()))
        max_rot_error = max(max_rot_error, float(abs(data.xmat[mj_body_ids].reshape(-1, 3, 3) - body_rot[i]).max()))
        for g, group, vertices in meshes:
            gap = float(np.min(vertices @ data.geom_xmat[g].reshape(3, 3)[2]) + data.geom_xpos[g, 2])
            mesh_gaps[i, group] = min(mesh_gaps[i, group], gap)
    if not np.isfinite(mesh_gaps).all():
        raise ValueError("Incomplete visible-mesh coverage")
    errors["independent_mujoco_body_position_error_m"] = max_pos_error
    errors["independent_mujoco_body_rotation_error"] = max_rot_error
    mesh_active = mesh_gaps <= tolerance
    mesh_omitted = mesh_active & ~labels["surface_active"]
    # Speed errors include float32 quantization before the original differentiation.
    error_limits = {key: (2e-5 if "speed" in key or "velocity" in key else 2e-6) for key in errors}
    passed = not any(checks.values()) and all(value <= error_limits[key] for key, value in errors.items())
    surface_rows = {
        name: dict(labeled_frames=int(labels["surface_active"][:, s].sum()),
                   independently_detected_usd_frames=int(continuous_active[:, s].sum()),
                   visible_mesh_within_1mm_frames=int(mesh_active[:, s].sum()),
                   visible_mesh_contact_without_usd_label_frames=int(mesh_omitted[:, s].sum()),
                   min_usd_gap_mm=float(1000 * verified["gaps"][:, s].min()),
                   min_mesh_gap_mm=float(1000 * mesh_gaps[:, s].min()))
        for s, name in enumerate(names)
    }
    report = dict(
        usd_geometric_labels_pass=passed, frames=n, fps=fps, duration_s=n / fps,
        full_body_contact_labels_verified=False,
        validation_scope="USD collider-label consistency only. This does not validate the reference's intended crawl supports; see check_crawl_support.py.",
        checked_collider_count=len(names), checked_material_sites=len(surface), checked_visual_mesh_count=len(meshes),
        contact_tolerance_mm=1000 * tolerance, patch_tolerance_mm=1000 * patch,
        sliding_threshold_m_s=.015, checks=checks, numerical_errors=errors, numerical_error_limits=error_limits,
        usd_contact_region_count=count_histogram(continuous_active.sum(1)),
        visual_mesh_contact_region_count=count_histogram(mesh_active.sum(1)),
        frames_with_unlabeled_visible_mesh_contact=int(mesh_omitted.any(1).sum()), regions=surface_rows,
        sampled_point_count_range=[int(expected.sum(1).min()), int(expected.sum(1).max())],
        contact_normal_note="World-up floor normal. Contact coordinates remain on the robot surface within 1 mm of the floor.",
        meaning="Labels denote sampled USD collider/floor proximity and material-point sliding. They do not denote all visible-mesh touches or measured supporting forces.",
        provenance={display_path(p): sha256(p) for p in (motion_path, case / "contact_geometry.npz", case / "contact_labels.csv", case / "contact_sites.json")},
        asset_layers_sha256=verified["asset_layers_sha256"],
        source_motion_labels_and_video_modified=False,
    )
    samples = []
    for i in (0, 100, n // 2, 325, 450, n - 1):
        samples.append(dict(frame=i, time_s=i / fps,
                            usd_contacts=[names[s] for s in np.flatnonzero(continuous_active[i])],
                            mesh_contacts=[names[s] for s in np.flatnonzero(mesh_active[i])],
                            usd_gap_mm={name: round(float(1000 * verified["gaps"][i, s]), 4) for s, name in enumerate(names)},
                            mesh_gap_mm={name: round(float(1000 * mesh_gaps[i, s]), 4) for s, name in enumerate(names)}))
    report["examples"] = samples
    np.savez_compressed(case / "label_audit_geometry.npz", time_s=np.arange(n) / fps,
                        surface_names=np.asarray(names), usd_gap_m=verified["gaps"], usd_active=continuous_active,
                        mesh_gap_m=mesh_gaps, mesh_active=mesh_active, stored_active=labels["surface_active"])
    with (case / "label_audit_frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "region", "saved_label", "usd_gap_mm", "usd_contact", "visible_mesh_gap_mm", "visible_mesh_within_1mm"])
        for i in range(n):
            for s, name in enumerate(names):
                writer.writerow([i, f"{i / fps:.6f}", name, labels["surface_mode"][i, s],
                                 f"{1000 * verified['gaps'][i, s]:.6f}", int(continuous_active[i, s]),
                                 f"{1000 * mesh_gaps[i, s]:.6f}", int(mesh_active[i, s])])
    (case / "label_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "eval_data/reference_crawl_forward")
    parser.add_argument("--motion", type=Path, default=ROOT / "retargeted_motion/a11_crawl_forward_female1_t1.npz")
    args = parser.parse_args()
    report = audit(args.case.resolve(), args.motion.resolve())
    print(json.dumps(report, indent=2))
    if not report.get("geometric_contact_labels_pass", report.get("usd_geometric_labels_pass", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
