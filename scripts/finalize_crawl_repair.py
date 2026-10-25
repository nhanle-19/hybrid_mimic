"""Validate a repaired crawl and build a staged motion, contact data and video.

Requires Isaac Sim's pxr modules for the independent USD kinematic check.
This writes a new output directory and never replaces the active reference.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from check_crawl_contacts import usd_geometry_and_fk
from repair_crawl_reference import Geometry, MODEL, RAW, ROOT, SUPPORTS


REGIONS = ("torso", "head", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
           "left_hand", "right_hand", "left_thigh", "right_thigh", "left_knee", "right_knee",
           "left_shin", "right_shin", "left_foot", "right_foot")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_definitions(model):
    """Every mesh vertex has one body and one anatomical region, without gaps."""
    local_points, body_ids, surface_ids, geom_ids, vertex_ids = [], [], [], [], []
    for g in range(model.ngeom):
        if not model.geom_bodyid[g] or model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        body_id = int(model.geom_bodyid[g])
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count].astype(float)
        r = Rotation.from_quat(model.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
        local = vertices @ r.T + model.geom_pos[g]
        if body in ("Trunk", "Waist"):
            region = np.full(count, REGIONS.index("torso"))
        elif body in ("H1", "H2"):
            region = np.full(count, REGIONS.index("head"))
        elif body.startswith(("AL", "AR")):
            region = np.full(count, REGIONS.index(("left" if body.startswith("AL") else "right") + "_upper_arm"))
        else:
            word = "left" if "Left" in body or body.startswith("left") else "right"
            side = word.title()
            if body.endswith("hand_link"):
                distal = (1 if word == "left" else -1) * local[:, 1] >= .180
                region = np.where(distal, REGIONS.index(word + "_hand"), REGIONS.index(word + "_forearm"))
            elif body.startswith("Hip_Yaw") or body.startswith("Shank"):
                child = "Shank_" + side if body.startswith("Hip_Yaw") else "Ankle_Cross_" + side
                axis = model.body_pos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, child)]
                along = local @ axis / np.dot(axis, axis)
                knee = along >= .5 if body.startswith("Hip_Yaw") else along <= .5
                other = "_thigh" if body.startswith("Hip_Yaw") else "_shin"
                region = np.where(knee, REGIONS.index(word + "_knee"), REGIONS.index(word + other))
            elif body.startswith("Hip_"):
                region = np.full(count, REGIONS.index(word + "_thigh"))
            elif body.startswith("Ankle_Cross") or body.endswith("foot_link"):
                region = np.full(count, REGIONS.index(word + "_foot"))
            else:
                raise ValueError(f"Unassigned visual body: {body}")
        local_points.append(local)
        body_ids.extend([body_id] * count)
        surface_ids.extend(region)
        geom_ids.extend([g] * count)
        vertex_ids.extend(range(count))
    return dict(local=np.concatenate(local_points), body=np.asarray(body_ids), surface=np.asarray(surface_ids),
                geom=np.asarray(geom_ids), vertex=np.asarray(vertex_ids))


def self_collision_pairs(model, geoms):
    ancestors = {}
    for body in range(1, model.nbody):
        parent, chain = body, []
        while parent:
            chain.append(parent)
            parent = int(model.body_parentid[parent])
        ancestors[body] = chain
    pairs = []
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1:]:
            a, b = ancestors[int(model.geom_bodyid[g1])], ancestors[int(model.geom_bodyid[g2])]
            common = set(a) & set(b)
            separation = min(a.index(c) + b.index(c) for c in common)
            if separation > 2:
                pairs.append((g1, g2))
    return pairs


def validate_and_label(candidate, baseline=RAW, allow_path_smoothing=False):
    baseline = Path(baseline).resolve()
    z = dict(np.load(candidate, allow_pickle=False))
    raw = dict(np.load(baseline, allow_pickle=False))
    np.testing.assert_array_equal(z["source_frame"], np.arange(len(raw["root_pos"])))
    if not allow_path_smoothing:
        np.testing.assert_array_equal(z["root_pos"][:, :2], raw["root_pos"][:, :2])
        np.testing.assert_array_equal(z["root_rot"], raw["root_rot"])
    else:
        assert np.linalg.norm(z['root_pos'][:, :2]-raw['root_pos'][:, :2], axis=-1).max() < .03
        rotation_change = (Rotation.from_quat(z['root_rot'])*Rotation.from_quat(raw['root_rot']).inv()).magnitude()
        assert rotation_change.max() < .15, 'Root rotation change exceeds modest smoothing allowance'
    np.testing.assert_allclose(z["fps"], raw["fps"], rtol=0, atol=0)
    geometry = Geometry()
    model, data = geometry.model, geometry.data
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    np.testing.assert_array_equal(z["joint_names"], joint_names)
    qpos = np.c_[z["root_pos"], z["root_rot"][:, [3, 0, 1, 2]], z["dof_pos"]]
    joint_margin = np.minimum(z["dof_pos"] - model.jnt_range[1:, 0], model.jnt_range[1:, 1] - z["dof_pos"])
    assert np.isfinite(qpos).all(), "Non-finite motion"
    assert joint_margin.min() >= -1e-8, "Joint limit violation"
    defs = point_definitions(model)
    n, fps = len(qpos), float(z["fps"])
    body_pos, body_rot, region_gaps, counts, usd_gaps, com_margin = [], [], [], [], [], []
    active_frames, active_points, active_world = [], [], []
    pairs = self_collision_pairs(model, sorted(set(defs["geom"])))
    collisions = {}
    for frame, q in enumerate(qpos):
        low, _, _ = geometry.at(q)
        usd_gaps.append(low[:len(geometry.surfaces)])
        rotations = data.xmat.reshape(-1, 3, 3)
        body_pos.append(data.xpos.copy())
        body_rot.append(rotations.copy())
        world = np.einsum("nij,nj->ni", rotations[defs["body"]], defs["local"]) + data.xpos[defs["body"]]
        gaps = np.full(len(REGIONS), np.inf)
        np.minimum.at(gaps, defs["surface"], world[:, 2])
        assert np.isfinite(gaps).all()
        region_gaps.append(gaps)
        mask = (world[:, 2] <= .001) & (world[:, 2] <= gaps[defs["surface"]] + .0005)
        ids = np.flatnonzero(mask)
        assert len(ids) >= 3, f"Insufficient ground-contact points at frame {frame}"
        active_frames.extend([frame] * len(ids))
        active_points.extend(ids)
        active_world.append(world[ids])
        counts.append(np.bincount(defs["surface"][ids], minlength=len(REGIONS)))
        hull = ConvexHull(world[ids, :2])
        com = data.subtree_com[1, :2]
        com_margin.append(float(-(hull.equations[:, :2] @ com + hull.equations[:, 2]).max()))
        for g1, g2 in pairs:
            distance = mj.mj_geomDistance(model, data, int(g1), int(g2), .01, None)
            if distance < -.001:
                key = str((int(g1), int(g2)))
                row = collisions.setdefault(key, dict(frames=0, worst_penetration_m=0.))
                row["frames"] += 1
                row["worst_penetration_m"] = max(row["worst_penetration_m"], float(-distance))
    gaps, counts, usd_gaps = np.asarray(region_gaps), np.asarray(counts), np.asarray(usd_gaps)
    body_pos, body_rot = np.asarray(body_pos), np.asarray(body_rot)
    frames, point_ids = np.asarray(active_frames), np.asarray(active_points)
    world = np.concatenate(active_world)
    bodies = defs["body"][point_ids]
    local = defs["local"][point_ids]
    prev, after = np.maximum(frames - 1, 0), np.minimum(frames + 1, n - 1)
    fixed_point_prev = np.einsum("nij,nj->ni", body_rot[prev, bodies], local) + body_pos[prev, bodies]
    fixed_point_after = np.einsum("nij,nj->ni", body_rot[after, bodies], local) + body_pos[after, bodies]
    velocity = (fixed_point_after - fixed_point_prev) / ((after - prev) / fps)[:, None]
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    active = counts > 0
    surface_speed = np.zeros(active.shape)
    np.maximum.at(surface_speed, (frames, defs["surface"][point_ids]), speed)
    modes = np.where(active, np.where(surface_speed > .015, "sliding_contact", "contact"), "non_contact")
    support_columns = [REGIONS.index(s) for s in SUPPORTS]
    assert gaps.min() >= .0001 - 2e-6, "Visible floor penetration"
    assert usd_gaps.min() >= -2e-6, "USD floor penetration"
    assert gaps[:, support_columns].max() <= .0006 + 2e-6, "A required support is floating"
    assert active[:, support_columns].all(), "A support contact was omitted"
    assert not collisions, f"Non-neighbor mesh intersections: {collisions}"
    assert min(com_margin) > 0, "CoM projection outside the support polygon"
    # Independent SciPy FK from the composed USD joints verifies every body and
    # all collision-shape gaps at the repaired poses, not only a sampled frame.
    source_geometry = json.loads((ROOT / "outputs/crawl_feasibility_20261022/contact_work/contact_sites.json").read_text())
    independent = usd_geometry_and_fk(z, source_geometry)
    fk_error = 0.
    for name, pos in independent["positions"].items():
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        fk_error = max(fk_error, float(abs(body_pos[:, body] - pos).max()),
                       float(abs(body_rot[:, body] - independent["rotations"][name]).max()))
    gap_error = float(abs(independent["gaps"] - usd_gaps).max())
    assert max(fk_error, gap_error) < 2e-6
    labels = dict(
        annotation_version=np.asarray(3), geometry_basis=np.asarray("visible_mesh"), fps=z["fps"], time_s=np.arange(n) / fps,
        surface_names=np.asarray(REGIONS), surface_active=active, surface_point_count=counts,
        surface_clearance_m=gaps, surface_mode=modes, surface_tangent_speed_m_s=surface_speed,
        body_names=np.asarray([mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or "world" for b in range(model.nbody)]),
        body_position_w=body_pos, body_rotation_w=body_rot, point_position_body=defs["local"],
        point_body_index=defs["body"], point_surface_index=defs["surface"],
        point_geom_index=defs["geom"], point_mesh_vertex_index=defs["vertex"],
        frame_offsets=np.r_[0, np.cumsum(counts.sum(1))], contact_frame=frames,
        contact_point_index=point_ids, contact_surface_index=defs["surface"][point_ids], contact_position_w=world,
        contact_material_velocity_w=velocity, contact_tangent_speed_m_s=speed,
        contact_mode=np.where(speed > .015, "sliding_contact", "contact"), contact_gap_m=world[:, 2],
        contact_normal_w=np.tile([0., 0., 1.], (len(frames), 1)),
        usd_surface_names=np.asarray([s["name"] for s in geometry.surfaces]), usd_surface_clearance_m=usd_gaps,
    )
    metrics = {}
    for name, motion in (("raw", raw), ("corrected", z)):
        dq = np.diff(motion["dof_pos"], axis=0)
        ddq = np.diff(dq, axis=0)
        metrics[name] = dict(max_joint_frame_step_deg=float(np.rad2deg(abs(dq).max())),
                             max_joint_speed_rad_s=float(abs(dq).max() * fps),
                             p99_joint_speed_rad_s=float(np.percentile(abs(dq), 99) * fps),
                             max_joint_acceleration_rad_s2=float(abs(ddq).max() * fps ** 2),
                             p99_joint_acceleration_rad_s2=float(np.percentile(abs(ddq), 99) * fps ** 2))
    assert metrics["corrected"]["max_joint_frame_step_deg"] < 8, "Abrupt joint jump"
    change = z["dof_pos"] - raw["dof_pos"]
    report = dict(
        validation_status="geometric_crawl_repair_passed", frames=n, fps=fps, duration_s=n / fps,
        full_body_geometric_contact_labels_verified=True, physical_execution_verified=False,
        contact_geometry_basis="All visual mesh vertices, partitioned into 16 anatomical regions. USD collider clearance checked separately.",
        support_schedule="Both knees and both hands remain grounded throughout the clip and may slide. This is an explicit repair choice, not a recovered human force-contact schedule.",
        required_support_regions=list(SUPPORTS), required_support_gap_range_mm=[float(gaps[:, support_columns].min() * 1000), float(gaps[:, support_columns].max() * 1000)],
        contact_frames_by_region={name: int(active[:, s].sum()) for s, name in enumerate(REGIONS)},
        simultaneous_region_count={str(int(v)): int((active.sum(1) == v).sum()) for v in np.unique(active.sum(1))},
        min_visible_mesh_floor_gap_mm=float(gaps.min() * 1000), min_usd_collider_floor_gap_mm=float(usd_gaps.min() * 1000),
        joint_limit_violations=0, min_joint_limit_margin_rad=float(joint_margin.min()),
        non_neighbor_visible_mesh_intersections=collisions, checked_mesh_pairs=len(pairs),
        self_collision_scope="Convex mesh distance for body pairs more than two joints apart; joint housings and neighboring links excluded.",
        min_static_com_support_margin_mm=float(min(com_margin) * 1000),
        static_support_note="CoM projection inside the contact polygon; not a dynamic force/torque feasibility test.",
        independent_usd_fk_max_error=float(fk_error), independent_usd_gap_max_error_m=gap_error,
        horizontal_path_preserved=bool(np.array_equal(z['root_pos'][:, :2], raw['root_pos'][:, :2])),
        root_orientation_preserved=bool(np.array_equal(z['root_rot'], raw['root_rot'])), timing_preserved=True,
        max_joint_change_from_raw_deg=float(np.rad2deg(abs(change).max())),
        rms_joint_change_from_raw_deg=float(np.rad2deg(np.sqrt(np.mean(change ** 2)))), smoothness=metrics,
        ground_gap_tolerance_m=.001, patch_height_tolerance_m=.0005, sliding_speed_threshold_m_s=.015,
        source_motion=str(baseline.relative_to(ROOT)), source_motion_sha256=digest(baseline), model_sha256=digest(MODEL),
        allow_path_smoothing=allow_path_smoothing,
        candidate_sha256=digest(candidate),
        limitations=["Geometric reference only; physical tracking and dynamic support forces are not validated.",
                     "The current WBC models feet only; knee/hand support requires full-body controller/collision integration.",
                     "Neighboring-link self-collision clearance is not certified.",
                     "The portable motion still requires conversion before training."],
    )
    return z, labels, report


def render(z, output):
    model = mj.MjModel.from_xml_path(str(MODEL))
    data = mj.MjData(model)
    width, height = 1280, 720
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    option = mj.MjvOption()
    option.geomgroup[3] = 0
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.7, 130, -15
    qpos = np.c_[z["root_pos"], z["root_rot"][:, [3, 0, 1, 2]], z["dof_pos"]]
    path = gaussian_filter1d(qpos[:, :3], 4., axis=0)
    fps, n = float(z["fps"]), len(qpos)
    with mj.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
            str(output / "reference.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for i, q in enumerate(qpos):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            camera.lookat[:] = path[i]
            camera.lookat[2] = .22
            renderer.update_scene(data, camera, scene_option=option)
            frame = renderer.render()
            cv2.rectangle(frame, (0, 0), (width, 70), (247, 249, 252), -1)
            cv2.putText(frame, "T1 | Corrected crawl", (24, 43), cv2.FONT_HERSHEY_SIMPLEX, .85, (25, 35, 49), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{i / fps:05.2f} / {n / fps:.2f} s", (width - 250, 43), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 35, 49), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, height - 37), (width, height), (247, 249, 252), -1)
            cv2.putText(frame, "Knees and hands grounded | Geometric reference; physical execution untested", (24, height - 14), cv2.FONT_HERSHEY_SIMPLEX, .5, (76, 86, 100), 1, cv2.LINE_AA)
            writer.append_data(frame)
            if i == n // 2:
                cv2.imwrite(str(output / "preview.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % 100 == 0 or i == n - 1:
                print("Rendered", i + 1, "/", n, flush=True)
    count = 0
    with imageio.get_reader(str(output / "reference.mp4")) as reader:
        metadata = reader.get_meta_data()
        for frame in reader:
            assert frame.shape[:2] == (height, width)
            count += 1
    assert count == n
    return dict(path="reference.mp4", frames_decoded=count, fps=metadata["fps"], width=width, height=height,
                sha256=digest(output / "reference.mp4"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/crawl_repair/corrected.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/crawl_repair/release")
    parser.add_argument("--video", action="store_true")
    parser.add_argument('--baseline', type=Path, default=RAW)
    parser.add_argument('--allow-path-smoothing', action='store_true')
    args = parser.parse_args()
    z, labels, report = validate_and_label(args.input, args.baseline, args.allow_path_smoothing)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    motion_path = output / "a11_crawl_forward_female1_t1.npz"
    z["contact_annotations"] = np.asarray("eval_data/reference_crawl_forward/contact_geometry.npz")
    z["contact_annotation_version"] = np.asarray(3)
    np.savez_compressed(motion_path, **z)
    np.savez_compressed(output / "contact_geometry.npz", **labels)
    with (output / "contact_labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s"] + [f"{name}_{field}" for name in REGIONS for field in ("mode", "points", "clearance_mm")])
        for i, t in enumerate(labels["time_s"]):
            writer.writerow([i, f"{t:.6f}"] + [value for j in range(len(REGIONS)) for value in
                             (labels["surface_mode"][i, j], int(labels["surface_point_count"][i, j]),
                              f"{1000 * labels['surface_clearance_m'][i, j]:.6f}")])
    sites = dict(annotation_version=3, source="T1 visual mesh vertices", model=str(MODEL.relative_to(ROOT)),
                 model_sha256=digest(MODEL), coordinates="contact_geometry.npz", ground_height_m=0.,
                 contact_tolerance_m=.001, patch_tolerance_m=.0005, surface_names=list(REGIONS),
                 definition="Every mesh vertex has one anatomical region and a fixed body-local coordinate. Knees use the adjacent half of thigh/shin meshes; hands use the distal 180 mm boundary on the signed forearm-link y axis.",
                 physics_note="Geometric annotations do not add these contact surfaces to the WBC/PhysX collision model.")
    (output / "contact_sites.json").write_text(json.dumps(sites, indent=2) + "\n")
    resolved_motion = motion_path.resolve()
    report["motion"] = str(resolved_motion.relative_to(ROOT) if resolved_motion.is_relative_to(ROOT) else resolved_motion)
    report["motion_sha256"] = digest(motion_path)
    report["reproduction_argv"] = ["python", "scripts/finalize_crawl_repair.py", "--input", str(args.input),
                                   "--baseline", str(args.baseline), "--output", str(output)]
    if args.allow_path_smoothing:
        report["reproduction_argv"].append("--allow-path-smoothing")
    if args.video:
        report["reproduction_argv"].append("--video")
    if args.video:
        report["video"] = render(z, output)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifact_sha256.json"}
    (output / "artifact_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
