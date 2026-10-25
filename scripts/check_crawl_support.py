"""Check anatomical crawl surfaces and the height edits, not just USD labels.

python scripts/check_crawl_support.py

The knee area is selected from the half of each adjoining link nearest the
knee hinge. Hand tips are the distal mesh beyond 180 mm along the forearm link;
the remainder is the forearm/elbow region. These mesh partitions are analysis
regions, not new physics colliders. Their boundary sensitivity is also checked.
"""

import csv
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "eval_data/reference_crawl_forward"
WORK = ROOT / "outputs/crawl_feasibility_20261022"
MOTION = ROOT / "retargeted_motion/a11_crawl_forward_female1_t1.npz"
MODEL = ROOT / "GMR/assets/booster_t1/T1_serial.xml"


def read(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_groups(model, knee_fraction=.5, hand_start=.180):
    groups = {}
    for side, word, sign in (("Left", "left", 1), ("Right", "right", -1)):
        for body, kind in ((f"Hip_Yaw_{side}", "knee"), (f"Shank_{side}", "knee"),
                           (f"{word}_hand_link", "forearm"), (f"{word}_hand_link", "hand")):
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
            geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id
                        and model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH)
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count].astype(float)
            rotation = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
            local = vertices @ rotation.T + model.geom_pos[geom]
            if kind == "knee":
                next_body = f"Shank_{side}" if "Hip" in body else f"Ankle_Cross_{side}"
                link_axis = model.body_pos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, next_body)]
                along = local @ link_axis / np.dot(link_axis, link_axis)
                selected = along >= knee_fraction if "Hip" in body else along <= knee_fraction
            else:
                selected = sign * local[:, 1] >= hand_start
                if kind == "forearm":
                    selected = ~selected
            if not selected.any():
                raise ValueError(f"Empty anatomical region: {word}_{kind}")
            groups.setdefault(f"{word}_{kind}", []).append((geom, vertices[selected]))
    return groups


def measure(model, qpos, groups):
    data = mj.MjData(model)
    names = list(groups)
    gaps = np.full((len(qpos), len(names)), np.inf)
    points = np.empty((*gaps.shape, 3))
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        for region, name in enumerate(names):
            for geom, vertices in groups[name]:
                rotation = data.geom_xmat[geom].reshape(3, 3)
                heights = vertices @ rotation[2] + data.geom_xpos[geom, 2]
                lowest = int(np.argmin(heights))
                if heights[lowest] < gaps[frame, region]:
                    gaps[frame, region] = heights[lowest]
                    points[frame, region] = rotation @ vertices[lowest] + data.geom_xpos[geom]
    np.testing.assert_allclose(points[:, :, 2], gaps, atol=1e-12)
    if not np.isfinite(gaps).all():
        raise ValueError("Missing mesh measurements")
    return gaps, points


def intervals(mask, fps):
    ids = np.flatnonzero(mask)
    return [dict(first_frame=int(part[0]), last_frame=int(part[-1]),
                 first_time_s=float(part[0] / fps), last_time_s=float(part[-1] / fps))
            for part in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(part)]


def main():
    current, raw = read(MOTION), read(WORK / "female1_forward_raw.npz")
    labels = read(CASE / "contact_geometry.npz")
    if int(labels.get("annotation_version", 1)) >= 3:
        from check_crawl_contacts import audit
        report = audit(CASE, MOTION)
        (CASE / "crawl_support_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["geometric_contact_labels_pass"]:
            raise SystemExit(1)
        return
    model = mj.MjModel.from_xml_path(str(MODEL))
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    np.testing.assert_array_equal(current["joint_names"], joint_names)
    np.testing.assert_array_equal(current["root_pos"][:, :2], raw["root_pos"][:, :2])
    np.testing.assert_array_equal(current["root_rot"], raw["root_rot"])
    np.testing.assert_allclose(current["fps"], raw["fps"], rtol=0, atol=1e-12)
    qpos = lambda z: np.c_[z["root_pos"], z["root_rot"][:, [3, 0, 1, 2]], z["dof_pos"]]
    groups = surface_groups(model)
    names = list(groups)
    current_gaps, points = measure(model, qpos(current), groups)
    raw_gaps, _ = measure(model, qpos(raw), groups)
    tolerance = .001
    near = .005
    contact = current_gaps <= tolerance
    # Recheck the same surfaces under substantially different anatomical cuts.
    sensitivity = {}
    for fraction, tip in ((.4, .170), (.6, .190)):
        gaps, _ = measure(model, qpos(current), surface_groups(model, fraction, tip))
        sensitivity[f"knee_link_fraction_{fraction}_hand_start_{tip}"] = {
            "contact_frame_counts": dict(zip(names, map(int, (gaps <= tolerance).sum(0)))),
            "different_contact_decisions": int(np.count_nonzero((gaps <= tolerance) != contact)),
        }
    # A uniform root lowering shifts every surface by the same amount. Verify
    # this with an independent FK pass, then test clearance to all USD shapes.
    lowered = qpos(current).copy()
    lowered[:, 2] -= .010
    lowered_gaps, _ = measure(model, lowered, groups)
    np.testing.assert_allclose(lowered_gaps, current_gaps - .010, rtol=0, atol=1e-12)
    lowest_collider = labels["surface_clearance_m"].min(1)
    needed_lowering = np.maximum(current_gaps - tolerance, 0.)
    penetration_if_lowered = np.maximum(needed_lowering - lowest_collider[:, None], 0.)
    root_raise = current["root_pos"][:, 2] - raw["root_pos"][:, 2]
    fps, n = float(current["fps"]), len(current_gaps)
    info = {}
    for j, name in enumerate(names):
        saved_name = name if name.endswith("knee") else name.split("_")[0] + "_forearm_hand"
        saved_column = labels["surface_names"].tolist().index(saved_name) if saved_name in labels["surface_names"] else None
        stored = np.zeros(n, dtype=bool) if saved_column is None else labels["surface_active"][:, saved_column]
        info[name] = dict(
            frames_within_1mm=int(contact[:, j].sum()), frames_within_5mm=int((current_gaps[:, j] <= near).sum()),
            mesh_gap_mm=dict(min=float(1000 * current_gaps[:, j].min()), median=float(1000 * np.median(current_gaps[:, j])),
                             max=float(1000 * current_gaps[:, j].max())),
            saved_label_name=saved_name if saved_column is not None else None,
            saved_label_frames=int(stored.sum()), touching_without_saved_region_label=int((contact[:, j] & ~stored).sum()),
            contact_intervals=intervals(contact[:, j], fps),
            hovering_frames_that_cannot_be_fixed_by_root_lowering_without_collider_penetration=int(
                ((~contact[:, j]) & (penetration_if_lowered[:, j] > tolerance)).sum()),
        )
    report = dict(
        status="incomplete_crawl_reference_and_contact_labels", full_body_contact_labels_verified=False,
        scope="Anatomical knee, forearm and hand surface check on the saved retarget, accounting for reference height edits.",
        frames=n, fps=fps, regions=info,
        root_raise_from_raw_mm=dict(min=float(root_raise.min() * 1000), median=float(np.median(root_raise) * 1000),
                                    mean=float(root_raise.mean() * 1000), max=float(root_raise.max() * 1000)),
        region_definition=dict(knee="Half of each adjoining mesh nearest the knee hinge, measured along the hip-to-knee or knee-to-ankle axis.",
                               hand="Distal mesh >=180 mm along the signed forearm-link y axis.",
                               forearm="Remaining forearm/hand-link mesh, including the elbow housing."),
        partition_sensitivity=sensitivity,
        findings=[
            "The existing schema has no explicit knee labels. The near-floor thigh-link surfaces were confirmed to lie at the knee housings.",
            "Both hand tips reach the 1 mm geometric tolerance somewhere in the clip; the current right forearm/hand label is never active.",
            "The forearm shafts do not reach the 1 mm tolerance in the current poses. They cannot be declared contacts merely because a crawl was intended.",
            "Ground preparation raised the entire robot and then grounded its lowest allowed limb without enforcing multiple support regions.",
            "The general retargeting configuration matches feet more strongly than knees and has no wrist/hand surface or floor-contact targets.",
            "Lowering the root enough to ground hovering limbs causes other colliders to penetrate the floor; multi-contact correction requires pose and contact-geometry work.",
        ],
        previous_audit_scope="USD label consistency only; its pass result is not a successful full-body crawl contact validation.",
        limitations=["A 1 mm surface-gap label is geometric, not a measured force or proof of balance.",
                     "The intended per-frame forearm and hand support schedule has not been established from the human source.",
                     "Anatomical partitions are analysis regions and do not modify the physics collision model."],
        provenance={str(p.relative_to(ROOT)): sha256(p) for p in (MOTION, WORK / "female1_forward_raw.npz", MODEL,
                     CASE / "contact_geometry.npz", ROOT / "GMR/general_motion_retargeting/ik_configs/smplx_to_t1.json")},
        reproduce="python scripts/check_crawl_support.py",
    )
    np.savez_compressed(CASE / "crawl_support_audit.npz", fps=fps, time_s=np.arange(n) / fps,
                        region_names=np.asarray(names), current_gap_m=current_gaps, raw_gap_m=raw_gaps,
                        current_lowest_position_w=points, within_1mm=contact, root_raise_m=root_raise,
                        collider_penetration_if_lowered_m=penetration_if_lowered)
    with (CASE / "crawl_support_audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "region", "mesh_gap_mm", "within_1mm", "raw_mesh_gap_mm",
                         "root_raise_mm", "collider_penetration_if_lowered_to_contact_mm"])
        for i in range(n):
            for j, name in enumerate(names):
                writer.writerow([i, f"{i / fps:.6f}", name, f"{1000 * current_gaps[i, j]:.6f}", int(contact[i, j]),
                                 f"{1000 * raw_gaps[i, j]:.6f}", f"{1000 * root_raise[i]:.6f}",
                                 f"{1000 * penetration_if_lowered[i, j]:.6f}"])
    (CASE / "crawl_support_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "root_raise_from_raw_mm", "partition_sensitivity")}, indent=2))
    for name, row in info.items():
        print(name, row["frames_within_1mm"], "contact frames;", row["mesh_gap_mm"], "mm")


if __name__ == "__main__":
    main()
