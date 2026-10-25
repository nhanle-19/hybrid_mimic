"""Correct the saved GMR crawl with simultaneous knee/hand floor constraints.

This is a geometric reference repair, not a physics rollout. The original
horizontal path, root orientation and frame timing are retained. The support
sites may slide; planted-hand XY positions and recovery lifts are not preserved.
Source motion files are never changed.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from check_crawl_support import surface_groups


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/crawl_feasibility_20261022/female1_forward_raw.npz"
MODEL = ROOT / "GMR/assets/booster_t1/T1_serial.xml"
SUPPORTS = ("left_knee", "right_knee", "left_hand", "right_hand")


class Geometry:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(str(MODEL))
        self.data = mj.MjData(self.model)
        self.columns = np.r_[2, np.arange(6, self.model.nv)]
        self.jp = np.zeros((3, self.model.nv))
        self.jr = self.jp.copy()
        self.surfaces = json.loads((ROOT / "configs/crawl_contact_surfaces.json").read_text())["surfaces"]
        for s in self.surfaces:
            s["body_id"] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, s["body"])
            s["transform"] = np.asarray(s["transform_to_body"])
        self.meshes = []
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] and self.model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[g]
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                vertices = self.model.mesh_vert[start:start + count].astype(float)
                self.meshes.append((g, vertices[ConvexHull(vertices).vertices]))
        self.groups = surface_groups(self.model)
        self.group_names = list(self.groups)
        self.support_indices = [len(self.surfaces) + len(self.meshes) + self.group_names.index(name) for name in SUPPORTS]

    def mesh_low(self, geom, vertices):
        r = self.data.geom_xmat[geom].reshape(3, 3)
        point = vertices[np.argmin(vertices @ r[2])]
        return self.data.geom_xpos[geom] + r @ point, int(self.model.geom_bodyid[geom])

    def at(self, q):
        self.data.qpos[:] = q
        mj.mj_forward(self.model, self.data)
        lows, jac, points = [], [], []

        def append(point, body):
            mj.mj_jac(self.model, self.data, self.jp, self.jr, point, body)
            lows.append(point[2])
            jac.append(self.jp[2, self.columns].copy())
            points.append(point)

        for s in self.surfaces:
            b = s["body_id"]
            br = self.data.xmat[b].reshape(3, 3)
            transform = s["transform"]
            r = br @ transform[:3, :3]
            centre = self.data.xpos[b] + br @ transform[:3, 3]
            row = r[2]
            if s["shape"] == "Cube":
                local = -s["half_size"] * np.sign(row)
            elif s["shape"] == "Cylinder":
                local = np.r_[-s["radius"] * row[:2] / max(np.linalg.norm(row[:2]), 1e-12),
                              -s["height"] / 2 * np.sign(row[2])]
            else:
                local = -s["radius"] * row / np.linalg.norm(row)
            append(centre + r @ local, b)
        for geom, vertices in self.meshes:
            append(*self.mesh_low(geom, vertices))
        for group in self.groups.values():
            candidates = [self.mesh_low(g, v) for g, v in group]
            append(*min(candidates, key=lambda pair: pair[0][2]))
        return np.asarray(lows), np.asarray(jac), np.asarray(points)


def project(geometry, base, target, initial=None, maxiter=150, hand_targets=(), hand_scale=.01):
    # Root height plus the 23 articulated joints; horizontal path and orientation
    # stay fixed so this cannot replace the crawl with a stationary pose.
    scale = np.r_[.045, np.full(23, .28)]
    lower = np.r_[-.15, geometry.model.jnt_range[1:, 0] - base[7:]]
    upper = np.r_[.15, geometry.model.jnt_range[1:, 1] - base[7:]]
    last_x = last_value = None

    def evaluate(x):
        nonlocal last_x, last_value
        if last_x is None or not np.array_equal(x, last_x):
            q = base.copy()
            q[2] += x[0]
            q[7:] += x[1:]
            last_value = geometry.at(q)
            last_x = x.copy()
        return last_value

    floor = np.r_[np.zeros(len(geometry.surfaces)), np.full(len(geometry.meshes) + len(geometry.groups), .00010)]
    support = geometry.support_indices
    # A small band is more robust at changes of the lowest mesh vertex than an
    # exact equality. All supports must stay between 0.1 and 0.6 mm above floor.
    def constraint(x):
        low = evaluate(x)[0]
        return 100 * np.r_[low - floor, .0006 - low[support]]

    def jacobian(x):
        jac = evaluate(x)[1]
        return 100 * np.r_[jac, -jac[support]]

    def anchor_terms(x):
        evaluate(x)
        cost, derivative = 0., np.zeros_like(x)
        for body, local, desired in hand_targets:
            point = geometry.data.xpos[body]+geometry.data.xmat[body].reshape(3, 3)@local
            error = (point[:2]-desired)/hand_scale
            mj.mj_jac(geometry.model, geometry.data, geometry.jp, geometry.jr, point, body)
            cost += .5*np.dot(error, error)
            derivative += error@geometry.jp[:2, geometry.columns]/hand_scale
        return cost, derivative

    def objective(x):
        return .5*np.sum(((x-target)/scale)**2)+(anchor_terms(x)[0] if hand_targets else 0.)

    def gradient(x):
        return (x-target)/scale**2+(anchor_terms(x)[1] if hand_targets else 0.)
    result = minimize(objective, np.clip(target if initial is None else initial, lower, upper),
                      jac=gradient, bounds=list(zip(lower, upper)),
                      constraints=[dict(type="ineq", fun=constraint, jac=jacobian)],
                      method="SLSQP", options=dict(ftol=1e-9, maxiter=maxiter))
    q = base.copy()
    q[2] += result.x[0]
    q[7:] += result.x[1:]
    low, _, points = evaluate(result.x)
    violation = float(max(0., -constraint(result.x).min() / 100))
    row = dict(success=violation < 2e-6, optimizer_success=bool(result.success), iterations=int(result.nit),
               violation_m=violation, support_gap_mm=dict(zip(SUPPORTS, map(float, 1000 * low[support]))),
               min_usd_gap_mm=float(1000 * low[:len(geometry.surfaces)].min()),
               min_mesh_gap_mm=float(1000 * low[len(geometry.surfaces):len(geometry.surfaces) + len(geometry.meshes)].min()),
               root_change_mm=float(1000 * result.x[0]), max_joint_change_deg=float(np.rad2deg(abs(result.x[1:]).max())),
               message=str(result.message))
    return q, result.x, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/crawl_repair/corrected.npz")
    parser.add_argument("--frames", type=int, nargs="*", help="Specific frames for feasibility probes only")
    parser.add_argument("--smooth-passes", type=int, default=5)
    args = parser.parse_args()
    z = dict(np.load(RAW))
    q0 = np.c_[z["root_pos"], z["root_rot"][:, [3, 0, 1, 2]], z["dof_pos"]]
    geometry = Geometry()
    ids = args.frames if args.frames else list(range(len(q0)))
    out, deltas, rows = [], [], []
    started = time.monotonic()
    for frame in ids:
        previous = deltas[-1] if deltas and rows[-1]["success"] else np.zeros(24)
        target = .65 * previous
        q, delta, row = project(geometry, q0[frame], target, initial=previous)
        row["frame"] = frame
        out.append(q)
        deltas.append(delta)
        rows.append(row)
        if frame % 20 == 0 or args.frames or not row["success"]:
            print(json.dumps(dict(row, elapsed_s=round(time.monotonic() - started, 1))), flush=True)
    if not args.frames:
        for iteration in range(args.smooth_passes):
            # Smooth the repaired trajectory itself, including inherited IK
            # spikes, then reproject every frame onto the contact constraints.
            poses = np.asarray(out)
            targets = gaussian_filter1d(np.c_[poses[:, 2], poses[:, 7:]], 1.5, axis=0)
            targets -= np.c_[q0[ids, 2], q0[ids, 7:]]
            for i, frame in enumerate(ids):
                q, delta, row = project(geometry, q0[frame], targets[i], initial=deltas[i])
                if row["success"]:
                    out[i], deltas[i], rows[i] = q, delta, dict(row, frame=frame)
            print("Smooth pass", iteration + 1, "failed", sum(not r["success"] for r in rows), flush=True)
    out = np.asarray(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, fps=z["fps"], root_pos=out[:, :3], root_rot=out[:, [4, 5, 6, 3]],
                        dof_pos=out[:, 7:], source_frame=np.asarray(ids),
                        joint_names=np.asarray([mj.mj_id2name(geometry.model, mj.mjtObj.mjOBJ_JOINT, j)
                                                for j in range(1, geometry.model.njnt)]))
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Saved", args.output, "failed", sum(not r["success"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
