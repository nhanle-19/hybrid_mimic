"""Build reversible T1 references with a supported, stationary one-second ending.

Run with .venvs/dsms/bin/python. The source library is never overwritten.
This checks geometry and bounded static/dynamic support forces at the ending;
it does not certify tracking of the complete motion by a learned policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, RotationSpline

from dsms_t1_model import ROOT, actuator_limits, build_model, load_reference, sha
from mirror_training_motions import generate

FEET = ('left_foot_link', 'right_foot_link')
ASSET = ROOT / 'source/whole_body_tracking/whole_body_tracking/assets/booster/t1'
GENERATOR_SHA256 = sha(__file__)


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. - 15.*u + 6.*u*u)


def slowing_phase(u):
    """Unit-distance phase with initial rate 2, final rate 0 and zero end accelerations.

    Traversing D source seconds in 2D output seconds starts at original speed.
    Its derivative is 2(1-3u²+2u³), which stays nonnegative on [0, 1].
    """
    return 2.*u - 2.*u**3 + u**4


class EndingBuilder:
    def __init__(self, model_path):
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.bids = [self.model.body(n).id for n in FEET]
        contacts = json.loads((ASSET / 'wbc_acc_contacts.json').read_text())
        self.vertices = np.array([contacts['bodies'][n]['sole_vertices'] for n in FEET])
        motors = actuator_limits()
        self.names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
        self.efforts = np.array([motors[n]['effort'] for n in self.names])
        self.speeds = np.array([motors[n]['speed'] for n in self.names])

    def geometry(self, q, velocity=None):
        d, m = self.data, self.model
        d.qpos[:] = q
        d.qvel[:] = 0. if velocity is None else velocity
        mj.mj_forward(m, d)
        rotations = d.xmat[self.bids].reshape(2, 3, 3).copy()
        positions = d.xpos[self.bids].copy()
        corners = np.einsum('fij,fvj->fvi', rotations, self.vertices) + positions[:, None]
        return positions, rotations, corners, d.subtree_com[1].copy()

    def support(self, q, velocity=None, acceleration=None):
        """Inverse-dynamics force feasibility, with only near-ground sole corners.

        No force is permitted on an elevated corner. The inscribed square
        friction pyramid lies inside mu=0.6's circular cone. Joint torques are
        bounded by the actual T1 actuator limits. Minimize peak torque fraction.
        """
        _, _, corners, com = self.geometry(q, velocity)
        d, m = self.data, self.model
        selected = np.argwhere((corners[..., 2] >= -.001) & (corners[..., 2] <= .001))
        if len(selected) < 3:
            return dict(passed=False, contacts=len(selected), reason='Insufficient supported sole corners')
        jacobians = []
        for foot, vertex in selected:
            jp = np.zeros((3, m.nv))
            mj.mj_jac(m, d, jp, None, corners[foot, vertex], self.bids[foot])
            jacobians.append(jp)
        matrix = np.concatenate(jacobians, axis=0).T
        required = d.qfrc_bias.copy()
        if acceleration is not None:
            mass = np.empty((m.nv, m.nv))
            mj.mj_fullM(m, d, mass)
            required += mass @ acceleration
        nf = matrix.shape[1]
        inequalities, bounds = [], []
        for i in range(len(selected)):
            for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                row = np.zeros(nf+1)
                row[3*i:3*i+3] = (x, y, -.6/np.sqrt(2.))
                inequalities.append(row)
                bounds.append(0.)
        inequalities.extend(np.c_[-matrix[6:], -self.efforts])
        bounds.extend(-required[6:])
        inequalities.extend(np.c_[matrix[6:], -self.efforts])
        bounds.extend(required[6:])
        result = linprog(np.r_[np.zeros(nf), 1.], A_ub=np.asarray(inequalities), b_ub=bounds,
                         A_eq=np.c_[matrix[:6], np.zeros(6)], b_eq=required[:6],
                         bounds=[(-500., 500.), (-500., 500.), (0., 500.)]*len(selected)+[(0., 1.)],
                         method='highs')
        record = dict(passed=bool(result.success), contacts=len(selected))
        if result.success:
            torque = required[6:] - matrix[6:] @ result.x[:-1]
            hull = ConvexHull(corners[selected[:, 0], selected[:, 1], :2])
            record.update(peak_torque_fraction=float(result.x[-1]),
                          equilibrium_residual=float(np.max(np.abs(matrix[:6]@result.x[:-1]-required[:6]))),
                          com_support_margin_m=float(-(hull.equations[:, :2]@com[:2]+hull.equations[:, 2]).max()),
                          joint_torque_nm=torque.tolist())
        else:
            record['reason'] = 'No bounded friction/torque solution'
        return record

    def settled_pose(self, q):
        """Ground and level both feet while preserving their horizontal placements/yaw."""
        positions, rotations, _, _ = self.geometry(q)
        target_p = positions.copy()
        yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
        target_r = Rotation.from_euler('z', yaw[:, None]).as_matrix()
        target_p[:, 2] = -np.einsum('fij,fvj->fvi', target_r, self.vertices)[..., 2].min(1)
        root_r = Rotation.from_quat(q[[4, 5, 6, 3]])
        middle = target_p[:, :2].mean(0)

        def decode(x):
            pose = q.copy()
            pose[:3] += x[:3]
            pose[3:7] = (Rotation.from_rotvec(x[3:6])*root_r).as_quat()[[3, 0, 1, 2]]
            pose[7:] += x[6:]
            return pose

        def residual(x):
            p, r, _, com = self.geometry(decode(x))
            return np.r_[((p-target_p)*300.).ravel(),
                         (Rotation.from_matrix(r@target_r.transpose(0, 2, 1)).as_rotvec()*30.).ravel(),
                         (com[:2]-middle)*10., x[:3]*3., x[3:6]*2., x[6:]*.3]

        lower = np.r_[[-.08]*3, [-.2]*3, np.maximum(self.model.jnt_range[1:, 0]-q[7:], -.6)]
        upper = np.r_[[.08]*3, [.2]*3, np.minimum(self.model.jnt_range[1:, 1]-q[7:], .6)]
        result = least_squares(residual, np.clip(np.zeros(self.model.nv), lower, upper),
                               bounds=(lower, upper), method='dogbox', max_nfev=120,
                               ftol=1e-9, xtol=1e-9)
        target = decode(result.x)
        p, _, corners, _ = self.geometry(target)
        # A single tiny rigid lift makes the saved pose strictly nonpenetrating.
        target[2] += max(0., -float(corners[..., 2].min()))
        support = self.support(target)
        self_penetration = max((max(0., -float(c.dist)) for c in self.data.contact
                                if 0 not in [int(self.model.geom_bodyid[g]) for g in c.geom]), default=0.)
        record = dict(max_foot_target_error_m=float(np.abs(p-target_p).max()),
                      max_joint_adjustment_rad=float(np.abs(result.x[6:]).max()),
                      root_adjustment_m=float(np.linalg.norm(result.x[:3])),
                      max_self_penetration_m=self_penetration, support=support)
        record['passed'] = (record['max_foot_target_error_m'] < .0005 and record['support']['passed']
                            and record['support']['peak_torque_fraction'] < .85
                            and self_penetration < .003
                            and record['support']['com_support_margin_m'] > .01)
        return target, record

    def velocities(self, q, fps):
        velocity = np.zeros((len(q), self.model.nv))
        for i in range(len(q)):
            a, b = max(0, i-1), min(len(q)-1, i+1)
            mj.mj_differentiatePos(self.model, velocity[i], (b-a)/fps, q[a], q[b])
        return velocity

    def fit_feet(self, goal, target_p, target_r):
        """Keep the requested sole trajectories while staying close to the pose goal."""
        root_r = Rotation.from_quat(goal[[4, 5, 6, 3]])

        def decode(x):
            q = goal.copy()
            q[:3] += x[:3]
            q[3:7] = (Rotation.from_rotvec(x[3:6])*root_r).as_quat()[[3, 0, 1, 2]]
            q[7:] += x[6:]
            return q

        def residual(x):
            p, r, _, _ = self.geometry(decode(x))
            return np.r_[((p-target_p)*3000.).ravel(),
                         (Rotation.from_matrix(r@target_r.transpose(0, 2, 1)).as_rotvec()*300.).ravel(),
                         x[:3]*10., x[3:6]*3., x[6:]]

        lower = np.r_[[-.12]*3, [-.3]*3, np.maximum(self.model.jnt_range[1:, 0]-goal[7:], -.8)]
        upper = np.r_[[.12]*3, [.3]*3, np.minimum(self.model.jnt_range[1:, 1]-goal[7:], .8)]
        result = least_squares(residual, np.clip(np.zeros(self.model.nv), lower, upper),
                               bounds=(lower, upper), method='dogbox', max_nfev=50,
                               ftol=1e-8, xtol=1e-8)
        q = decode(result.x)
        error = float(np.abs(self.geometry(q)[0]-target_p).max())
        return q, error

    def make_tail(self, original, end, target, fps, source_seconds=.8):
        """Retain the prefix; slow the last source window and blend in a grounded pose."""
        span = min(round(source_seconds*fps), end)
        start = end-span
        t = np.arange(len(original))/fps
        u = np.linspace(0., 1., 2*span+1)
        source_t = start/fps + span/fps*slowing_phase(u)
        scalar = PchipInterpolator(t, np.c_[original[:, :3], original[:, 7:]], axis=0)(source_t)
        rotation = RotationSpline(t, Rotation.from_quat(original[:, [4, 5, 6, 3]]))(source_t)
        source_poses = np.c_[scalar[:, :3], rotation.as_quat()[:, [3, 0, 1, 2]], scalar[:, 3:]]
        blend = smoothstep(u)
        scalar = (1.-blend[:, None])*scalar + blend[:, None]*np.r_[target[:3], target[7:]]
        difference = Rotation.from_quat(target[[4, 5, 6, 3]]) * rotation.inv()
        rotation = Rotation.from_rotvec(blend[:, None]*difference.as_rotvec())*rotation
        tail = np.c_[scalar[:, :3], rotation.as_quat()[:, [3, 0, 1, 2]], scalar[:, 3:]]
        tail[0], tail[-1] = original[start], target
        final_p, final_r, _, _ = self.geometry(target)
        lift, foot_errors = [], []
        for i, pose in enumerate(tail):
            if 0 < i < len(tail)-1:
                p, r, corners, _ = self.geometry(source_poses[i])
                weight = smoothstep(u[i]/.7)
                desired_p = (1.-weight)*p + weight*final_p
                difference = Rotation.from_matrix(final_r@r.transpose(0, 2, 1))
                desired_r = (Rotation.from_rotvec(weight*difference.as_rotvec())*Rotation.from_matrix(r)).as_matrix()
                # Preserve clearance during landing, then lock both sole poses.
                gap = (1.-weight)*np.maximum(0., corners[..., 2].min(1))
                desired_p[:, 2] = gap-np.einsum('fij,fvj->fvi', desired_r, self.vertices)[..., 2].min(1)
                tail[i], error = self.fit_feet(pose, desired_p, desired_r)
                pose = tail[i]
                foot_errors.append(error)
            corners = self.geometry(pose)[2]
            correction = max(0., -float(corners[..., 2].min()))
            pose[2] += correction
            lift.append(correction)
        q = np.concatenate((original[:start], tail, np.repeat(tail[-1:], round(fps), axis=0)))
        source_frames = np.r_[np.arange(start), source_t*fps, np.full(round(fps), end)]
        return q, source_frames, dict(edited_from_frame=start, source_end_frame=end,
                                     transition_seconds=2*span/fps, hold_seconds=1.,
                                     feet_locked_from_transition_fraction=.7,
                                     max_foot_ik_error_m=max(foot_errors, default=0.),
                                     max_floor_lift_m=float(max(lift)))

    def validate_tail(self, q, start, fps):
        v = self.velocities(q, fps)
        a = np.gradient(v, 1./fps, axis=0)
        tail = q[start:]
        lower, upper = self.model.jnt_range[1:].T
        excess = max(0., float(np.maximum(lower-tail[:, 7:], tail[:, 7:]-upper).max()))
        speed = float((np.abs(v[start:, 6:])/self.speeds).max())
        floor, collisions = [], []
        for pose in tail:
            floor.append(float(self.geometry(pose)[2][..., 2].min()))
            for contact in self.data.contact:
                bodies = [int(self.model.geom_bodyid[g]) for g in contact.geom]
                # Feet-ground are intentional; penetrating body-body contacts are not.
                if 0 not in bodies and contact.dist < -.003:
                    collisions.append(float(contact.dist))
        # Inverse-dynamics feasibility is a diagnostic over the edited part.
        # The source references have sliding/partial contacts and are not exact dynamics solutions.
        samples = list(range(start, len(q)-round(fps), 5)) + [len(q)-1]
        checks = [self.support(q[i], v[i], a[i]) for i in samples]
        hold_start = len(q)-round(fps)-1
        locked_start = start+int(np.ceil(.7*(hold_start-start)))
        locked_checks = [self.support(q[i], v[i], a[i]) for i in range(locked_start, hold_start+1)]
        final = self.support(q[-1])
        hold_error = float(np.abs(q[-round(fps):]-q[-1]).max())
        hold_speed = float(np.abs(v[-round(fps):]).max())
        record = dict(joint_limit_excess_rad=excess, max_joint_speed_limit_fraction=speed,
                      max_root_speed_m_s=float(np.linalg.norm(v[start:, :3], axis=1).max()),
                      max_root_acceleration_m_s2=float(np.linalg.norm(a[start:, :3], axis=1).max()),
                      min_sole_height_m=min(floor), max_self_penetration_m=-min(collisions, default=0.),
                      hold_pose_error=hold_error, hold_velocity_max=hold_speed,
                      inverse_dynamics_samples=len(checks),
                      inverse_dynamics_passes=sum(c['passed'] for c in checks),
                      planted_settling_frames=len(locked_checks),
                      planted_settling_feasible_frames=sum(c['passed'] for c in locked_checks),
                      final_support=final)
        record['passed'] = (excess < 1e-6 and speed < 1. and min(floor) > -2e-6
                            and not collisions and hold_error < 1e-12 and hold_speed < 1e-9
                            and final['passed'] and final['peak_torque_fraction'] < .85
                            and all(c['passed'] for c in locked_checks)
                            and record['max_root_acceleration_m_s2'] < 12.)
        return record

    def time_scale_active(self, q, source_frames, recipe, factor, fps):
        """Slow the moving part without stretching the required one-second hold."""
        if not np.isfinite(factor) or factor < 1.:
            raise ValueError('Active time scale must be finite and at least one')
        hold_start = len(q)-round(fps)-1
        times = np.arange(hold_start+1, dtype=float)
        samples = np.linspace(0., hold_start, round(hold_start*factor)+1)
        scalar = PchipInterpolator(times, np.c_[q[:hold_start+1, :3], q[:hold_start+1, 7:]], axis=0)(samples)
        quat = RotationSpline(times, Rotation.from_quat(q[:hold_start+1, [4, 5, 6, 3]]))(samples).as_quat()
        active = np.c_[scalar[:, :3], quat[:, [3, 0, 1, 2]], scalar[:, 3:]]
        lift = []
        for pose in active:
            shift = max(0., -float(self.geometry(pose)[2][..., 2].min()))
            pose[2] += shift
            lift.append(shift)
        active[-1] = q[-1]
        new_q = np.r_[active, np.repeat(q[-1:], round(fps), axis=0)]
        new_frames = np.r_[np.interp(samples, times, source_frames[:hold_start+1]),
                           np.full(round(fps), source_frames[-1])]
        recipe = dict(recipe, active_time_scale=factor,
                      edited_from_frame=round(recipe['edited_from_frame']*factor),
                      transition_seconds=recipe['transition_seconds']*factor,
                      max_time_resample_floor_lift_m=max(lift))
        return new_q, new_frames, recipe

    def export(self, path, q, template, fps, source_frames):
        m, d = self.model, self.data
        bids = [m.body(n).id for n in template['body_names'].tolist()]
        joints = [m.joint(n).id for n in template['joint_names'].tolist()]
        positions, rotations = [], []
        for pose in q:
            d.qpos[:] = pose
            mj.mj_kinematics(m, d)
            positions.append(d.xpos[bids].copy())
            rotations.append(d.xmat[bids].reshape(-1, 3, 3).copy())
        positions, rotations = np.array(positions), np.array(rotations)
        quat = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(len(q), len(bids), 4)
        prev, following = np.maximum(np.arange(len(q))-1, 0), np.minimum(np.arange(len(q))+1, len(q)-1)
        relative = rotations[following]@rotations[prev].transpose(0, 1, 3, 2)
        angular = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(len(q), len(bids), 3)
        angular *= (fps/(following-prev))[:, None, None]
        jp = q[:, m.jnt_qposadr[joints]]
        values = dict(fps=np.array([fps]), joint_names=template['joint_names'], body_names=template['body_names'],
                      joint_pos=jp, joint_vel=np.gradient(jp, 1./fps, axis=0), body_pos_w=positions,
                      body_quat_w=quat[..., [3, 0, 1, 2]], body_lin_vel_w=np.gradient(positions, 1./fps, axis=0),
                      body_ang_vel_w=angular, original_frame=source_frames,
                      preparation_note=np.array('Deliberate static endpoint; see manifest for crop, time warp, IK, and validation.'))
        np.savez_compressed(path, **values)


def build(args):
    args.source, args.output, args.work = args.source.resolve(), args.output.resolve(), args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    model_path = build_model(args.work/'model')
    editor = EndingBuilder(model_path)
    source_manifest = json.loads((args.source/'manifest.json').read_text())
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    records, decisions = [], []
    for original in source_manifest['motions']:
        if original.get('mirrored_from'):
            continue
        path = args.source/original['file']
        if args.only and path.stem not in args.only:
            continue
        assert sha(path) == original['sha256']
        template, states, fps = load_reference(path, editor.model)
        q0 = states[:, :editor.model.nq]
        n = len(q0)
        decision = dict(file=path.name, original_frames=n, original_sha256=sha(path), attempts=[])
        first_collision = n
        for i, pose in enumerate(q0):
            editor.geometry(pose)
            if any(0 not in [int(editor.model.geom_bodyid[g]) for g in c.geom] and c.dist < -.003
                   for c in editor.data.contact):
                first_collision = i
                break
        decision['first_source_self_collision_frame'] = first_collision if first_collision < n else None
        # Preserve intentional stationary/toe-supported endings when they already balance.
        stationary = np.max(np.abs(states[-min(10, n):, editor.model.nq:])) < 1e-7
        support = editor.support(q0[-1])
        selected = None
        if stationary and first_collision == n and support['passed'] and support['peak_torque_fraction'] < .85:
            q = np.r_[q0, np.repeat(q0[-1:], round(fps), axis=0)]
            frames = np.r_[np.arange(n), np.full(round(fps), n-1)]
            recipe = dict(method='Existing static ending; append one-second hold', edited_from_frame=n-1,
                          source_end_frame=n-1, transition_seconds=0., hold_seconds=1., max_floor_lift_m=0.)
            check = editor.validate_tail(q, n-1, fps)
            if check['passed']:
                selected = q, frames, recipe, check
        if selected is None:
            # Try the complete source first, then deliberately crop at late supported phases.
            periodic = path.stem in ('13_bounce', '18_stand_to_walk', '20_side_step_left', '21_side_step_right',
                                    '22_advance', '27_crouch_to_walk', '28_stand_to_walk_back',
                                    '32_forward_walk_bk', '33_walk_dg', '34_dance_mm', '35_dance_tr')
            minimum = max(round((.25 if periodic else .6)*n), round(1.5*fps))
            candidates = [n-1]+list(range(n-6, minimum, -10))
            for end in candidates:
                # Endpoint edits cannot repair collisions in the unchanged prefix.
                if end-round(1.2*fps) >= first_collision:
                    continue
                _, _, corners, _ = editor.geometry(q0[end])
                if corners[..., 2].min(1).max() > .03:
                    continue
                target, pose_check = editor.settled_pose(q0[end])
                if not pose_check['passed']:
                    decision['attempts'].append(dict(end_frame=end, pose=pose_check))
                    continue
                for seconds in (.8, 1.2):
                    if end-round(seconds*fps) >= first_collision:
                        continue
                    q, frames, recipe = editor.make_tail(q0, end, target, fps, seconds)
                    recipe.update(method='C2 phase deceleration and grounded endpoint IK', endpoint_ik=pose_check)
                    check = editor.validate_tail(q, recipe['edited_from_frame'], fps)
                    decision['attempts'].append(dict(end_frame=end, source_transition_seconds=seconds, validation=check))
                    if check['passed'] and recipe['max_floor_lift_m'] < .025 and recipe['max_foot_ik_error_m'] < .0005:
                        selected = q, frames, recipe, check
                        break
                if selected is not None:
                    break
        if selected is None:
            decision['status'] = 'excluded_pending_replacement'
            print(f'EXCLUDED {path.stem}', flush=True)
        else:
            q, frames, recipe, check = selected
            destination = args.output/path.name
            editor.export(destination, q, template, fps, frames)
            record = dict(file=path.name, source=str(path.relative_to(ROOT)), source_sha256=sha(path),
                          sha256=sha(destination), frames=len(q), fps=fps, family=original.get('family'),
                          source_provenance=original, ending=recipe, endpoint_validation=check)
            records.append(record)
            decision.update(status='included', output_frames=len(q), ending=recipe, validation=check)
            print(f'INCLUDED {path.stem}: {n} -> {len(q)} frames, source end {recipe["source_end_frame"]}/{n-1}', flush=True)
        decisions.append(decision)
        (args.work/'decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
    manifest = dict(version=1, motions=records, total_frames=sum(r['frames'] for r in records),
                    source_manifest=str(args.source/'manifest.json'), source_manifest_sha256=sha(args.source/'manifest.json'),
                    generator='scripts/prepare_static_motion_endings.py', generator_sha256=GENERATOR_SHA256,
                    model_sha256=sha(model_path), sampling='equal probability per episode reset',
                    purpose='Motion tracking with deliberate, supported one-second static endings',
                    limitation='Endpoint geometry and static support validated; full-motion dynamic tracking is not certified. Inverse-dynamics tail checks are diagnostic, not an acceptance guarantee.',
                    excluded=[r['file'] for r in decisions if r['status'] != 'included'])
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    generate(args.output)
    print(f'Wrote {len(records)} originals plus mirrors to {args.output}', flush=True)


def replay_selection(args):
    """Reproduce the reviewed selection without repeating the candidate search."""
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text())
    args.output, args.work = args.output.resolve(), args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = build_model(args.work/'model')
    editor = EndingBuilder(model_path)
    records = []
    for entry in selection['motions']:
        source = ROOT/entry['input']
        assert sha(source) == entry['input_sha256'], source
        template, states, fps = load_reference(source, editor.model)
        original = states[:, :editor.model.nq]
        end = entry['source_end_frame']
        if entry['method'] == 'preserve_static':
            assert end == len(original)-1
            assert np.max(np.abs(states[-10:, editor.model.nq:])) < 1e-7
            q = np.r_[original, np.repeat(original[-1:], round(fps), axis=0)]
            frames = np.r_[np.arange(len(original)), np.full(round(fps), end)]
            recipe = dict(method='Existing static ending; append one-second hold', edited_from_frame=end,
                          source_end_frame=end, transition_seconds=0., hold_seconds=1., max_floor_lift_m=0.)
        else:
            target, pose_check = editor.settled_pose(original[end])
            assert pose_check['passed'], (entry['file'], pose_check)
            q, frames, recipe = editor.make_tail(original, end, target, fps, entry['source_transition_seconds'])
            recipe.update(method='C2 phase deceleration, grounded foot trajectories and endpoint IK',
                          endpoint_ik=pose_check)
            assert recipe['max_foot_ik_error_m'] < .0005 and recipe['max_floor_lift_m'] < .025
        if entry.get('active_time_scale', 1.) != 1.:
            q, frames, recipe = editor.time_scale_active(q, frames, recipe, entry['active_time_scale'], fps)
        check = editor.validate_tail(q, recipe['edited_from_frame'], fps)
        assert check['passed'], (entry['file'], check)
        # Check the entire exported trajectory, including the unedited prefix.
        max_penetration = 0.
        for pose in q:
            corners = editor.geometry(pose)[2]
            assert corners[..., 2].min() >= -2e-6
            for contact in editor.data.contact:
                if 0 not in [int(editor.model.geom_bodyid[g]) for g in contact.geom]:
                    max_penetration = max(max_penetration, -float(contact.dist))
        assert max_penetration <= .003, (entry['file'], max_penetration)
        check['full_clip_max_self_penetration_m'] = max_penetration
        destination = args.output/entry['file']
        editor.export(destination, q, template, fps, frames)
        records.append(dict(file=entry['file'], source=entry['input'], source_sha256=sha(source),
                            sha256=sha(destination), frames=len(q), fps=fps,
                            source_provenance=entry.get('source_provenance'), replacement_for=entry.get('replacement_for'),
                            ending=recipe, endpoint_validation=check))
        print(f'Validated {entry["file"]}: {len(q)/fps:.2f} s', flush=True)
    manifest = dict(version=1, motions=records, total_frames=sum(r['frames'] for r in records),
                    selection_config=str(selection_path.relative_to(ROOT)), selection_config_sha256=sha(selection_path),
                    generator='scripts/prepare_static_motion_endings.py', generator_sha256=GENERATOR_SHA256,
                    model_sha256=sha(model_path), sampling='equal probability per episode reset',
                    purpose='Deliberately settled references with one-second static endings',
                    excluded=selection['excluded'], replacements=selection.get('replacements', []),
                    limitation='Geometry and bounded-force feasibility checked for the final planted settling phase and static pose. The preceding motion/landing is a kinematic reference, not a certified dynamic rollout or a learned-policy evaluation.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    generate(args.output)
    print(f'Finished {len(records)*2} motion files in {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'retargeted_motion/training_list')
    parser.add_argument('--output', type=Path, default=ROOT/'retargeted_motion/training_static_end')
    parser.add_argument('--work', type=Path, default=ROOT/'outputs/static_end_build_20261125')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--selection', type=Path, help='Replay reviewed source/crop choices from a selection JSON')
    args = parser.parse_args()
    (replay_selection if args.selection else build)(args)
