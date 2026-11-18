"""Integration checks for the T1 import/export boundary, in the DSMS environment."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
pytest.importorskip('cyipopt')
pytest.importorskip('pxr')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dsms_t1_model import ROOT, build_model, load_reference, actuator_limits
from retarget_force2_dsms import export_motion, replay_metrics, physical_context, physical_sample, optimize_rollout


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    return mujoco.MjModel.from_xml_path(str(build_model(tmp_path_factory.mktemp('dsms_model'))))


def test_training_pose_and_body_velocity_roundtrip(model, tmp_path):
    template, states, fps = load_reference(ROOT / 'retargeted_motion/training_list/21_side_step_right.npz', model)
    selected = states[[0, 40, 170]].copy()
    path = tmp_path / 'roundtrip.npz'
    export_motion(path, model, selected, template, fps)
    out = dict(np.load(path))
    np.testing.assert_allclose(out['joint_pos'], template['joint_pos'][[0, 40, 170]], atol=1e-12)
    np.testing.assert_allclose(out['body_pos_w'], template['body_pos_w'][[0, 40, 170]], atol=2e-6)
    # Velocities must be at body origins, not at the displaced body COMs.
    d = mujoco.MjData(model)
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in template['body_names'].tolist()]
    eps = 1e-6
    for i, x in enumerate(selected):
        positions = []
        for dt in (-eps, eps):
            d.qpos[:] = x[:model.nq]
            mujoco.mj_integratePos(model, d.qpos, x[model.nq:], dt)
            mujoco.mj_kinematics(model, d)
            positions.append(d.xpos[bids].copy())
        np.testing.assert_allclose(out['body_lin_vel_w'][i], (positions[1] - positions[0]) / (2 * eps), atol=2e-8)


def test_independent_replay_rejects_stitched_or_modified_states(model):
    _, reference, _ = load_reference(ROOT / 'retargeted_motion/training_list/03_standing.npz', model)
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = reference[0, :model.nq], reference[0, model.nq:]
    u = d.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].copy()
    states = [reference[0].copy()]
    ctx = physical_context(model, actuator_limits())
    samples = []
    for _ in range(8):
        d.ctrl[:] = u
        mujoco.mj_step(model, d)
        states.append(np.r_[d.qpos, d.qvel])
        samples.append(physical_sample(model, d, ctx))
    states = np.array(states)
    controls = np.tile(u, (8, 1))
    metrics = replay_metrics(model, states, controls, reference, actuator_limits())
    assert metrics['replay_max_state_error'] == 0.
    assert metrics['max_torque_limit_fraction'] <= 1.000001
    for key in samples[0]:
        aggregate = sum if key == 'torso_head_floor_contact_samples' else max
        assert aggregate(s[key] for s in samples) == metrics[key]
    altered = states.copy()
    altered[4, 0] += .01
    assert replay_metrics(model, altered, controls, reference, actuator_limits())['replay_max_state_error'] >= .0099


def rollout_stub(status=-1):
    """A finite optimizer proposal with deliberately poor tracking/constraint error."""
    return SimpleNamespace(
        n_steps=5, dt=.02, steps_per_solve=2,
        problem=SimpleNamespace(add_option=lambda *args: None),
        dyn=SimpleNamespace(nu=1, u_lb=np.array([-1.]), u_ub=np.array([1.]),
                            dynamics=lambda x, u: x.copy()),
        spline=SimpleNamespace(fit=lambda u: u),
        pack=lambda x, u: x,
        constraints=lambda x: np.array([.4]),
        solve_step=lambda x: (np.zeros(1), x, np.zeros((2, 1)), dict(status=status, obj_val=100.)),
        _ref=lambda t: np.array([0., 0., .6, 1., 0., 0., 0.]),
    )


@pytest.mark.parametrize('status', [-1, 2])
def test_full_duration_despite_bad_tracking_and_unconverged_solver(status):
    mpc = rollout_stub(status)
    # Low, upside down, and over two metres from the reference. These used to
    # trigger the driver's root-error, torso-height, and orientation guards.
    x = np.array([2., 0., .1, 0., 1., 0., 0.])
    progress = []
    states, controls, solves, failure = optimize_rollout(
        mpc, x, 80, 8, lambda steps, *args: progress.append(steps))
    assert failure is None
    assert len(states) == 6 and len(controls) == 5
    assert progress == [2, 4, 5]  # Include the final partial replan interval.
    assert [s['status'] for s in solves] == [status] * 3
    assert all(s['max_constraint_residual'] == .4 for s in solves)


@pytest.mark.parametrize('problem, expected', [
    ('control_nan', 'nonfinite_optimizer_control'),
    ('state_nan', 'nonfinite_simulation'),
    ('out_of_bounds', 'control_bound_violation'),
    ('empty_control', 'empty_optimizer_control'),
])
def test_unusable_numerical_results_are_not_padded_into_full_clips(problem, expected):
    mpc = rollout_stub()
    if problem == 'state_nan':
        mpc.dyn.dynamics = lambda x, u: np.full_like(x, np.nan)
    else:
        u = {'control_nan': np.array([[np.nan]]), 'out_of_bounds': np.array([[2.]]),
             'empty_control': np.empty((0, 1))}[problem]
        mpc.solve_step = lambda x: (np.zeros(1), x, u, dict(status=-1, obj_val=100.))
    states, controls, _, failure = optimize_rollout(mpc, np.zeros(7), 80, 8, lambda *args: None)
    assert failure == expected
    assert len(states) == 1 and len(controls) == 0
