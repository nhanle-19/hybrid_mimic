"""Physical residual tests on the USD-derived T1, independent of Isaac Sim."""
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_acc_model import WBCACCModel, FEET
from wbc_acc_qp import WBCACCQP, Contact, ExternalWrench, MotionTask, friction_rays
from wbc_acc_control import force_diagnostics, reference_contact_schedule
from wbc_acc_control import build_reference_tasks, swing_tasks
from types import SimpleNamespace


@pytest.fixture
def model():
    return WBCACCModel()


def state_at_rest(model):
    q = pin.neutral(model.model); q[2] = .72
    return model.state(q, np.zeros(model.model.nv))


def test_no_swing_task_retains_posture_and_pelvis(model):
    state = state_at_rest(model)
    _, tasks = build_reference_tasks(state, state, np.zeros(2))
    assert [task.name for task in tasks] == ['posture', 'pelvis_orientation']
    assert swing_tasks(state, state, ()) == []
    cfg = SimpleNamespace(swing_weight=0., posture_weight=.1, pelvis_weight=5.)
    assert swing_tasks(state, state, (), cfg) == []
    # An explicit positive weight remains available for controlled comparisons.
    cfg.swing_weight = 10.
    assert len(swing_tasks(state, state, (), cfg)) == 2


@pytest.mark.parametrize('seed', range(5))
def test_centroidal_identity_and_bias(model, seed):
    rng = np.random.default_rng(seed)
    q = pin.integrate(model.model, pin.neutral(model.model), rng.normal(0, .1, model.model.nv))
    v = rng.normal(0, .2, model.model.nv)
    s = model.state(q, v)
    np.testing.assert_allclose(s['momentum'], s['Ag']@v, atol=1e-10)
    acceleration = rng.normal(0, .3, model.model.nv)
    np.testing.assert_allclose(pin.rnea(model.model, model.data, q, v, acceleration),
                               s['M']@acceleration+s['bias'], atol=1e-10)
    rate = pin.computeCentroidalMomentumTimeVariation(model.model, model.data, q, v, acceleration).vector.copy()
    np.testing.assert_allclose(rate, s['Ag']@acceleration+s['Ag_bias'], atol=1e-10)
    eps = 1e-6
    plus = model.state(pin.integrate(model.model, q, eps*v), v)
    minus = model.state(pin.integrate(model.model, q, -eps*v), v)
    np.testing.assert_allclose((plus['Ag']-minus['Ag'])@v/(2*eps), s['Ag_bias'], atol=1e-7)
    for foot in FEET:
        np.testing.assert_allclose((plus['frames'][foot]['J']-minus['frames'][foot]['J'])@v/(2*eps),
                                   s['frames'][foot]['bias'], atol=1e-7)


@pytest.mark.parametrize('active', [(), FEET[:1], FEET[1:], FEET])
def test_all_eight_physical_invariants(model, active):
    state = state_at_rest(model)
    limits = np.full(23, 60.)
    result = WBCACCQP(limits).solve(state, np.zeros(6), [Contact(f) for f in active])
    assert result['variable_count'] == 29+16*len(active)
    assert max(result['metrics'].values()) < 2e-5
    # Independent RNEA, compared with physical point forces and commanded torque.
    generalized_contact = np.zeros(29)
    for foot, forces in result['contact_forces'].items():
        frame = state['frames'][foot]
        wrench = np.zeros(6)
        for point, force in forces:
            wrench += np.r_[force, np.cross(point-frame['position'], force)]
        generalized_contact += frame['J'].T@wrench
    inverse = pin.rnea(model.model, model.data, state['q'], state['v'], result['acceleration'])
    np.testing.assert_allclose(inverse-generalized_contact, np.r_[np.zeros(6), result['torque']], atol=2e-5)
    diag = force_diagnostics(result, state, limits)
    for i, foot in enumerate(FEET):
        if foot not in active:
            assert np.count_nonzero(diag['point_forces'][i]) == 0
        else:
            assert diag['normal_forces'][i].min() >= -1e-5
            assert np.max(diag['tangential_forces'][i]-.6*diag['normal_forces'][i]) < 1e-5
            assert -.09001 <= diag['cop'][i, 0] <= .10001
            assert abs(diag['cop'][i, 1]) <= .04001
    assert diag['torque_utilization'].max() <= 1+1e-6
    if not active:
        np.testing.assert_allclose(result['rate'][:3], state['mass']*state['gravity'], atol=2e-5)


def test_moving_stance_and_external_wrench(model):
    q = pin.neutral(model.model); q[2] = .72
    for name in ('Left_Knee_Pitch', 'Right_Knee_Pitch'):
        q[model.model.joints[model.model.getJointId(name)].idx_q] = .2
    state = model.state(q, np.zeros(29))
    jac = np.vstack([state['frames'][f]['J'] for f in FEET])
    v = np.random.default_rng(9).normal(0, .05, 29)
    v -= np.linalg.pinv(jac)@(jac@v)
    state = model.state(q, v)
    result = WBCACCQP(np.full(23, 60.)).solve(state, np.zeros(6), [Contact(f) for f in FEET],
                   external=[ExternalWrench('Trunk', np.array([8., 0, 0, 0, 0, 0]))])
    assert max(result['metrics'].values()) < 2e-5
    np.testing.assert_allclose(result['external_centroidal_wrench'][:3], [8, 0, 0])


def test_nonvertical_friction_basis():
    normal = np.array([.2, .1, 1.]); normal /= np.linalg.norm(normal)
    rays = friction_rays(normal, .5)
    for rho in np.random.default_rng(7).uniform(0, 20, (20, 4)):
        f = rays@rho; fn = normal@f
        assert fn >= 0
        assert np.linalg.norm(f-fn*normal) <= .5*fn+1e-12


def test_impossible_hard_task_is_rejected_not_clipped(model):
    state = state_at_rest(model)
    with pytest.raises(RuntimeError, match='WBCACC QP'):
        WBCACCQP(np.full(23, 60.)).solve(state, np.zeros(6), [],
            [MotionTask(np.eye(29), np.zeros(29), 1., hard=True)])


def test_torque_limits_inside_qp(model):
    state = state_at_rest(model)
    limits = np.full(23, 1.)
    result = WBCACCQP(limits).solve(state, np.array([1000., 0, 0, 0, 0, 0]), [Contact(f) for f in FEET])
    assert np.max(np.abs(result['torque'])) <= 1+2e-5
    assert result['metrics']['base_inverse_dynamics'] < 2e-5


def test_explicit_schedule_is_not_policy_weight(model):
    state = state_at_rest(model)
    schedule = reference_contact_schedule([state])
    np.testing.assert_array_equal(schedule, [[True, True]])
    raised = model.state(state['q'], state['v'])
    raised['frames'][FEET[1]]['position'][2] += .2
    np.testing.assert_array_equal(reference_contact_schedule([state, raised]), [[True, True], [True, False]])


def test_simulator_float32_quaternion_is_normalized(model):
    rng = np.random.default_rng(45)
    q = pin.integrate(model.model, pin.neutral(model.model), rng.normal(0, .1, 29)).astype(np.float32)
    q[3:7] *= 1.00001
    v = rng.normal(0, .1, 29).astype(np.float32)
    state = model.state(q, v)
    np.testing.assert_allclose(state['momentum'], state['Ag']@state['v'], atol=1e-10)
    result = WBCACCQP(np.full(23, 60.)).solve(state, np.zeros(6), [])
    assert result['metrics']['base_inverse_dynamics'] < 2e-5


def test_tilted_reference_keeps_low_foot_in_contact(model):
    state = state_at_rest(model)
    state['frames'][FEET[0]]['rotation'] = pin.exp3(np.array([0., .3, 0.]))
    state['frames'][FEET[1]]['position'][2] += .3
    np.testing.assert_array_equal(reference_contact_schedule([state]*10), [[True, False]]*10)


def test_policy_only_changes_contact_weights_not_reference_feedback(model):
    from wbc_acc_control import contact_weights
    state = state_at_rest(model)
    q = state['q'].copy(); q[0] += .1; q[7] += .2
    reference = model.state(q, np.zeros(29))
    rate, tasks = build_reference_tasks(state, reference, np.zeros(2))
    assert np.linalg.norm(rate) > 0
    changed_rate, changed_tasks = build_reference_tasks(state, reference, np.array([-10., 10.]))
    np.testing.assert_array_equal(changed_rate, rate)
    for before, after in zip(tasks, changed_tasks):
        np.testing.assert_array_equal(before.target_minus_bias, after.target_minus_bias)
        np.testing.assert_array_equal(before.jacobian, after.jacobian)
    np.testing.assert_allclose(contact_weights([0., 0.]), [50.05, 50.05])
    np.testing.assert_allclose(contact_weights([-1e6, 1e6]), [.1, 100.])
    with pytest.raises(ValueError, match='two finite'):
        build_reference_tasks(state, reference, np.zeros(29))
    with pytest.raises(ValueError, match='two finite'):
        contact_weights([float('nan'), 0.])
    with pytest.raises(ValueError, match='bounds'):
        contact_weights([0., 0.], SimpleNamespace(contact_weight_min=10., contact_weight_max=1.))


def test_tangential_weights_have_effect_without_new_variables(model):
    state = state_at_rest(model)
    limits = np.full(23, 60.)
    contact = Contact(FEET[0])
    desired = np.array([100., 40., 0., 0., 0., 0.])
    qp = WBCACCQP(limits)
    low = qp.solve(state, desired, [contact], contact_weights={FEET[0]: .1})
    high = qp.solve(state, desired, [contact], contact_weights={FEET[0]: 100.})
    hard = qp.solve(state, desired, [contact])
    low_acc = low['contact_acceleration'][FEET[0]]
    high_acc = high['contact_acceleration'][FEET[0]]
    assert np.linalg.norm(low_acc[:2]) > 1e-3
    assert np.linalg.norm(high_acc[:2]) < np.linalg.norm(low_acc[:2])*.9
    for result in (low, high, hard):
        assert result['variable_count'] == 45
        assert max(result['metrics'].values()) < 2e-5
        np.testing.assert_allclose(result['contact_acceleration'][FEET[0]][2:], 0., atol=2e-5)
    np.testing.assert_allclose(hard['contact_acceleration'][FEET[0]], 0., atol=2e-5)
