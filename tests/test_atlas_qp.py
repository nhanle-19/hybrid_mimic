"""Physical residual tests on the USD-derived T1, independent of Isaac Sim."""
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from atlas_model import AtlasModel, FEET
from atlas_qp import AtlasQP, Contact, ExternalWrench, MotionTask, friction_rays
from atlas_control import force_diagnostics, reference_contact_schedule


@pytest.fixture
def model():
    return AtlasModel()


def state_at_rest(model):
    q = pin.neutral(model.model); q[2] = .72
    return model.state(q, np.zeros(model.model.nv))


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
    result = AtlasQP(limits).solve(state, np.zeros(6), [Contact(f) for f in active])
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
    result = AtlasQP(np.full(23, 60.)).solve(state, np.zeros(6), [Contact(f) for f in FEET],
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
    with pytest.raises(RuntimeError, match='Atlas QP'):
        AtlasQP(np.full(23, 60.)).solve(state, np.zeros(6), [],
            [MotionTask(np.eye(29), np.zeros(29), 1., hard=True)])


def test_torque_limits_inside_qp(model):
    state = state_at_rest(model)
    limits = np.full(23, 1.)
    result = AtlasQP(limits).solve(state, np.array([1000., 0, 0, 0, 0, 0]), [Contact(f) for f in FEET])
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
    result = AtlasQP(np.full(23, 60.)).solve(state, np.zeros(6), [])
    assert result['metrics']['base_inverse_dynamics'] < 2e-5


def test_tilted_reference_keeps_low_foot_in_contact(model):
    state = state_at_rest(model)
    state['frames'][FEET[0]]['rotation'] = pin.exp3(np.array([0., .3, 0.]))
    state['frames'][FEET[1]]['position'][2] += .3
    np.testing.assert_array_equal(reference_contact_schedule([state]*10), [[True, False]]*10)


def hybrid_objective(model):
    return dict(base_body='Trunk', base_weight=.001,
                contact_weights={f: .001 for f in FEET}, angular_force_scale=20.,
                torque_reference=np.zeros(23), torque_weights=np.ones(23))


def test_hybrid_torque_reference_and_base_wrench(model):
    """Hybrid mode must use policy torque references, not silently drop them."""
    state = state_at_rest(model)
    solver = AtlasQP(np.full(23, 60.))
    objective = hybrid_objective(model)
    baseline = solver.solve(state, np.zeros(6), [], hybrid=objective)
    objective['torque_reference'] = np.linspace(-5., 5., 23)
    changed = solver.solve(state, np.zeros(6), [], hybrid=objective)
    assert changed['variable_count'] == 35
    assert np.linalg.norm(changed['torque']-baseline['torque']) > 1.
    assert max(changed['metrics'].values()) < 2e-5
    # Auxiliary support is an optimization output, explicitly accounted for.
    assert np.linalg.norm(changed['auxiliary_base_wrench']) > 1.
    generalized = state['frames']['Trunk']['J'].T@changed['auxiliary_base_wrench']
    inverse = pin.rnea(model.model, model.data, state['q'], state['v'], changed['acceleration'])
    np.testing.assert_allclose(inverse-generalized, np.r_[np.zeros(6), changed['torque']], atol=2e-5)
    objective['base_weight'] = 100.
    costly_base = solver.solve(state, np.zeros(6), [], hybrid=objective)
    assert np.linalg.norm(costly_base['auxiliary_base_wrench']) < np.linalg.norm(changed['auxiliary_base_wrench'])


@pytest.mark.parametrize('body', ['left_foot_link', 'right_foot_link', 'left_hand_link', 'right_hand_link'])
def test_hybrid_contact_logits_affect_physical_force_cost(model, body):
    state = state_at_rest(model)
    solver = AtlasQP(np.full(23, 60.))
    contact = Contact(body) if body in FEET else Contact(body, points=np.zeros((1, 3)), constrain_rotation=False)
    objective = hybrid_objective(model)
    objective['base_weight'] = 1.
    objective['contact_weights'] = {body: .00001}
    cheap = solver.solve(state, np.zeros(6), [contact], hybrid=objective)
    objective['contact_weights'][body] = 100.
    expensive = solver.solve(state, np.zeros(6), [contact], hybrid=objective)
    def force_norm(result):
        return np.linalg.norm(sum(f for _, f in result['contact_forces'][body]))
    assert force_norm(expensive) < force_norm(cheap)
    assert max(expensive['metrics'].values()) < 2e-5
    # Feet diagnostics remain valid when a hand point is also represented.
    force_diagnostics(expensive, state, solver.torque_limits)


def test_hybrid_limits_apply_to_pd_plus_feedforward(model):
    state = state_at_rest(model)
    limits = np.full(23, 60.)
    pd = np.linspace(-180., 180., 23)
    result = AtlasQP(limits).solve(state, np.zeros(6), [],
        hybrid=hybrid_objective(model), pd_torque=pd)
    total = pd+result['torque']
    np.testing.assert_allclose(total, result['total_torque'], atol=1e-10)
    assert np.max(np.abs(total)-limits) <= 2e-5
    # FF must be allowed to counteract an already over-limit PD command.
    assert np.max(np.abs(result['torque'])) > limits.max()
    generalized = state['frames']['Trunk']['J'].T@result['auxiliary_base_wrench']
    inverse = pin.rnea(model.model, model.data, state['q'], state['v'], result['acceleration'])
    np.testing.assert_allclose(inverse-generalized, np.r_[np.zeros(6), total], atol=2e-5)
    diag = force_diagnostics(result, state, limits)
    np.testing.assert_allclose(diag['torque_utilization'], np.abs(total)/limits)


@pytest.mark.parametrize('pd', [np.zeros(22), np.full(23, np.nan)])
def test_reject_invalid_pd_torque(model, pd):
    with pytest.raises(ValueError, match='PD torque'):
        AtlasQP(np.full(23, 60.)).solve(state_at_rest(model), np.zeros(6), [], pd_torque=pd)


def test_hybrid_balance_has_no_joint_posture_task(model):
    from atlas_control import hybrid_balance_tasks
    state = state_at_rest(model)
    linear, angular = np.array([.2, -.1, .3]), np.array([.1, .2, -.3])
    rate, tasks = hybrid_balance_tasks(state, linear, angular, 200.)
    assert len(tasks) == 1 and tasks[0].name == 'base'
    np.testing.assert_allclose(tasks[0].jacobian[:, 6:], 0., atol=1e-12)
    nominal = np.zeros(29)
    base = state['frames']['Trunk']
    nominal[:6] = np.linalg.solve(base['J'][:, :6], np.r_[linear, angular]-base['bias'])
    np.testing.assert_allclose(rate, state['Ag']@nominal+state['Ag_bias'])
