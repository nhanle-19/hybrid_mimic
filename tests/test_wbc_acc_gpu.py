"""Numerical parity of the training QP with the independent Pinocchio/OSQP path."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from wbc_acc_model import WBCACCModel, FEET
from wbc_acc_qp import WBCACCQP, Contact
from wbc_acc_control import build_reference_tasks, swing_tasks, contact_weights
from wbc_acc_torch_model import WBCACCTorchModel
from wbc_acc_gpu_control import WBCACCGPUQP, rotation_log


@pytest.fixture
def cfg():
    # Read the actual controller defaults without loading Isaac Sim.
    import ast
    path = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/controller_cfg.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    return SimpleNamespace(**{n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)})


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA driver unavailable'))]


def standing_state(model, count=1):
    from wbc_acc_contact_policy import sole_vertices
    q = torch.zeros((count, 30), device=model.device, dtype=torch.float64)
    q[:, 6], q[:, 2] = 1., .72
    v = torch.zeros((count, 29), device=model.device, dtype=torch.float64)
    s = model.state(q, v)
    q[:, 2] -= s['frames'][FEET[0]]['position'][:, 2]+sole_vertices(model.device)[0, :, 2].min()
    s = model.state(q, v)
    for frame in s['frames'].values():
        frame['acceleration'] = torch.zeros_like(frame['velocity'])
    return s


def support_actions(count=1, device='cpu'):
    a = torch.zeros((count, 14), dtype=torch.float64, device=device)
    a[:, 0], a[:, 7] = 1., 1.
    return a


@pytest.mark.parametrize('device', DEVICES)
def test_standing_zero_support_and_physical_invariants(cfg, device):
    from wbc_acc_contact_policy import available_vertices
    from wbc_acc_torch_model import mv
    model = WBCACCTorchModel(device)
    s = standing_state(model, 4)
    a = support_actions(4, device)
    a[1, 0], a[2, 7], a[3, 0], a[3, 7] = 0., 0., 0., 0.
    limits = torch.full((4, 23), 60., dtype=torch.float64, device=device)
    masks = available_vertices(s, cfg)
    assert masks.all()
    result = WBCACCGPUQP(cfg).solve(s, s, a, masks, limits)
    assert not result['failed'].any()
    assert result['residual'].max() < 2e-5
    assert result['wrenches'][1, 0].count_nonzero() == 0
    assert result['wrenches'][2, 1].count_nonzero() == 0
    assert result['wrenches'][3].count_nonzero() == 0
    # Independent inverse dynamics reconstructed from resultant body wrenches.
    generalized = sum(mv(s['frames'][foot]['J'].transpose(-1, -2), result['wrenches'][:, j]) for j, foot in enumerate(FEET))
    inverse = mv(s['M'], result['acceleration'])+s['bias']-generalized
    torch.testing.assert_close(inverse[:, :6], torch.zeros_like(inverse[:, :6]), atol=2e-5, rtol=0)
    torch.testing.assert_close(inverse[:, 6:], result['torque'], atol=2e-5, rtol=0)
    assert (result['torque'].abs() <= limits+2e-5).all()
    assert result['contact_acceleration'][0].abs().max() < .01
    np.testing.assert_allclose(result['wrenches'][0, :, 2].sum().cpu(), float(s['mass'])*9.81, atol=.1)
    for j in range(2):
        force = result['wrenches'][:, j, :3]
        assert (force[:, :2].norm(dim=-1) <= .6*force[:, 2]+2e-5).all()
        assert (force[:, 2] <= a[:, j*7]*cfg.contact_force_max[j]+2e-5).all()
    expected = WBCACCGPUQP(cfg, backend='osqp').solve(s, s, a, masks, limits)
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'], expected['acceleration'], atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize('device', DEVICES)
def test_1024_environment_batch(cfg, device):
    from wbc_acc_contact_policy import available_vertices
    s = standing_state(WBCACCTorchModel(device), 1024)
    a = support_actions(1024, device)
    limits = torch.full((1024, 23), 60., dtype=torch.float64, device=device)
    r = WBCACCGPUQP(cfg).solve(s, s, a, available_vertices(s, cfg), limits)
    assert not r['failed'].any()
    assert r['torque'].device == torch.device(device)
    torch.testing.assert_close(r['torque'][:1].expand_as(r['torque']), r['torque'])


def test_all_six_weights_change_motion_requests(cfg):
    import copy
    from wbc_acc_contact_policy import available_vertices
    s = standing_state(WBCACCTorchModel(), 2)
    ref = copy.deepcopy(s)
    ref['frames'][FEET[0]]['velocity'][:] = torch.tensor([.05, .03, .04, .02, .03, .04])
    a = support_actions(2)
    a[0, 1:7], a[1, 1:7] = -30., 30.
    r = WBCACCGPUQP(cfg).solve(s, ref, a, available_vertices(s, cfg), torch.full((2, 23), 60., dtype=torch.float64))
    assert not r['failed'].any()
    target = ref['frames'][FEET[0]]['velocity'][0]*torch.tensor(cfg.contact_velocity_gain)
    assert (r['contact_acceleration'][1, 0]-target).norm() < (r['contact_acceleration'][0, 0]-target).norm()
    assert r['contact_acceleration'][1, 0, 2:].abs().max() > .01  # No leftover normal/angular stationary equality.


def test_geometry_restricts_edge_support_and_unavailable_foot(cfg):
    from wbc_acc_contact_policy import available_vertices, sole_vertices
    from wbc_acc_torch_model import rotation_xyzw
    m = WBCACCTorchModel();s = standing_state(m)
    # Whole robot pitch puts only the actual low edges on the plane.
    q = s['q'].clone();q[:, 4] = np.sin(.15); q[:, 6] = np.cos(.15)
    s = m.state(q, s['v'])
    vertices = sole_vertices()
    f = s['frames'][FEET[0]]
    low = (vertices[0]@f['rotation'].transpose(-1, -2)+f['position'][:, None])[..., 2].min()
    q[:, 2] -= low
    s = m.state(q, s['v'])
    for f in s['frames'].values(): f['acceleration'] = torch.zeros_like(f['velocity'])
    mask = available_vertices(s, cfg)
    assert mask[0, 0].sum() == 2
    r = WBCACCGPUQP(cfg).solve(s, s, support_actions(), mask, torch.full((1, 23), 60., dtype=torch.float64))
    assert not r['failed'].any()
    rho = r['rho'].reshape(1, 2, 4, 4)
    assert rho[~mask].count_nonzero() == 0
    q[:, 2] += .2
    airborne = m.state(q, s['v'])
    assert not available_vertices(airborne, cfg).any()


def test_solver_failure_is_flagged(cfg):
    from wbc_acc_contact_policy import available_vertices
    s = standing_state(WBCACCTorchModel())
    r = WBCACCGPUQP(cfg, max_iterations=1).solve(s, s, support_actions(), available_vertices(s, cfg), torch.full((1, 23), 60., dtype=torch.float64))
    assert r['failed'].all()
    assert torch.isfinite(r['torque']).all()


@pytest.mark.parametrize('device', DEVICES)
def test_all_contact_patterns_share_one_fixed_batch(cfg, device, monkeypatch):
    import wbc_acc_gpu_control as control
    n = 256
    s = standing_state(WBCACCTorchModel(device), n)
    a = support_actions(n, device)
    # All eight-corner availability patterns, including no support, a single
    # corner, an edge, a full foot, and both feet. Activations can disable a
    # geometrically available foot independently.
    masks = ((torch.arange(n, device=device)[:, None] >> torch.arange(8, device=device)) & 1).bool().reshape(n, 2, 4)
    a[::7, 0] = 0.
    a[::11, 7] = 0.
    limits = torch.full((n, 23), 60., dtype=torch.float64, device=device)
    calls = []
    solve = control.solve_batched_qp
    def counted(h, g, matrix, bound, **kwargs):
        calls.append((h.shape, matrix.shape, h.device))
        return solve(h, g, matrix, bound, **kwargs)
    monkeypatch.setattr(control, 'solve_batched_qp', counted)
    result = WBCACCGPUQP(cfg).solve(s, s, a, masks, limits)
    assert calls == [(torch.Size([n, 55, 55]), torch.Size([n, 80, 55]), torch.device(device))]
    assert not result['failed'].any()
    assert result['residual'].max() < 2e-5
    assert result['rho'].reshape(n, 2, 4, 4)[~result['available']].count_nonzero() == 0
    assert (result['torque'].abs() <= limits+2e-5).all()
    # Cross-check representative patterns with a separate optimization solver.
    ids = torch.tensor([0, 1, 3, 15, 16, 51, 127, 255], device=device)
    subset = control.select_state(s, ids)
    expected = WBCACCGPUQP(cfg, backend='osqp').solve(subset, subset, a[ids], masks[ids], limits[ids])
    assert not expected['failed'].any()
    torch.testing.assert_close(result['acceleration'][ids], expected['acceleration'], atol=2e-3, rtol=1e-3)
    # Force sharing has only 1e-5 objective regularization; solvers agree much
    # more closely on acceleration than on the redundant friction-ray split.
    torch.testing.assert_close(result['torque'][ids], expected['torque'], atol=.01, rtol=1e-3)
    torch.testing.assert_close(result['wrenches'][ids], expected['wrenches'], atol=.02, rtol=1e-3)


def test_fixed_batch_keeps_failed_environment_isolated(cfg):
    from wbc_acc_contact_policy import available_vertices
    s = standing_state(WBCACCTorchModel(), 3)
    # A bad model momentum identity must invalidate only that environment.
    s['momentum'][1, 0] += 1.
    result = WBCACCGPUQP(cfg).solve(s, s, support_actions(3), available_vertices(s, cfg),
                                  torch.full((3, 23), 60., dtype=torch.float64))
    assert result['failed'].tolist() == [False, True, False]
    assert result['torque'][1].count_nonzero() == 0
    torch.testing.assert_close(result['torque'][0], result['torque'][2])


def test_rotation_log_matches_pinocchio():
    axis = np.array([1., 2., 3.])/np.sqrt(14)
    for angle in (0., 1e-10, .5, np.pi-1e-8):
        r = pin.exp3(axis*angle)
        np.testing.assert_allclose(rotation_log(torch.tensor(r)).numpy(), pin.log3(r), atol=1e-7)


@pytest.fixture
def action_modules(monkeypatch):
    """Load the actual action routing without starting Isaac Sim."""
    import importlib.util
    from types import ModuleType
    root = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking'
    managers = ModuleType('isaaclab.managers')
    class ActionTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self._asset = env.scene[cfg.asset_name]
    managers.ActionTerm, managers.ActionTermCfg = ActionTerm, type('ActionTermCfg', (), {})
    utils = ModuleType('isaaclab.utils'); utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', managers)
    monkeypatch.setitem(sys.modules, 'isaaclab.utils', utils)
    for name, module in [('wbc_acc_torch_model', sys.modules['wbc_acc_torch_model']),
                         ('wbc_acc_gpu_control', sys.modules['wbc_acc_gpu_control']),
                         ('wbc_acc_contact_policy', sys.modules['wbc_acc_contact_policy'])]:
        monkeypatch.setitem(sys.modules, 'whole_body_tracking.utils.'+name, module)
    loaded = []
    for name in ('controller_cfg', 'wbc_acc_action', 'wbc_acc_gpu_action'):
        spec = importlib.util.spec_from_file_location('wbc_acc_gpu_test.'+name, root/'tasks/tracking/config/t1_wbc_acc'/f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[1:]


def test_backend_dispatch_and_cuda_required(action_modules, monkeypatch):
    action, gpu = action_modules
    cfg = action.WBCACCActionCfg()
    assert cfg.backend == 'batched'
    with pytest.raises(ValueError, match='requires --device cuda'):
        action.WBCACCAction(cfg, SimpleNamespace(device='cpu', scene={'robot': None}))
    initialized = []
    monkeypatch.setattr(gpu.WBCACCGPUAction, '__init__', lambda self, cfg, env: initialized.append(cfg.backend))
    term = action.WBCACCAction(cfg, None)
    assert type(term) is gpu.WBCACCGPUAction
    assert initialized == ['batched']
    cfg.backend = 'osqp'
    assert type(action.WBCACCAction.__new__(action.WBCACCAction, cfg, None)) is gpu.WBCACCGPUAction
    cfg.backend = 'typo'
    with pytest.raises(ValueError, match='backend must be'):
        action.WBCACCAction(cfg, None)


def test_gpu_action_reset_and_ground_contacts(action_modules):
    _, gpu = action_modules
    term = object.__new__(gpu.WBCACCGPUAction)
    term._env = SimpleNamespace(num_envs=2, device='cpu')
    term.cfg = SimpleNamespace(record_diagnostics=True)
    term.estimated_contacts = torch.ones((2, 2), dtype=torch.bool)
    term.qp_failed = torch.ones(2, dtype=torch.bool)
    term.external = torch.ones((2, 24, 6))
    term._raw = torch.ones((2, 14))
    term._torques = torch.ones((2, 23))
    term._qp_update_pending = False
    term.pending = [{'sample': 0}, {'sample': 1}]
    term.last_prediction = None
    term.set_active_contacts([[True, False], [False, True]])
    assert term.contact_override.device.type == 'cpu'
    term.reset(torch.tensor([1]))
    assert term.qp_failed.tolist() == [True, False]
    assert term._torques[0].all() and not term._torques[1].any()
    assert term._qp_update_pending
    assert term.estimated_contacts[0].all() and not term.estimated_contacts[1].any()
    assert term.external[0].all() and not term.external[1].any()
    assert term._raw[0].all() and not term._raw[1].any()
    assert term.pending == [{'sample': 0}, None]
    sensors = {name: SimpleNamespace(body_names=[foot], data=SimpleNamespace(force_matrix_w=torch.ones((2, 1, 1, 3))))
               for name, foot in zip(('left_foot_ground_contact', 'right_foot_ground_contact'), FEET)}
    term._env.scene = SimpleNamespace(sensors=sensors)
    assert term._gpu_ground_forces().shape == (2, 2, 3)
    sensors['left_foot_ground_contact'].data.force_matrix_w = torch.zeros((2, 1, 2, 3))
    with pytest.raises(RuntimeError, match='one foot and one ground filter'):
        term._gpu_ground_forces()


def test_action_applies_bounded_fallback_and_logs_failure(action_modules, cfg):
    from wbc_acc_contact_policy import available_vertices
    _, gpu = action_modules
    term = object.__new__(gpu.WBCACCGPUAction)
    term.cfg = SimpleNamespace(record_diagnostics=False, controller=cfg)
    state = standing_state(WBCACCTorchModel())
    state['v'][:, 6:] = 100.
    term._env = SimpleNamespace(num_envs=1, device='cpu', physics_dt=.002, cfg=SimpleNamespace(decimation=10),
        extras={}, scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(time_steps=torch.zeros(1, dtype=torch.long))))
    term.references = state
    term._states = lambda: state
    term.validated = True
    term._raw = support_actions()
    term._torques = torch.zeros((1, 23))
    term._gpu_ground_forces = lambda: torch.zeros((1, 2, 3))
    term._actual_wrenches = lambda: torch.zeros((1, 2, 6))
    term.estimated_contacts = torch.zeros((1, 2), dtype=torch.bool)
    term.contact_override = None
    term.limits = torch.full((1, 23), 10., dtype=torch.float64)
    term.has_external = False
    term.solver = WBCACCGPUQP(cfg, max_iterations=1)
    term.failure_count = torch.zeros(1, dtype=torch.long)
    term.qp_failed = torch.zeros(1, dtype=torch.bool)
    term.previous_wrench = term.last_prediction = None
    term._clock = 0
    term._qp_update_pending = True
    term.pin_to_sim = list(range(23))
    applied = []
    term._asset = SimpleNamespace(set_joint_effort_target=lambda x: applied.append(x.clone()))
    term.apply_actions()
    assert term.failure_count.item() == 1
    assert term._env.extras['log']['WBCACC/solver_failure_fraction'] == 1
    assert torch.isfinite(applied[0]).all()
    assert applied[0].abs().max() <= 10.
    torch.testing.assert_close(applied[0], torch.full_like(applied[0], -10.))
    assert term.qp_failed.item()
    # The remaining nine physics substeps hold torque and skip dynamics/QP.
    original_states = term._states
    term._states = lambda: pytest.fail('Dynamics recomputed during torque hold')
    for _ in range(9):
        term.apply_actions()
        torch.testing.assert_close(applied[-1], applied[0])
    assert term._clock == 10
    assert term.failure_count.item() == 1
    term._states = original_states
    # A successful later substep must not erase an earlier failure or resume
    # QP torques before the termination manager can reset the environment.
    result = term.solver.solve(state, state, term._raw.double(),
                              available_vertices(state, cfg), term.limits)
    result['failed'].zero_()
    result['torque'].zero_()
    calls = []
    def solve(*args):
        calls.append(True)
        return result
    term.solver = SimpleNamespace(solve=solve)
    term.process_actions(term._raw.clone())
    term.apply_actions()
    assert len(calls) == 1
    assert term.qp_failed.item()
    assert term.failure_count.item() == 1
    torch.testing.assert_close(applied[-1], applied[0])
    # Reset discards the old command and permits a fresh successful solve.
    term.external = torch.zeros((1, 24, 6))
    term.reset()
    assert not term._torques.any()
    term.process_actions(support_actions())
    term.apply_actions()
    assert len(calls) == 2
    assert not term.qp_failed.any()
    assert not applied[-1].any()
    for _ in range(9):
        term.apply_actions()
    assert len(calls) == 2


def test_reference_window_has_no_actual_state_inputs(monkeypatch):
    import importlib.util
    from types import ModuleType
    managers = ModuleType('isaaclab.managers')
    managers.ObservationGroupCfg = type('ObservationGroupCfg', (), {})
    managers.ObservationTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    managers.SceneEntityCfg = object
    utils = ModuleType('isaaclab.utils'); utils.configclass = lambda cls: cls
    shared = ModuleType('whole_body_tracking.tasks.tracking.tracking_env_cfg')
    shared.ObservationsCfg = SimpleNamespace(PrivilegedCfg=type('PrivilegedCfg', (), {'actual_state': True}))
    for name, module in [('isaaclab.managers', managers), ('isaaclab.utils', utils), (shared.__name__, shared)]:
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc/observations.py'
    spec = importlib.util.spec_from_file_location('wbc_acc_observation_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    motion = SimpleNamespace(time_step_total=12, joint_pos=torch.zeros((12, 23)), joint_vel=torch.zeros((12, 23)))
    for name, width in [('_body_pos_w', 3), ('_body_quat_w', 4), ('_body_lin_vel_w', 3), ('_body_ang_vel_w', 3)]:
        setattr(motion, name, torch.zeros((12, 24, width)))
    command = SimpleNamespace(motion=motion, time_steps=torch.tensor([0, 11]))
    env = SimpleNamespace(device='cpu', command_manager=SimpleNamespace(get_term=lambda _: command), robot_state=torch.zeros(10))
    before = module.reference_window(env)
    env.robot_state[:] = 100.
    torch.testing.assert_close(module.reference_window(env), before)
    motion.joint_pos[10] = 1.
    after = module.reference_window(env)
    assert not torch.equal(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])  # End-of-clip lookahead clamps.
    assert module.WBCACCObservationsCfg.CriticCfg.actual_state


def test_measured_wrench_includes_normal_and_friction_moments(action_modules):
    _, gpu = action_modules
    term = object.__new__(gpu.WBCACCGPUAction)
    term._raw = torch.zeros((2, 14))
    normal = torch.tensor([[10.], [20.], [30.]])
    points = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
    normals = torch.tensor([[0., 0., 1.]]).repeat(3, 1)
    counts, starts = torch.tensor([[2], [1]]), torch.tensor([[0], [2]])
    view = SimpleNamespace(
        get_contact_data=lambda dt: (normal, points, normals, torch.zeros(3), counts, starts),
        get_friction_data=lambda dt: (torch.tensor([[2., 0., 0.], [3., 0., 0.]]),
            torch.tensor([[0., 0., -1.], [0., 0., -1.]]), torch.ones((2, 1), dtype=torch.long), torch.tensor([[0], [1]])))
    term._env = SimpleNamespace(num_envs=2, device='cpu', physics_dt=.002,
        scene=SimpleNamespace(sensors={name: SimpleNamespace(contact_physx_view=view)
            for name in ('left_foot_ground_contact', 'right_foot_ground_contact')}))
    term._asset = SimpleNamespace(body_names=list(FEET), data=SimpleNamespace(body_link_pos_w=torch.zeros((2, 2, 3))))
    result = term._actual_wrenches()
    expected = torch.tensor([[2., 0., 30., 0., 8., 0.], [3., 0., 30., 30., -3., 0.]])
    torch.testing.assert_close(result[:, 0], expected)
    torch.testing.assert_close(result[:, 1], expected)
