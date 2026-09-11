"""Batched GPU-capable kernels compared to the independent Pinocchio/OSQP reference."""
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
import torch
import pytest


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def device(request):
    return request.param

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/whole_body_tracking/whole_body_tracking/utils'))
from atlas_model import AtlasModel
from atlas_torch_model import AtlasTorchModel


def test_batched_dynamics_match_pinocchio(device):
    reference = AtlasModel()
    batched = AtlasTorchModel(device)
    assert batched.joint_names == reference.joint_names
    rng = np.random.default_rng(92)
    qs = np.stack([pin.integrate(reference.model, pin.neutral(reference.model), rng.normal(0, .25, 29)) for _ in range(5)])
    vs = rng.normal(0, .5, (5, 29))
    result = batched.state(torch.tensor(qs, device=device), torch.tensor(vs, device=device))
    for i in range(5):
        truth = reference.state(qs[i], vs[i])
        for key in ['M', 'bias', 'Ag', 'Ag_bias', 'com', 'momentum']:
            np.testing.assert_allclose(result[key][i].cpu().numpy(), truth[key], atol=1e-8, err_msg=key)
        for name in reference.frame_ids:
            for key in ['position', 'rotation', 'J', 'bias', 'velocity']:
                np.testing.assert_allclose(result['frames'][name][key][i].cpu().numpy(), truth['frames'][name][key], atol=1e-8, err_msg=f'{name}/{key}')


def test_batched_qp_matches_osqp_with_different_active_bounds(device):
    import osqp
    from scipy import sparse
    from atlas_batched_qp import solve_batched_qp
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(8, 12, 12))
    hessian = matrix.transpose(0, 2, 1)@matrix+np.eye(12)*.1
    linear = rng.normal(size=(8, 12))*8
    constraint = np.broadcast_to(np.r_[np.eye(12), -np.eye(12)], (8, 24, 12)).copy()
    bound = np.ones((8, 24))
    x, info = solve_batched_qp(*(torch.tensor(value, device=device) for value in (hessian, linear, constraint, bound)))
    assert x.device == torch.device(device)
    assert info['converged'].all(), info
    for i in range(8):
        ref = osqp.OSQP()
        ref.setup(P=sparse.csc_matrix(np.triu(hessian[i])), q=linear[i], A=sparse.csc_matrix(constraint[i]),
                  l=np.full(24, -np.inf), u=bound[i], eps_abs=1e-10, eps_rel=1e-10, verbose=False, polish=True)
        result = ref.solve()
        np.testing.assert_allclose(x[i].cpu().numpy(), result.x, atol=2e-5)


def atlas_cfg():
    from types import SimpleNamespace
    return SimpleNamespace(friction=.6,momentum_weights=(1.,1.,1.,10.,10.,10.),force_weight=1e-5,
        acceleration_weight=1e-5,force_objective_weight=.001,force_logit_clip=10.,angular_force_scale=20.,
        base_acceleration_weight=200.,enforce_stance=False,batched_max_iterations=60,
        batched_tolerance=1e-9,residual_tolerance=2e-5)


def test_batched_atlas_matches_osqp_mixed_contacts(device, monkeypatch):
    import osqp
    original_setup = osqp.OSQP.setup

    def accurate_setup(self, **kwargs):
        # Polishing degrades this ill-conditioned oracle's stationarity.
        kwargs.update(polish=False, eps_abs=1e-10, eps_rel=0., max_iter=100000)
        return original_setup(self, **kwargs)

    monkeypatch.setattr(osqp.OSQP, 'setup', accurate_setup)
    from atlas_batched_control import BatchedAtlasQP
    from atlas_qp import AtlasQP, Contact, MotionTask
    torch.set_num_threads(1)
    ref, model = AtlasModel(), AtlasTorchModel(device)
    cfg = atlas_cfg()
    names = ['left_foot_link','right_foot_link','left_hand_link','right_hand_link']
    solver = BatchedAtlasQP(names,cfg,device)
    rng = np.random.default_rng(8)
    q = np.stack([pin.integrate(ref.model,pin.neutral(ref.model),rng.normal(0,.1,29)) for _ in range(4)])
    v = rng.normal(0,.1,(4,29))
    state = model.state(torch.tensor(q,device=device),torch.tensor(v,device=device))
    pd = torch.tensor(rng.normal(0,20,(4,23)), device=device)
    limits = torch.full((4,23),60.,dtype=torch.float64, device=device)
    acceleration = torch.tensor(rng.normal(0,1,(4,6)), device=device)
    logits = torch.tensor(rng.normal(0,1,(4,5)), device=device)
    torque_ref = torch.tensor(rng.normal(0,2,(4,23)), device=device)
    tw = torch.ones_like(pd)
    active = torch.tensor([[False]*4,[True,False,False,False],[True,True,False,False],[False,True,True,True]], device=device)
    result = solver.solve(state,pd,limits,acceleration,logits,torque_ref,tw,active)
    assert result['total_torque'].device == torch.device(device)
    assert result['valid'].all(), (result['solver_info'],result['equality_error'],result['inequality_error'])
    assert torch.all(result['total_torque'].abs() <= limits+cfg.residual_tolerance)
    torch.testing.assert_close(result['torque']+pd, result['total_torque'])
    assert torch.all(result['forces'][~active] == 0)
    normal = result['forces'][:,:,2]
    tangent = torch.linalg.vector_norm(result['forces'][:,:,:2],dim=-1)
    assert torch.all(normal >= -cfg.residual_tolerance)
    assert torch.all(tangent <= cfg.friction*normal+cfg.residual_tolerance)
    for i in range(4):
        s = ref.state(q[i],v[i])
        contacts = [Contact(name) if k<2 else Contact(name,points=np.zeros((1,3)),constrain_rotation=False)
                    for k,name in enumerate(names) if active[i,k]]
        objective = dict(base_body='Trunk',base_weight=.001*np.exp(-float(logits[i,0])),
            contact_weights={name:.001*np.exp(-float(logits[i,k+1])) for k,name in enumerate(names)},
            angular_force_scale=20.,torque_reference=torque_ref[i].cpu().numpy(),torque_weights=tw[i].cpu().numpy())
        base=s['frames']['Trunk']
        task=MotionTask(base['J'],acceleration[i].cpu().numpy()-base['bias'],200.,'base')
        truth=AtlasQP(np.full(23,60.),enforce_stance=False).solve(s,result['desired_rate'][i].cpu().numpy(),contacts,
            [task],hybrid=objective,pd_torque=pd[i].cpu().numpy())
        np.testing.assert_allclose(result['total_torque'][i].cpu().numpy(),truth['total_torque'],atol=2e-3,rtol=1e-4)
        np.testing.assert_allclose(result['rate'][i].cpu().numpy(),truth['rate'],atol=2e-3,rtol=1e-4)


def test_large_pd_cancellation_and_batch_independence(device):
    from atlas_batched_control import BatchedAtlasQP
    torch.manual_seed(91)
    cfg = atlas_cfg()
    cfg.batched_tolerance = 1e-7
    model = AtlasTorchModel(device)
    solver = BatchedAtlasQP(['left_foot_link','right_foot_link','left_hand_link','right_hand_link'],cfg,device)
    q = torch.zeros((16,30), device=device, dtype=torch.float64)
    q[:,6] = 1.
    q[:,7:] = .2*torch.randn_like(q[:,7:])
    v = .3*torch.randn((16,29), device=device, dtype=torch.float64)
    pd = 200*torch.randn((16,23), device=device, dtype=torch.float64)
    limits = torch.full_like(pd,30.)
    acc = torch.randn((16,6), device=device, dtype=torch.float64)
    logits = torch.randn((16,5), device=device, dtype=torch.float64)
    active = torch.rand((16,4), device=device)>.5
    result = solver.solve(model.state(q,v),pd,limits,acc,logits,torch.zeros_like(pd),torch.ones_like(pd),active)
    assert result['valid'].all(), result['solver_info']
    assert torch.all(result['total_torque'].abs() <= limits+cfg.residual_tolerance)
    torch.testing.assert_close(result['torque']+pd,result['total_torque'])
    order = torch.arange(15,-1,-1,device=device)
    permuted = solver.solve(model.state(q[order],v[order]),pd[order],limits[order],acc[order],logits[order],
        torch.zeros_like(pd),torch.ones_like(pd),active[order])
    assert permuted['valid'].all()
    torch.testing.assert_close(permuted['total_torque'],result['total_torque'][order],atol=1e-6,rtol=1e-6)


def test_factorization_failure_does_not_poison_healthy_batch_member(device):
    from atlas_batched_qp import solve_batched_qp
    hessian = torch.eye(2, dtype=torch.float64, device=device).repeat(2, 1, 1)
    hessian[1].neg_()  # Deliberately violate convexity in just one member.
    linear = torch.ones((2, 2), dtype=torch.float64, device=device)
    inequality = torch.zeros((2, 1, 2), dtype=torch.float64, device=device)
    inequality[:, 0, 0] = 1.
    bound = torch.ones((2, 1), dtype=torch.float64, device=device)
    x, info = solve_batched_qp(hessian, linear, inequality, bound)
    assert info['converged'].tolist() == [True, False]
    assert info['factorization_failed'].tolist() == [False, True]
    torch.testing.assert_close(x[0], -linear[0])
    assert torch.isfinite(x).all()


def test_warm_start_matches_cold_after_state_contact_and_reset_changes(device):
    from atlas_batched_control import BatchedAtlasQP
    torch.manual_seed(31)
    cfg = atlas_cfg()
    cfg.batched_warm_start = True
    cfg.batched_tolerance = 1e-7  # Match the training configuration for saturated actions.
    names = ['left_foot_link','right_foot_link','left_hand_link','right_hand_link']
    warm = BatchedAtlasQP(names, cfg, device)
    cfg_cold = atlas_cfg()
    cfg_cold.batched_tolerance = cfg.batched_tolerance
    cold = BatchedAtlasQP(names, cfg_cold, device)
    model = AtlasTorchModel(device)
    batch = 8
    q = torch.zeros((batch,30), dtype=torch.float64, device=device)
    q[:,6] = 1.
    v = torch.zeros((batch,29), dtype=torch.float64, device=device)
    pd = torch.zeros((batch,23), dtype=torch.float64, device=device)
    limits = torch.full_like(pd, 30.)
    active = torch.ones((batch,4), dtype=torch.bool, device=device)
    for step in range(4):
        q[:,7:] += .02*torch.randn_like(q[:,7:])
        v += .1*torch.randn_like(v)
        pd += 30*torch.randn_like(pd)
        active = ~active if step == 2 else active
        if step == 3:
            # Invalidate only reset environments, preserving other warm starts.
            warm.warm_start['valid'][::2] = False
        args = (model.state(q,v), pd, limits,
                torch.randn((batch,6), dtype=q.dtype, device=device),
                torch.randn((batch,5), dtype=q.dtype, device=device),
                torch.zeros_like(pd), torch.ones_like(pd), active)
        actual, expected = warm.solve(*args), cold.solve(*args)
        assert actual['valid'].all(), actual['solver_info']
        assert expected['valid'].all(), expected['solver_info']
        torch.testing.assert_close(actual['total_torque'], expected['total_torque'], atol=2e-3, rtol=1e-4)
        torch.testing.assert_close(actual['rate'], expected['rate'], atol=2e-3, rtol=1e-4)
        assert torch.all(actual['total_torque'].abs() <= limits+cfg.residual_tolerance)
        assert torch.all(actual['forces'][~active] == 0)
