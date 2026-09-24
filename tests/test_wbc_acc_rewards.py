"""WBCACC reward regressions without launching Isaac Sim."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import torch

PATH = Path(__file__).resolve().parents[1] / 'source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1_wbc_acc'
spec = importlib.util.spec_from_file_location('wbc_acc_rewards_test', PATH / 'rewards.py')
rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewards)


def test_mapped_rate_ignores_saturated_raw_changes():
    previous = torch.full((2, 14), 100.)
    current = previous * 2
    env = NS(action_manager=NS(action=current, prev_action=previous))
    torch.testing.assert_close(rewards.mapped_action_rate(env), torch.zeros(2))
    # Both activation channels move through their full physical range.
    current[:, [0, 7]] = 0
    torch.testing.assert_close(rewards.mapped_action_rate(env), torch.full((2,), 2/14))


def test_qp_failure_is_per_environment():
    failure = torch.tensor([True, False])
    env = NS(action_manager=NS(get_term=lambda _: NS(qp_failed=failure)),
             termination_manager=NS(terminated=failure), step_dt=.02)
    assert rewards.qp_failed(env).tolist() == [True, False]


def test_alive_reward_excludes_failures_but_includes_timeouts():
    env = NS(termination_manager=NS(
        terminated=torch.tensor([False, True, False, True]),
        time_outs=torch.tensor([False, False, True, True])))
    torch.testing.assert_close(rewards.alive_reward(env), torch.tensor([1., 0., 1., 0.]))


def test_alive_return_grows_with_survival_time_independent_of_step_dt():
    env = NS(termination_manager=NS(terminated=torch.tensor([False])))
    for dt in (.01, .02, .04):
        for seconds in (1., 2.):
            episode_return = (rewards.alive_reward(env) * dt).repeat(round(seconds / dt)).sum()
            torch.testing.assert_close(episode_return, torch.tensor(seconds))


def contact_env(linear, angular, counts=(1, 1), force=10.):
    # Two environments, one point per foot, displaced one meter along x.
    points = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    normal = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    view = NS(get_contact_data=lambda **_: (torch.full((2, 1), force), points, normal, None,
        torch.tensor(counts), torch.tensor([0, 1])))
    data = NS(body_link_pos_w=torch.zeros(2, 2, 3),
              body_link_lin_vel_w=torch.tensor(linear).expand(2, 2, 3),
              body_ang_vel_w=torch.tensor(angular).expand(2, 2, 3))
    robot = NS(body_names=['left_foot_link', 'right_foot_link'], data=data)
    class Scene(dict):
        sensors = {f'{s}_foot_ground_contact': NS(contact_physx_view=view) for s in ('left', 'right')}
    return NS(scene=Scene(robot=robot), num_envs=2, device='cpu', physics_dt=.002)


def test_slip_allows_stationary_toe_pivot():
    env = contact_env([0., -1., 0.], [0., 0., 1.])
    torch.testing.assert_close(rewards.excessive_slip(env), torch.zeros(2))


def test_slip_uses_loaded_contacts_and_tangent_only():
    env = contact_env([1., 0., 4.], [0., 0., 0.], counts=(1, 0))
    torch.testing.assert_close(rewards.excessive_slip(env), torch.tensor([2*.95**2, 0.]))
    env = contact_env([1., 0., 0.], [0., 0., 0.], force=.5)
    torch.testing.assert_close(rewards.excessive_slip(env), torch.zeros(2))
