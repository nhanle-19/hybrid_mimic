from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def randomize_lcc(
        env,
        env_ids: torch.Tensor | None,
        lcc_range: dict[str, tuple[float, float]],
):
    """
    Make a dictionary of randomized lcc_rand_dict that defines a set of offsets and
    scalar multiples to randomize aspects of the com dict.
    vel_range: tuple[float, float],
    angvel_range: tuple[float, float],
    mass_fac_range: tuple[float, float],
    i_fac_range: tuple[float, float],
    jac_fac_range: tuple[float, float],
    pos_range: tuple[float, float],

    Buffer dimensions are provided by ``env.lcc_bias`` so this event remains
    independent of the robot's joint, body, and end-effector counts.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    vel_range = lcc_range.get("vel_range", (0.0, 0.0))
    angvel_range = lcc_range.get("angvel_range", (0.0, 0.0))
    mass_fac_range = lcc_range.get("mass_fac_range", (1.0, 1.0))
    i_fac_range = lcc_range.get("i_fac_range", (1.0, 1.0))
    jac_fac_range = lcc_range.get("jac_fac_range", (1.0, 1.0))
    pos_range = lcc_range.get("pos_range", (0.0, 0.0))

    lcc_rand_dict = env.lcc_bias

    def sample_uniform(key, value_range):
        shape = (len(env_ids), *lcc_rand_dict[key].shape[1:])
        return math_utils.sample_uniform(value_range[0], value_range[1], shape, device=env.device)

    vel_offsets = sample_uniform("com_vel", vel_range)
    angvel_offsets = sample_uniform("com_angvel", angvel_range)
    mass_facs = sample_uniform("mass_fac", mass_fac_range)
    i_facs = sample_uniform("i_fac", i_fac_range)
    jac_facs = sample_uniform("jac_fac", jac_fac_range)
    pos_offsets = sample_uniform("pos", pos_range)
    grav_vec = sample_uniform("grav_vec", (-0.5, 0.5))

    lcc_rand_dict["com_vel"][env_ids] = vel_offsets
    lcc_rand_dict["com_angvel"][env_ids] = angvel_offsets
    lcc_rand_dict["mass_fac"][env_ids] = mass_facs
    lcc_rand_dict["i_fac"][env_ids] = i_facs
    lcc_rand_dict["jac_fac"][env_ids] = jac_facs
    lcc_rand_dict["pos"][env_ids] = pos_offsets
    lcc_rand_dict["grav_vec"][env_ids] = grav_vec
    env.lcc_bias = lcc_rand_dict
