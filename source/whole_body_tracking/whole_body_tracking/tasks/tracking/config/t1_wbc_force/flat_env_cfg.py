"""Separate force-limit policy task sharing the WBC-ACC simulation and rewards."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from ..t1_wbc_acc.flat_env_cfg import T1WBCACCEnvCfg, T1WBCACCEnvEvalCfg, T1WBCACCStandingEnvCfg
from ..t1_wbc_acc.rewards import qp_failed
from .wbc_force_action import WBCForceActionCfg
from .rewards import mapped_force_action_rate


@configclass
class WBCForceActionsCfg:
    wbc_force: WBCForceActionCfg = WBCForceActionCfg()


def configure_wbc_force(cfg):
    previous = cfg.actions.wbc_acc
    cfg.actions = WBCForceActionsCfg()
    cfg.actions.wbc_force.record_diagnostics = previous.record_diagnostics
    cfg.actions.wbc_force.standing_only = previous.standing_only
    cfg.rewards.action_rate_l2 = RewardTermCfg(func=mapped_force_action_rate, weight=-.1)
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed, params={'action_name': 'wbc_force'})


@configclass
class T1WBCForceEnvCfg(T1WBCACCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_force(self)


@configclass
class T1WBCForceEnvEvalCfg(T1WBCACCEnvEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_force(self)


@configclass
class T1WBCForceStandingEnvCfg(T1WBCACCStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_force(self)
