"""Force-limit policy task sharing WBC-ACC simulation with survival-only reward."""
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg
from ..t1_wbc_acc.flat_env_cfg import T1WBCACCEnvCfg, T1WBCACCEnvEvalCfg, T1WBCACCStandingEnvCfg, WBCAliveOnlyRewardsCfg
from ..t1_wbc_acc.rewards import qp_failed
from .wbc_force_action import WBCForceActionCfg


@configclass
class WBCForceActionsCfg:
    wbc_force: WBCForceActionCfg = WBCForceActionCfg()


def configure_wbc_force(cfg):
    previous = cfg.actions.wbc_acc
    cfg.actions = WBCForceActionsCfg()
    cfg.actions.wbc_force.record_diagnostics = previous.record_diagnostics
    cfg.actions.wbc_force.standing_only = previous.standing_only
    cfg.rewards = WBCAliveOnlyRewardsCfg()
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


@configclass
class T1WBCForceToyStandingEnvCfg(T1WBCForceStandingEnvCfg):
    """Deterministic toes-up reference fixture with fall and QP failure checks."""

    def __post_init__(self):
        super().__post_init__()
        from whole_body_tracking.tasks.tracking.tracking_env_cfg import TerminationsCfg
        self.scene.num_envs = 1
        self.commands.motion.motion_file = 'eval_data/toy_standing/reference.npz'
        # GPU PhysX contact jitter exceeds the generic 10-micrometre patch
        # threshold. Keep a nearly level sole available within 0.5 mm; the
        # independent 1 mm ground-gap gate still applies to every vertex.
        self.actions.wbc_force.controller.contact_patch_height_tolerance = .0005
        self.terminations = TerminationsCfg()
        self.terminations.ee_body_pos.params['body_names'] = list(self.commands.motion.eef_names)
        self.terminations.qp_infeasible = TerminationTermCfg(
            func=qp_failed, params={'action_name': 'wbc_force'})
