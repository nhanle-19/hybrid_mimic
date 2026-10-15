"""Authority-limit policy task sharing WBC-ACC simulation with survival-only reward."""
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg
from ..t1_wbc_acc.flat_env_cfg import T1WBCACCEnvCfg, T1WBCACCEnvEvalCfg, T1WBCACCStandingEnvCfg, WBCAliveOnlyRewardsCfg
from ..t1_wbc_acc.rewards import qp_failed
from .wbc_authority_action import WBCAuthorityActionCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TerminationsCfg


@configclass
class WBCAuthorityActionsCfg:
    wbc_authority: WBCAuthorityActionCfg = WBCAuthorityActionCfg()


def configure_wbc_authority(cfg):
    previous = cfg.actions.wbc_acc
    cfg.actions = WBCAuthorityActionsCfg()
    cfg.actions.wbc_authority.record_diagnostics = previous.record_diagnostics
    cfg.actions.wbc_authority.standing_only = previous.standing_only
    cfg.rewards = WBCAliveOnlyRewardsCfg()
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed, params={'action_name': 'wbc_authority'})


@configclass
class T1WBCAuthorityEnvCfg(T1WBCACCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_authority(self)


@configclass
class T1WBCAuthorityEnvEvalCfg(T1WBCACCEnvEvalCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_authority(self)


@configclass
class T1WBCAuthorityStandingEnvCfg(T1WBCACCStandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # A standing preflight must detect falls/tracking divergence, not only
        # QP failures and timeouts as in permissive playback evaluation.
        self.terminations = TerminationsCfg()
        configure_wbc_authority(self)


@configclass
class T1WBCAuthorityToyStandingEnvCfg(T1WBCAuthorityStandingEnvCfg):
    """Same toes-up reference fixture as the force-only toy task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.motion.motion_file = 'eval_data/toy_standing/reference.npz'
        self.actions.wbc_authority.controller.contact_patch_height_tolerance = .0005
        self.terminations.ee_body_pos.params['body_names'] = list(self.commands.motion.eef_names)
