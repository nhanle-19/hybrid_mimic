"""Force-limit policy with sole-point/COM tracking and contact rewards."""
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from ..t1_wbc_acc.flat_env_cfg import T1WBCACCEnvCfg, T1WBCACCEnvEvalCfg, T1WBCACCStandingEnvCfg
from ..t1_wbc_acc.rewards import qp_failed
from ..t1_wbc_acc.observations import actual_root_joint_state
from .observations import reference_window
from .wbc_force_action import WBCForceActionCfg
from .motion import WBCForceTrainingMotion
from .reward_terms import (
    MotionCompletionReward, SwingForcePenalty,
    ForceActionChangePenalty, FalseZeroPenalty, ForceCapGapPenalty,
    SolePointTrackingPenalty, COMTrackingPenalty, reference_motion_complete,
)


@configclass
class WBCForceActionsCfg:
    wbc_force: WBCForceActionCfg = WBCForceActionCfg()


@configclass
class WBCForceRewardsCfg:
    # All terms share 100 / reference_frame_count scaling per policy step.
    motion_completion = RewardTermCfg(func=MotionCompletionReward, weight=1.)
    # Sum over eight corners: one-quarter the old mean weight gives twice its cost.
    swing_force = RewardTermCfg(func=SwingForcePenalty, weight=-.125,
                               params={'clearance_start': .001, 'clearance_full': .005, 'unload_lead_time': 0.})
    action_change = RewardTermCfg(func=ForceActionChangePenalty, weight=-.5)
    # Like swing_force, sum with one-quarter the old mean weight doubles its cost.
    false_zero = RewardTermCfg(func=FalseZeroPenalty, weight=-.1,
                              params={'clearance_start': .001, 'clearance_full': .005})
    force_cap_gap = RewardTermCfg(func=ForceCapGapPenalty, weight=-.08)
    # Split the former tracking penalty budget between geometry and balance.
    # Sum standardized squared errors, then bound each cost with 1-exp(-error).
    sole_point_tracking = RewardTermCfg(func=SolePointTrackingPenalty, weight=-.25, params={'error_scale': .05})
    com_tracking = RewardTermCfg(func=COMTrackingPenalty, weight=-.25,
                                params={'error_scale': .05, 'velocity_error_scale': .2, 'angular_velocity_error_scale': .5})


def configure_wbc_force(cfg):
    previous = cfg.actions.wbc_acc
    cfg.actions = WBCForceActionsCfg()
    cfg.actions.wbc_force.record_diagnostics = previous.record_diagnostics
    cfg.actions.wbc_force.standing_only = previous.standing_only
    # Both networks keep current joints/root and five measured-state frames;
    # future references contain all-link poses/velocities without joint fields.
    for group in (cfg.observations.policy, cfg.observations.critic):
        # A group-level history would override the individual term lengths.
        group.history_length = None
        group.reference.func = reference_window
        group.reference.params = {'offsets': (0, 1, 3, 5, 10)}
        group.reference.history_length = 0
        group.actual_state = ObservationTermCfg(func=actual_root_joint_state,
                                                history_length=5, flatten_history_dim=True)
    cfg.rewards = WBCForceRewardsCfg()
    # Contact-cap learning prioritizes balance and feasible motion. Limb-height
    # tracking error alone is not a failure; retain the coarse torso guards.
    cfg.terminations.ee_body_pos = None
    cfg.terminations.qp_infeasible = TerminationTermCfg(func=qp_failed, params={'action_name': 'wbc_force'})


@configclass
class T1WBCForceEnvCfg(T1WBCACCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_wbc_force(self)
        # Keep whole-motion attempts represented when the adaptive sampler
        # concentrates on short suffixes. Other tasks retain their sampler.
        self.commands.motion.start_from_beginning_probability = .2
        self.commands.motion.class_type = WBCForceTrainingMotion
        # End a training attempt before MotionCommand silently resamples the
        # robot at the clip boundary. Adaptive phase starts remain enabled.
        self.terminations.motion_complete = TerminationTermCfg(func=reference_motion_complete, time_out=True)


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
        self.terminations = TerminationsCfg()
        self.terminations.ee_body_pos = None
        self.terminations.qp_infeasible = TerminationTermCfg(
            func=qp_failed, params={'action_name': 'wbc_force'})
