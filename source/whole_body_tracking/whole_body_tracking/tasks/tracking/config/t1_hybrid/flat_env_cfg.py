from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import BOOSTER_T1_LOWGAIN_CFG, T1_ANKLE_JOINT_NAMES, T1_LG_ACTION_SCALE
from whole_body_tracking.tasks.tracking.tracking_hybrid_env_cfg import TrackingHybridEnvCfg, TrackingHybridEnvEvalCfg

from .controller_cfg import T1HybridControllerCfg


@configclass
class T1HybridEnvCfg(TrackingHybridEnvCfg):
    hybrid_controller: T1HybridControllerCfg = T1HybridControllerCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_T1_LOWGAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_LG_ACTION_SCALE
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = [
            "Trunk",
            "Hip_Roll_Left",
            "Hip_Yaw_Left",
            "Shank_Left",
            "left_foot_link",
            "Ankle_Cross_Left",
            "Hip_Roll_Right",
            "Hip_Yaw_Right",
            "Shank_Right",
            "right_foot_link",
            "Ankle_Cross_Right",
            "Waist",
            "AL2",
            "AL3",
            "left_hand_link",
            "AR2",
            "AR3",
            "right_hand_link",
        ]
        end_effector_names = list(self.hybrid_controller.end_effector_body_names)
        self.commands.motion.eef_names = end_effector_names
        self.commands.motion.ankle_names = T1_ANKLE_JOINT_NAMES
        self.rewards.contact_hybrid.params["sensor_cfg"].body_names = end_effector_names
        self.rewards.force_correctness.params["sensor_cfg"].body_names = end_effector_names

@configclass
class T1HybridEnvEvalCfg(TrackingHybridEnvEvalCfg):
    hybrid_controller: T1HybridControllerCfg = T1HybridControllerCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BOOSTER_T1_LOWGAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_LG_ACTION_SCALE
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = [
            "Trunk",
            "Hip_Roll_Left",
            "Hip_Yaw_Left",
            "Shank_Left",
            "left_foot_link",
            "Ankle_Cross_Left",
            "Hip_Roll_Right",
            "Hip_Yaw_Right",
            "Shank_Right",
            "right_foot_link",
            "Ankle_Cross_Right",
            "Waist",
            "AL2",
            "AL3",
            "left_hand_link",
            "AR2",
            "AR3",
            "right_hand_link",
        ]
        end_effector_names = list(self.hybrid_controller.end_effector_body_names)
        self.commands.motion.eef_names = end_effector_names
        self.commands.motion.ankle_names = T1_ANKLE_JOINT_NAMES
        self.rewards.contact_hybrid.params["sensor_cfg"].body_names = end_effector_names
        self.rewards.force_correctness.params["sensor_cfg"].body_names = end_effector_names
