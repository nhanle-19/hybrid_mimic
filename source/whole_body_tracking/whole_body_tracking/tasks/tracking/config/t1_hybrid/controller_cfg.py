from isaaclab.utils import configclass

from whole_body_tracking.robots.t1 import T1_END_EFFECTOR_BODY_NAMES


@configclass
class T1HybridControllerCfg:
    """T1-specific parameters that cannot be derived from the Isaac articulation."""

    end_effector_body_names: list[str] = T1_END_EFFECTOR_BODY_NAMES.copy()

    # Aggregate body-frame inertia used by the centroidal controller approximation.
    nominal_angular_inertia: tuple[tuple[float, float, float], ...] = (
        (2.76900149, 4.50170509e-4, 3.66299529e-2),
        (4.50170509e-4, 2.30203655, -4.42839862e-4),
        (3.66299529e-2, -4.42839862e-4, 5.62235551e-1),
    )

    torque_limits_cost: dict[str, float] = {
        "AAHead_yaw": 7.0,
        "Left_Shoulder_Pitch": 18.0,
        "Right_Shoulder_Pitch": 18.0,
        "Waist": 30.0,
        "Head_pitch": 7.0,
        "Left_Shoulder_Roll": 18.0,
        "Right_Shoulder_Roll": 18.0,
        "Left_Hip_Pitch": 45.0,
        "Right_Hip_Pitch": 45.0,
        "Left_Elbow_Pitch": 18.0,
        "Right_Elbow_Pitch": 18.0,
        "Left_Hip_Roll": 25.0,
        "Right_Hip_Roll": 25.0,
        "Left_Elbow_Yaw": 18.0,
        "Right_Elbow_Yaw": 18.0,
        "Left_Hip_Yaw": 25.0,
        "Right_Hip_Yaw": 25.0,
        "Left_Knee_Pitch": 60.0,
        "Right_Knee_Pitch": 60.0,
        "Left_Ankle_Pitch": 10.0,
        "Right_Ankle_Pitch": 10.0,
        "Left_Ankle_Roll": 7.5,
        "Right_Ankle_Roll": 7.5,
    }

    desired_linear_velocity_scale: float = 0.25
    desired_angular_velocity_scale: float = 0.50
    torque_action_scale: float = 0.10

    linear_velocity_gain: float = 10.0
    angular_velocity_gain: float = 15.0

    force_objective_weight: float = 1.0e-3
    torque_objective_weight: float = 4.0e2
    angular_force_scale: float = 20.0
    force_logit_clip: float = 10.0
    gravity_magnitude: float = 9.81
