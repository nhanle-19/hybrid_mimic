"""Action smoothness in the same normalized units as the learned force bounds."""
from whole_body_tracking.utils.wbc_force_gpu_control import decode_force_actions


def mapped_force_action_rate(env):
    manager = env.action_manager
    return (decode_force_actions(manager.action)-decode_force_actions(manager.prev_action)).square().mean(-1)
