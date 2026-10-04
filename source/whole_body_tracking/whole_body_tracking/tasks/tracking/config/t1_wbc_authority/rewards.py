"""Smoothness in bounded authority coordinates, consistent with the QP."""
from whole_body_tracking.utils.wbc_authority_gpu_control import decode_authority_actions


def mapped_authority_action_rate(env):
    manager = env.action_manager
    return (decode_authority_actions(manager.action)-decode_authority_actions(manager.prev_action)).square().mean(dim=(1, 2))
