import torch

def set_random_force(env, step, frc_mag):
    # apply random impulse forces to the robot at if timestep meets threshold
    # round(sin(env_num / 6) * 100 + 150 ) == step
    frc_mask = torch.arange(env.num_envs, device=env.device)
    frc_mask = torch.round(torch.sin(frc_mask / 6) * 100 + 150).to(torch.int64)
    
    frc = torch.randn((env.num_envs, 3), device=env.device)
    frc = frc * frc_mag / torch.linalg.norm(frc, dim=-1, keepdim=True)

    frc = torch.where(frc_mask[:, None] == step, frc, torch.zeros_like(frc))

    robot = env.scene["robot"]
    num_bodies = robot.num_bodies
    forces_b = torch.zeros((env.num_envs, num_bodies, 3), device=env.device)
    torques_b = torch.zeros((env.num_envs, num_bodies, 3), device=env.device)
    positions_b = torch.zeros((env.num_envs, num_bodies, 3), device=env.device)
    forces_b[:, 0, :] = frc  # apply to base link

    robot.apply_external_forces_torques(forces_b, torques_b, positions_b)
    robot.write_data_to_sim()

    return