"""Reward calculation for policy training.

Preserves original calculation pattern from drone_long_traj.py lines 687-795.
"""
import torch
from .config import EnvConfig
from utils.torch_utils import normalize
from utils.rotation import quaternion_yaw_forward


def calculate_reward(
    # Observation buffers (same structure as original)
    obs_buf: torch.Tensor,              # (N, obs_dim)
    privilege_obs_buf: torch.Tensor,    # (N, priv_obs_dim)
    prev_prev_actions: torch.Tensor,    # (N, 4)
    # Navigation inputs
    waypoints: torch.Tensor,            # (num_wp, 3) - self.reward_wp
    target: torch.Tensor,               # (1, 3) - self.target
    ref_traj: torch.Tensor,             # (num_points, 3) - self.ref_traj
    point_cloud,                        # ObstacleDistanceCalculator
    # Config
    cfg: EnvConfig,
    start_rotation: torch.Tensor,       # (4,) - self.start_rotation
    point_cloud_offset: torch.Tensor,   # (1, 3) - self.point_could_origin_offset
    num_envs: int,
    device: str = 'cuda:0',
) -> torch.Tensor:
    """
    Calculate total reward. Preserves original calculation from drone_long_traj.py.

    Source: drone_long_traj.py lines 687-773
    """
    # === 1. Survival reward === (line 695)
    survive_reward = cfg.survive_reward

    # === 2. Action penalty === (lines 698-699)
    action_penalty = torch.sum(torch.square(obs_buf[:, 6:9]), dim=-1) * cfg.action_penalty
    action_penalty += torch.sum(torch.square(obs_buf[:, 9] - 0.42)) * (2 * cfg.action_penalty)

    # === 3. Action change penalty === (lines 700-701)
    action_change_penalty = torch.sum(torch.square(obs_buf[:, 6:9] - obs_buf[:, 10:13]), dim=-1) * cfg.action_change_penalty
    action_change_penalty += torch.sum(torch.square(obs_buf[:, 9] - obs_buf[:, 13])) * (2 * cfg.action_change_penalty)

    # === 4. Smooth penalty (jerk) === (lines 702-703)
    smooth_penalty = torch.sum(torch.square(obs_buf[:, 6:9] - 2 * obs_buf[:, 10:13] + prev_prev_actions[:, 0:3]), dim=-1) * cfg.smooth_penalty
    smooth_penalty += torch.sum(torch.square(obs_buf[:, 9] - 2 * obs_buf[:, 13] + prev_prev_actions[:, 3])) * (2 * cfg.smooth_penalty)

    # === 5. Heading reward === (lines 706-709)
    target_dirs = normalize(privilege_obs_buf[:, 3:5])
    torso_quat = obs_buf[:, 2:6]  # (x,y,z,w)
    heading_vec = quaternion_yaw_forward(torso_quat)
    yaw_alignment = (heading_vec * target_dirs).sum(dim=-1)
    heading_reward = cfg.heading_strength * yaw_alignment

    # === 6. Linear velocity penalty === (line 710)
    lin_vel_reward = cfg.lin_vel_penalty * torch.sum(torch.square(obs_buf[:, 14:17]), dim=-1)

    # === 7. Pose penalty === (line 711)
    pose_penalty = torch.sum(torch.abs(obs_buf[:, 2:6] - start_rotation), dim=-1) * cfg.pose_penalty

    # === 8. Height penalty === (line 712)
    height_penalty = torch.square(obs_buf[:, 0] - cfg.target_height) * cfg.height_penalty

    # === 9. Out of map penalty === (lines 713-718)
    out_map_penalty = (
        torch.clamp(cfg.x_min - privilege_obs_buf[:, 0], min=0) ** 2 +
        torch.clamp(privilege_obs_buf[:, 0] - cfg.x_max, min=0) ** 2 +
        torch.clamp(cfg.y_min - privilege_obs_buf[:, 1], min=0) ** 2 +
        torch.clamp(privilege_obs_buf[:, 1] - cfg.y_max, min=0) ** 2
    ) * cfg.out_map_penalty

    # === 10. Waypoint reward === (lines 721-727)
    wp_reward = torch.zeros(num_envs, device=device)
    for i, waypoint in enumerate(waypoints):
        waypoint = waypoint.repeat((num_envs, 1))
        distances = torch.norm(privilege_obs_buf[:, 0:3] - waypoint, dim=1) ** 2
        factor = (torch.exp(1 / (distances + torch.ones(num_envs, device=device)))
                  - torch.ones(num_envs, device=device))
        wp_reward += factor * cfg.waypoint_strength

    # === 11. Target reward (trajectory tracking) === (lines 729-749)
    ref_x = privilege_obs_buf[:, 0] + point_cloud_offset[0, 0]
    ref_traj_x = ref_traj[:, 0]
    diff = ref_traj_x.unsqueeze(0) - ref_x.unsqueeze(1)
    mask = diff > 0.5
    diff_masked = torch.where(mask, diff, float('inf'))
    min_diffs, min_indices = torch.min(diff_masked, dim=1)
    no_valid_indices = torch.isinf(min_diffs)
    target_list = torch.empty((num_envs, 3), device=device)
    default_target = target[0] + point_cloud_offset[0]
    target_list[:] = default_target
    valid_indices = ~no_valid_indices
    selected_indices = min_indices[valid_indices]
    targets = ref_traj[selected_indices]
    target_list[valid_indices] = targets
    target_list = target_list - point_cloud_offset

    desire_velo_norm = (target_list - privilege_obs_buf[:, 0:3]) / torch.clamp(
        torch.norm(target_list - privilege_obs_buf[:, 0:3], dim=1, keepdim=True), min=1e-6)
    curr_velo_norm = obs_buf[:, 14:17] / torch.clamp(
        torch.norm(obs_buf[:, 14:17], dim=1, keepdim=True), min=1e-2)
    velo_dist = torch.norm(curr_velo_norm - desire_velo_norm, dim=1)
    target_reward = velo_dist * cfg.target_factor

    # === 12. Obstacle reward === (lines 751-757)
    drone_pos = privilege_obs_buf[:, 0:3] + point_cloud_offset
    drone_rot = obs_buf[:, 2:6]  # (x, y, z, w)
    obst_dist = point_cloud.compute_nearest_distances(drone_pos, drone_rot[:, [3, 0, 1, 2]])
    obst_reward = torch.where(
        obst_dist < cfg.obst_threshold,
        obst_dist * cfg.obstacle_strength,
        torch.zeros_like(obst_dist)
    )

    # === Total reward === (lines 759-772)
    reward = (
        survive_reward +
        obst_reward +
        pose_penalty +
        lin_vel_reward +
        heading_reward +
        target_reward +
        action_penalty +
        action_change_penalty +
        smooth_penalty +
        height_penalty +
        out_map_penalty +
        wp_reward
    )

    return reward


def check_termination(
    obs_buf: torch.Tensor,
    privilege_obs_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    cfg: EnvConfig,
    early_termination: bool = True,
) -> torch.Tensor:
    """
    Check early termination conditions.

    Source: drone_long_traj.py lines 775-790

    Returns:
        (N,) bool tensor, True = should reset
    """
    # === 1. Sanity check (NaN/Inf) === (line 776)
    condition_sanity = ~((~torch.isnan(obs_buf) & ~torch.isinf(obs_buf)).all(dim=1))

    # === 2. Body rate too high === (line 777)
    condition_body_rate = torch.linalg.norm(privilege_obs_buf[:, 10:13], dim=1) > cfg.body_rate_threshold

    # === 3. Height out of bounds === (line 778)
    condition_height = (privilege_obs_buf[:, 2] > 4.0) | (privilege_obs_buf[:, 2] < 0.0)

    # === 4. Out of map bounds === (lines 779-784)
    lf = cfg.map_limit_fact
    condition_out_of_bounds = (
        (privilege_obs_buf[:, 0] < lf * cfg.x_min) |
        (privilege_obs_buf[:, 0] > lf * cfg.x_max) |
        (privilege_obs_buf[:, 1] < lf * cfg.y_min) |
        (privilege_obs_buf[:, 1] > lf * cfg.y_max)
    )

    # === 5. Episode timeout === (line 785)
    condition_timeout = progress_buf > cfg.episode_length - 1

    # === Combine conditions === (lines 786-788)
    combined = condition_body_rate | condition_timeout
    if early_termination:
        combined = combined | condition_height | condition_out_of_bounds | condition_sanity

    return combined
