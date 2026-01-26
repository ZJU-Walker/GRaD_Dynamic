"""Single unified configuration for expert environment."""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EnvConfig:
    """
    Unified configuration for expert environment.
    Contains reward weights, map settings, and env parameters.
    """

    # ==========================================================================
    # Map Settings
    # ==========================================================================
    map_name: str = "gate_mid"
    gs_folder: str = "sv_1007_gate_mid"
    ply_file: str = "sv_1007_gate_mid.ply"

    # Start state
    start_pos: List[float] = field(default_factory=lambda: [0., 0., 1.25])
    start_rotation: List[float] = field(default_factory=lambda: [0., 0., 0., 1.])  # (x,y,z,w)

    # Navigation targets
    waypoints: List[List[float]] = field(default_factory=lambda: [
        [5.8, -0.1, 1.6],
        [7.6, 0.7, 1.3],
        [9.7, 1.5, 0.9],
        [11.8, 0, 1.5],
    ])
    target_pos: List[float] = field(default_factory=lambda: [11.8, 0, 1.5])

    # Map bounds
    x_min: float = -1.5
    x_max: float = 13.0
    y_min: float = -2.5
    y_max: float = 2.5

    # Coordinate offsets
    gs_origin_offset: List[float] = field(default_factory=lambda: [-6.0, 0.0, 0.0])
    point_cloud_offset: List[float] = field(default_factory=lambda: [-6.0, 0.0, 0.0])

    # ==========================================================================
    # Reward Weights
    # ==========================================================================
    survive_reward: float = 8.0

    # Dynamics penalties
    lin_vel_penalty: float = -2.0
    action_penalty: float = -1.0
    action_change_penalty: float = -1.0
    smooth_penalty: float = -1.0
    pose_penalty: float = -0.5
    height_penalty: float = -2.0

    # Navigation rewards
    heading_strength: float = 0.7 # 0.5
    waypoint_strength: float = 5.2 # 4.0
    target_factor: float = -4.0
    out_map_penalty: float = -1.5
    obstacle_strength: float = 1.0

    # Reward thresholds
    target_height: float = 1.3
    obst_threshold: float = 0.5

    # Dynamic obstacle reward (for DynamicDroneEnv)
    dynamic_obst_threshold: float = 2   # Distance threshold to apply reward (meters)
    dynamic_obst_strength: float = 5.0    # Reward multiplier (higher = stronger avoidance)

    # Phase 2 danger-aware reward weights (curriculum training)
    k_retreat: float = 8.0              # Retreat reward weight (encourage increasing distance when danger)
    k_no_forward: float = 5.0           # No-forward penalty weight (discourage forward when danger)
    k_backward: float = 0.5             # Backward regularization weight (prevent always reversing)
    danger_dist_threshold: float = 2.0  # Distance threshold for danger detection (meters)
    danger_ttc_threshold: float = 3.0   # TTC threshold for danger detection (seconds)

    # ==========================================================================
    # Environment Parameters
    # ==========================================================================
    episode_length: int = 1000

    # Termination thresholds
    obst_collision_limit: float = 0.20
    body_rate_threshold: float = 15.0
    map_limit_fact: float = 1.25

    # Noise
    obs_noise_level: float = 0.1

    # Action scaling
    br_delay_factor: float = 0.8
    thrust_delay_factor: float = 0.7
    br_action_strength: float = 0.5
    thrust_action_strength: float = 0.25


# =============================================================================
# Preset Configurations for Different Maps
# =============================================================================

def get_config(map_name: str = "gate_mid") -> EnvConfig:
    """Get EnvConfig with map-specific settings."""

    if map_name == "gate_mid":
        return EnvConfig(
            map_name="gate_mid",
            gs_folder="sv_1007_gate_mid",
            ply_file="sv_1007_gate_mid.ply",
            start_pos=[0.0, 0.0, 1.25],
            waypoints=[
                [5.8, -0.1, 1.6],
                [7.6, 0.7, 1.3],
                [9.7, 1.5, 0.9],
                [11.8, 0, 1.5],
            ],
            target_pos=[11.8, 0, 1.5],
        )

    elif map_name == "gate_right":
        return EnvConfig(
            map_name="gate_right",
            gs_folder="sv_917_3_right_nerfstudio",
            ply_file="sv_917_3_right_nerfstudio.ply",
            start_pos=[0.0, 0.0, 1.3],
            waypoints=[
                [6.0, -1.2, 1.5],
                [7.8, 0.6, 1.1],
                [9.7, 1.4, 0.7],
                [11.0, 0.5, 1.3],
            ],
            target_pos=[11.0, 0.5, 1.3],
        )

    elif map_name == "gate_left":
        return EnvConfig(
            map_name="gate_left",
            gs_folder="sv_917_3_left_nerfstudio",
            ply_file="sv_917_3_left_nerfstudio.ply",
            start_pos=[0.0, 0.0, 1.2],
            waypoints=[
                [-0.2, 1.2, 1.4],
                [3.7, 1.2, 0.6],
            ],
            target_pos=[7.0, -2.0, 1.2],
        )

    else:
        raise ValueError(f"Unknown map: {map_name}. Available: gate_mid, gate_right, gate_left")