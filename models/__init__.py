"""
Models Module.

Contains neural network models for drone navigation:
- vel_net: Velocity estimation network
- (future: policy network)
"""

# Velocity network exports
from .vel_net import (
    VELO_NET,
    VelObsHistBuffer,
    quaternion_to_rot6d,
    quaternion_to_rotation_matrix,
    rot6d_to_rotation_matrix,
    build_vel_observation,
    build_vel_observation_from_quat,
    extract_vel_obs_components,
    VelObsIndices,
)

__all__ = [
    # Velocity network
    'VELO_NET',
    'VelObsHistBuffer',
    'quaternion_to_rot6d',
    'quaternion_to_rotation_matrix',
    'rot6d_to_rotation_matrix',
    'build_vel_observation',
    'build_vel_observation_from_quat',
    'extract_vel_obs_components',
    'VelObsIndices',
]
