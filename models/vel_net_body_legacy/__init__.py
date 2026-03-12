"""
Velocity Network Module (Body Frame).

Auto-regressive GRU-based network for drone velocity estimation in BODY FRAME.

Body Frame Convention (FLU - Forward-Left-Up):
- v_x: velocity along drone's forward axis
- v_y: velocity along drone's left axis
- v_z: velocity along drone's up axis

Components:
- VELO_NET_BODY: Main velocity network (GRU-based, body frame output)
- VelObsHistBuffer: History buffer for observations
- Utilities: Rot6D conversion, observation building, world-to-body transformation
"""

from .vel_net_body import VELO_NET_BODY
from .vel_obs_buffer import VelObsHistBuffer
from .vel_obs_utils_body import (
    quaternion_to_rot6d,
    quaternion_to_rotation_matrix,
    rot6d_to_rotation_matrix,
    build_vel_observation,
    build_vel_observation_from_quat,
    extract_vel_obs_components,
    VelObsIndices,
    transform_worldvel_to_bodyvel,
)
# Import visual encoder from original vel_net (shared)
from models.vel_net.visual_encoder import (
    CompactEncoder,
    DualEncoder,
    preprocess_image,
)

__all__ = [
    # Model
    'VELO_NET_BODY',
    # Visual Encoder (shared from vel_net)
    'CompactEncoder',
    'DualEncoder',
    'preprocess_image',
    # Buffer
    'VelObsHistBuffer',
    # Utilities
    'quaternion_to_rot6d',
    'quaternion_to_rotation_matrix',
    'rot6d_to_rotation_matrix',
    'build_vel_observation',
    'build_vel_observation_from_quat',
    'extract_vel_obs_components',
    'VelObsIndices',
    'transform_worldvel_to_bodyvel',
]
