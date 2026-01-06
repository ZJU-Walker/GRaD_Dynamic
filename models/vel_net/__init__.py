"""
Velocity Network Module.

Auto-regressive GRU-based network for drone velocity estimation with PINN loss.

Components:
- VELO_NET: Main velocity network (GRU-based)
- CompactEncoder: MobileNetV3-Small based visual encoder
- DualEncoder: Combined RGB + Depth encoder
- VelObsHistBuffer: History buffer for observations
- Utilities: Rot6D conversion, observation building
"""

from .vel_net import VELO_NET
from .vel_obs_buffer import VelObsHistBuffer
from .vel_obs_utils import (
    quaternion_to_rot6d,
    quaternion_to_rotation_matrix,
    rot6d_to_rotation_matrix,
    build_vel_observation,
    build_vel_observation_from_quat,
    extract_vel_obs_components,
    VelObsIndices,
)
from .visual_encoder import (
    CompactEncoder,
    DualEncoder,
    preprocess_image,
)

__all__ = [
    # Model
    'VELO_NET',
    # Visual Encoder
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
]
