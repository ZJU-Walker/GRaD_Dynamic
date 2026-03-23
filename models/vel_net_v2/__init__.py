"""
VelNet V2: Multi-branch velocity estimation network.

Architecture:
  - RotationNet: Angular velocity from optical flow spatial structure (new)
  - GeometryBranch (TranslationBranch): Translation direction (RGB + flow + IMU + omega)
  - DynamicsBranch: Metric scale recovery (geometry outputs + IMU + actions)
  - FusionHead: Combine geometry + dynamics → final velocity
  - FlowEncoder: CNN encoder for dense optical flow fields

No Rot6D input — orientation-agnostic for real deployment.
RGB only — no depth encoder/features.
Dense optical flow via RAFT-Small (precomputed).
Full IMU accel in body frame. Gyro replaced by RotationNet.
"""

from models.vel_net_v2.flow_encoder import FlowEncoder
from models.vel_net_v2.rotation_net import RotationNet
from models.vel_net_v2.geometry_branch import GeometryBranch
from models.vel_net_v2.dynamics_branch import DynamicsBranch
from models.vel_net_v2.fusion_head import FusionHead
from models.vel_net_v2.vel_net_v2 import VelNetV2
from models.vel_net_v2.losses import OmegaLoss, TranslationLoss, JointLoss
# Backward-compatible aliases
from models.vel_net_v2.losses import GeometryLoss, DynamicsLoss
