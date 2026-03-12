"""
VelNet V2: Multi-branch velocity estimation network.

Architecture:
  - GeometryBranch: Angular velocity + translation direction (RGB + optical flow + IMU)
  - DynamicsBranch: Metric scale recovery (geometry outputs + IMU + actions)
  - FusionHead: Combine geometry + dynamics → final velocity
  - FlowEncoder: CNN encoder for dense optical flow fields

No Rot6D input — orientation-agnostic for real deployment.
RGB only — no depth encoder/features.
Dense optical flow via RAFT-Small (precomputed).
Full IMU (accel + gyro) in body frame.
"""

from models.vel_net_v2.flow_encoder import FlowEncoder
from models.vel_net_v2.geometry_branch import GeometryBranch
from models.vel_net_v2.dynamics_branch import DynamicsBranch
from models.vel_net_v2.fusion_head import FusionHead
from models.vel_net_v2.vel_net_v2 import VelNetV2
from models.vel_net_v2.losses import GeometryLoss, DynamicsLoss, JointLoss
