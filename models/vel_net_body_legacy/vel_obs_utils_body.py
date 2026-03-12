"""
Velocity Observation Utilities for Body Frame.

Helper functions for building velocity network observations:
- Rot6D representation conversion
- Observation vector construction
- World to body frame velocity transformation

BODY FRAME CONVENTION (FLU - Forward-Left-Up):
- v_x: velocity along drone's forward axis (+ = moving forward)
- v_y: velocity along drone's left axis (+ = moving left)
- v_z: velocity along drone's up axis (+ = moving up)

IMU FUSION MODE (76 dims, no action):
- Adds acceleration (IMU) to observation for physics-informed velocity estimation
- Network uses corrupted IMU + vision to correct velocity prediction
- Action removed from observations
- All velocities are in BODY FRAME
"""

import torch
from typing import Optional


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion tensor (B, 4) in xyzw format

    Returns:
        Rotation matrix (B, 3, 3)
    """
    # Normalize quaternion
    quat = quat / (quat.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Rotation matrix from quaternion
    # First row
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)

    # Second row
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)

    # Third row
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    # Stack into rotation matrix
    rot_matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return rot_matrix


def quaternion_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Rot6D representation.

    Rot6D uses the first two columns of the rotation matrix, flattened.
    This representation is continuous and suitable for neural networks.

    Reference: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        quat: Quaternion tensor (B, 4) in xyzw format
              OR (4,) for single quaternion

    Returns:
        Rot6D representation (B, 6) or (6,) for single input
    """
    single_input = quat.ndim == 1
    if single_input:
        quat = quat.unsqueeze(0)

    # Get rotation matrix
    rot_matrix = quaternion_to_rotation_matrix(quat)  # (B, 3, 3)

    # Take first two columns and flatten
    # Column 0: rot_matrix[:, :, 0] -> (B, 3)
    # Column 1: rot_matrix[:, :, 1] -> (B, 3)
    col0 = rot_matrix[:, :, 0]  # (B, 3)
    col1 = rot_matrix[:, :, 1]  # (B, 3)

    # Concatenate to get Rot6D
    rot6d = torch.cat([col0, col1], dim=-1)  # (B, 6)

    if single_input:
        rot6d = rot6d.squeeze(0)

    return rot6d


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert Rot6D back to rotation matrix using Gram-Schmidt orthogonalization.

    Args:
        rot6d: Rot6D representation (B, 6)

    Returns:
        Rotation matrix (B, 3, 3)
    """
    # Split into two column vectors
    col0 = rot6d[:, :3]  # (B, 3)
    col1 = rot6d[:, 3:]  # (B, 3)

    # Gram-Schmidt orthogonalization
    # Normalize first column
    col0 = col0 / (col0.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    # Make second column orthogonal to first
    dot = (col0 * col1).sum(dim=-1, keepdim=True)
    col1 = col1 - dot * col0
    col1 = col1 / (col1.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    # Third column is cross product
    col2 = torch.cross(col0, col1, dim=-1)

    # Stack into rotation matrix
    rot_matrix = torch.stack([col0, col1, col2], dim=-1)

    return rot_matrix


def transform_worldvel_to_bodyvel(vel_world: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Transform velocity from world frame to body frame.

    Body frame convention (FLU - Forward-Left-Up):
    - v_x: velocity along drone's forward axis (+ = moving forward)
    - v_y: velocity along drone's left axis (+ = moving left)
    - v_z: velocity along drone's up axis (+ = moving up)

    The quaternion represents body->world rotation, so we use R^T to go world->body.

    Args:
        vel_world: (B, 3) or (3,) velocity in world frame
        quat_xyzw: (B, 4) or (4,) quaternion [x, y, z, w] (body->world rotation)

    Returns:
        vel_body: velocity in body frame (same shape as input)
    """
    # Handle single input (3,) vs batch (B, 3)
    single_input = vel_world.ndim == 1
    if single_input:
        vel_world = vel_world.unsqueeze(0)
        quat_xyzw = quat_xyzw.unsqueeze(0)

    # Get rotation matrix (body->world)
    rot_matrix = quaternion_to_rotation_matrix(quat_xyzw)  # (B, 3, 3)

    # Transform: world->body = R^T @ vel_world
    # R is body->world, so R^T (transpose) is world->body
    vel_body = torch.bmm(rot_matrix.transpose(-1, -2), vel_world.unsqueeze(-1)).squeeze(-1)

    if single_input:
        vel_body = vel_body.squeeze(0)

    return vel_body


def build_vel_observation(rot6d: torch.Tensor,
                          prev_vel: torch.Tensor,
                          rgb_feat: Optional[torch.Tensor] = None,
                          depth_feat: Optional[torch.Tensor] = None,
                          accel: Optional[torch.Tensor] = None,
                          device: Optional[str] = None,
                          rgb_feat_dim: int = 32,
                          depth_feat_dim: int = 32) -> torch.Tensor:
    """
    Build 76-dim velocity observation vector (IMU fusion mode, no action).

    Observation structure:
        [0:6]   - Rot6D rotation (6 dims)
        [6:9]   - Previous velocity (3 dims) - auto-regressive term
        [9:41]  - RGB features (32 dims)
        [41:73] - Depth features (32 dims)
        [73:76] - Acceleration/IMU (3 dims) - for physics-informed fusion

    Args:
        rot6d: Rotation in Rot6D format (B, 6)
        prev_vel: Previous velocity estimate (B, 3)
        rgb_feat: RGB visual features (B, 32), zeros if None
        depth_feat: Depth visual features (B, 32), zeros if None
        accel: Acceleration/IMU reading (B, 3), zeros if None
        device: Device for output tensor
        rgb_feat_dim: Dimension of RGB features (default: 32)
        depth_feat_dim: Dimension of depth features (default: 32)

    Returns:
        Velocity observation (B, 76)
    """
    B = rot6d.shape[0]

    if device is None:
        device = rot6d.device

    # Handle optional visual features
    if rgb_feat is None:
        rgb_feat = torch.zeros(B, rgb_feat_dim, device=device)
    if depth_feat is None:
        depth_feat = torch.zeros(B, depth_feat_dim, device=device)
    if accel is None:
        accel = torch.zeros(B, 3, device=device)

    # Validate shapes
    assert rot6d.shape == (B, 6), f"rot6d shape mismatch: {rot6d.shape}"
    assert prev_vel.shape == (B, 3), f"prev_vel shape mismatch: {prev_vel.shape}"
    assert rgb_feat.shape == (B, rgb_feat_dim), f"rgb_feat shape mismatch: {rgb_feat.shape}"
    assert depth_feat.shape == (B, depth_feat_dim), f"depth_feat shape mismatch: {depth_feat.shape}"
    assert accel.shape == (B, 3), f"accel shape mismatch: {accel.shape}"

    # Concatenate all components
    vel_obs = torch.cat([
        rot6d,       # 6 dims
        prev_vel,    # 3 dims
        rgb_feat,    # 32 dims
        depth_feat,  # 32 dims
        accel,       # 3 dims (IMU)
    ], dim=-1)

    total_dim = 6 + 3 + rgb_feat_dim + depth_feat_dim + 3
    assert vel_obs.shape == (B, total_dim), f"vel_obs shape mismatch: {vel_obs.shape}"

    return vel_obs


def build_vel_observation_from_quat(quat: torch.Tensor,
                                    prev_vel: torch.Tensor,
                                    rgb_feat: Optional[torch.Tensor] = None,
                                    depth_feat: Optional[torch.Tensor] = None,
                                    accel: Optional[torch.Tensor] = None,
                                    rgb_feat_dim: int = 32,
                                    depth_feat_dim: int = 32) -> torch.Tensor:
    """
    Build velocity observation from quaternion (convenience function).

    Args:
        quat: Quaternion (B, 4) in xyzw format
        prev_vel: Previous velocity (B, 3)
        rgb_feat: RGB features (B, 32), optional
        depth_feat: Depth features (B, 32), optional
        accel: Acceleration/IMU reading (B, 3), optional
        rgb_feat_dim: Dimension of RGB features (default: 32)
        depth_feat_dim: Dimension of depth features (default: 32)

    Returns:
        Velocity observation (B, 76)
    """
    rot6d = quaternion_to_rot6d(quat)
    return build_vel_observation(rot6d, prev_vel,
                                 rgb_feat, depth_feat, accel, device=quat.device,
                                 rgb_feat_dim=rgb_feat_dim, depth_feat_dim=depth_feat_dim)


# Observation component indices for easy access
class VelObsIndices:
    """Indices for velocity observation components (76 dims total with IMU, no action)."""
    ROT6D_START = 0
    ROT6D_END = 6           # 6 dims

    PREV_VEL_START = 6
    PREV_VEL_END = 9        # 3 dims

    RGB_FEAT_START = 9
    RGB_FEAT_END = 41       # 32 dims

    DEPTH_FEAT_START = 41
    DEPTH_FEAT_END = 73     # 32 dims

    ACCEL_START = 73
    ACCEL_END = 76          # 3 dims (IMU acceleration)

    TOTAL_DIM = 76          # 6 + 3 + 32 + 32 + 3 = 76

    # Feature dimensions
    RGB_FEAT_DIM = 32
    DEPTH_FEAT_DIM = 32
    ACCEL_DIM = 3


def extract_vel_obs_components(vel_obs: torch.Tensor) -> dict:
    """
    Extract individual components from velocity observation.

    Args:
        vel_obs: Velocity observation (B, 76) with IMU, no action

    Returns:
        Dictionary with keys: rot6d, prev_vel, rgb_feat, depth_feat, accel
    """
    idx = VelObsIndices
    result = {
        'rot6d': vel_obs[:, idx.ROT6D_START:idx.ROT6D_END],          # (B, 6)
        'prev_vel': vel_obs[:, idx.PREV_VEL_START:idx.PREV_VEL_END],  # (B, 3)
        'rgb_feat': vel_obs[:, idx.RGB_FEAT_START:idx.RGB_FEAT_END],  # (B, 32)
        'depth_feat': vel_obs[:, idx.DEPTH_FEAT_START:idx.DEPTH_FEAT_END],  # (B, 32)
    }
    # Include accel if observation has 76 dims
    if vel_obs.shape[1] >= idx.ACCEL_END:
        result['accel'] = vel_obs[:, idx.ACCEL_START:idx.ACCEL_END]  # (B, 3)
    return result


if __name__ == '__main__':
    # Test quaternion to Rot6D conversion
    print("Testing quaternion_to_rot6d...")

    # Identity quaternion (no rotation): [0, 0, 0, 1]
    quat_identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    rot6d = quaternion_to_rot6d(quat_identity)
    print(f"Identity quat -> Rot6D: {rot6d}")
    # Expected: [1, 0, 0, 0, 1, 0] (first two columns of identity matrix)

    # Test batch conversion
    quats = torch.randn(4, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize
    rot6ds = quaternion_to_rot6d(quats)
    print(f"Batch conversion: {quats.shape} -> {rot6ds.shape}")

    # Test build_vel_observation (76 dims with IMU, no action)
    print("\nTesting build_vel_observation (76 dims with IMU, no action)...")
    B = 4
    rot6d = torch.randn(B, 6)
    prev_vel = torch.randn(B, 3)
    rgb_feat = torch.randn(B, 32)
    depth_feat = torch.randn(B, 32)
    accel = torch.randn(B, 3)  # IMU acceleration

    vel_obs = build_vel_observation(rot6d, prev_vel,
                                    rgb_feat, depth_feat, accel)
    print(f"Velocity observation shape: {vel_obs.shape}")  # Should be (4, 76)

    # Test extraction
    components = extract_vel_obs_components(vel_obs)
    print(f"Extracted components: {list(components.keys())}")
    print(f"Rot6D matches: {torch.allclose(components['rot6d'], rot6d)}")
    print(f"Prev_vel matches: {torch.allclose(components['prev_vel'], prev_vel)}")
    print(f"Accel matches: {torch.allclose(components['accel'], accel)}")
    print(f"RGB feat shape: {components['rgb_feat'].shape}")  # Should be (4, 32)
    print(f"Depth feat shape: {components['depth_feat'].shape}")  # Should be (4, 32)
    print(f"Accel shape: {components['accel'].shape}")  # Should be (4, 3)

    # Test with missing visual features and accel (default zeros)
    vel_obs_minimal = build_vel_observation(rot6d, prev_vel)
    print(f"Without visual features/accel: {vel_obs_minimal.shape}")  # Should be (4, 76)

    # Test transform_worldvel_to_bodyvel
    print("\nTesting transform_worldvel_to_bodyvel...")

    # Test 1: Identity quaternion (no rotation) - body vel should equal world vel
    quat_identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    vel_world = torch.tensor([[1.0, 2.0, 3.0]])
    vel_body = transform_worldvel_to_bodyvel(vel_world, quat_identity)
    print(f"Identity quat: world {vel_world[0].tolist()} -> body {vel_body[0].tolist()}")
    print(f"  Should be equal: {torch.allclose(vel_world, vel_body)}")

    # Test 2: 90 degree yaw rotation (rotate around Z axis)
    # quat for 90 deg yaw: [0, 0, sin(45), cos(45)] = [0, 0, 0.707, 0.707]
    quat_yaw90 = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]])
    vel_world = torch.tensor([[1.0, 0.0, 0.0]])  # Moving +X in world
    vel_body = transform_worldvel_to_bodyvel(vel_world, quat_yaw90)
    print(f"90deg yaw: world {vel_world[0].tolist()} -> body {vel_body[0].tolist()}")
    # When drone is rotated 90 deg yaw (facing +Y), world +X velocity appears as -Y in body frame
    print(f"  Expected ~[0, -1, 0]: {vel_body[0].tolist()}")

    # Test 3: Single input (non-batched)
    vel_world_single = torch.tensor([1.0, 2.0, 3.0])
    quat_single = torch.tensor([0.0, 0.0, 0.0, 1.0])
    vel_body_single = transform_worldvel_to_bodyvel(vel_world_single, quat_single)
    print(f"Single input shape: {vel_body_single.shape}")  # Should be (3,)

    # Test 4: Batch input
    B = 4
    vel_world_batch = torch.randn(B, 3)
    quat_batch = torch.randn(B, 4)
    quat_batch = quat_batch / quat_batch.norm(dim=-1, keepdim=True)
    vel_body_batch = transform_worldvel_to_bodyvel(vel_world_batch, quat_batch)
    print(f"Batch input shape: {vel_body_batch.shape}")  # Should be (4, 3)

    print("\nAll tests completed!")
