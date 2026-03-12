"""
Dataset for VelNetV2 training.

Extends existing dataset pattern with additional ground truth fields:
  - Optical flow (precomputed RAFT-Small)
  - Angular velocity GT (from quaternion derivatives)
  - Translation direction GT + scale GT
  - Confidence GT (based on velocity magnitude)
  - Gyroscope data (body-frame angular velocity + augmentation)

Same telemetry.npz data, same backbone_features.npz, plus optical_flow.npz.

IMU FUSION: Full IMU (accel + gyro) both augmented with noise.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import glob

from models.vel_net_body_legacy.vel_obs_utils_body import (
    quaternion_to_rotation_matrix,
    transform_worldvel_to_bodyvel,
)
from training.vel_net_body.dataset import IMUAugmentation


def compute_angular_velocity_body(orientations: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute body-frame angular velocity from consecutive quaternions.

    Args:
        orientations: (N, 4) xyzw quaternions
        dt: Time step between frames

    Returns:
        omega_body: (N-1, 3) body-frame angular velocity

    Method:
        delta_q = q[t+1] * conj(q[t])
        omega_world = 2 * delta_q.xyz / dt (small angle approximation)
        omega_body = R[t]^T @ omega_world
    """
    N = orientations.shape[0]
    omega_body = np.zeros((N - 1, 3), dtype=np.float32)

    for i in range(N - 1):
        q0 = orientations[i]    # xyzw
        q1 = orientations[i + 1]  # xyzw

        # Conjugate of q0: [-x, -y, -z, w]
        q0_conj = np.array([-q0[0], -q0[1], -q0[2], q0[3]])

        # delta_q = q1 * conj(q0) (Hamilton product)
        # Result represents rotation from frame t to frame t+1 in world frame
        dq = _quaternion_multiply(q1, q0_conj)

        # Normalize
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 1e-8:
            dq = dq / dq_norm

        # Ensure w > 0 for consistent small angle
        if dq[3] < 0:
            dq = -dq

        # World frame angular velocity (small angle: omega ≈ 2 * dq.xyz / dt)
        omega_world = 2.0 * dq[:3] / dt

        # Transform to body frame: omega_body = R^T @ omega_world
        R = _quaternion_to_rotation_matrix_np(q0)
        omega_body[i] = R.T @ omega_world

    return omega_body


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in xyzw format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quaternion_to_rotation_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix (numpy)."""
    q = quat / (np.linalg.norm(quat) + 1e-8)
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R


class VelNetV2Dataset(Dataset):
    """
    Sequence-based dataset for VelNetV2 training.

    Returns per chunk:
      - rgb_backbone_features: (L, 576) from backbone_features.npz (RGB only)
      - optical_flows: (L, 2, H, W) from optical_flow.npz
      - velocities_gt: (L, 3) body frame GT velocity
      - initial_prev_vel: (3,) body frame prev_vel for first frame
      - accel_gt / accel_aug: (L, 3) body frame accelerometer
      - gyro_gt / gyro_aug: (L, 3) body frame gyroscope
      - actions: (L, 4) [roll_rate, pitch_rate, yaw_rate, thrust]
      - angular_velocity_gt: (L, 3) body frame angular velocity
      - translation_direction_gt: (L, 3) = vel_body / ||vel_body||
      - translation_scale_gt: (L, 1) = ||vel_body||
      - confidence_gt: (L, 1) = 1 when ||vel_body|| > 0.05, else 0
      - orientations: (L, 4) quaternions xyzw (for coordinate transforms only)

    Args:
        data_dir: Path to sequences directory
        seq_length: Length of sequence chunks (None = full sequence)
        stride: Stride between sequence chunks
        imu_augmentation: Whether to apply IMU noise augmentation
        dt: Time step (default: 1/30)
        min_vel_threshold: Minimum velocity for valid direction (default: 0.05 m/s)
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        imu_augmentation: bool = True,
        dt: float = 1.0 / 30.0,
        min_vel_threshold: float = 0.05,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        self.dt = dt
        self.min_vel_threshold = min_vel_threshold

        # IMU augmentation (separate instances for accel and gyro)
        self.accel_aug = IMUAugmentation(enabled=imu_augmentation)
        self.gyro_aug = IMUAugmentation(enabled=imu_augmentation)

        # Find all sequences
        self.sequence_dirs = sorted(glob.glob(str(self.data_dir / "seq_*")))
        if len(self.sequence_dirs) == 0:
            raise ValueError(f"No sequences found in {data_dir}")

        # Load all sequence data
        self.sequences = []
        self._load_sequences()

        # Build index of chunks
        self.chunks = []
        self._build_chunk_index()

        print(f"[VelNetV2Dataset] {len(self.sequences)} sequences, {len(self.chunks)} chunks")

    def _load_sequences(self):
        """Load all sequence data into memory."""
        for seq_dir in self.sequence_dirs:
            seq_path = Path(seq_dir)

            telemetry = np.load(seq_path / "telemetry.npz")

            # Compute body-frame angular velocity from quaternions
            orientations = telemetry['orientations'].astype(np.float32)
            omega_body = compute_angular_velocity_body(orientations, self.dt)
            # Pad first frame with second frame's value (omega_body has N-1 entries)
            omega_body_padded = np.zeros((len(orientations), 3), dtype=np.float32)
            omega_body_padded[1:] = omega_body
            omega_body_padded[0] = omega_body[0] if len(omega_body) > 0 else np.zeros(3)

            seq_data = {
                'path': str(seq_path),
                'timestamps': telemetry['timestamps'].astype(np.float32),
                'positions': telemetry['positions'].astype(np.float32),
                'velocities': telemetry['velocities'].astype(np.float32),
                'orientations': orientations,
                'actions': telemetry['actions'].astype(np.float32),
                'angular_velocity_body': omega_body_padded,
                'n_frames': len(telemetry['timestamps']),
            }
            self.sequences.append(seq_data)

    def _build_chunk_index(self):
        """Build index of sequence chunks."""
        for seq_idx, seq in enumerate(self.sequences):
            n_frames = seq['n_frames']

            if self.seq_length is None:
                # Use full sequence (skip first frame since we need prev)
                self.chunks.append({
                    'seq_idx': seq_idx,
                    'start': 1,
                    'end': n_frames,
                })
            else:
                for start in range(1, n_frames - self.seq_length + 1, self.stride):
                    end = min(start + self.seq_length, n_frames)
                    if end - start >= 2:
                        self.chunks.append({
                            'seq_idx': seq_idx,
                            'start': start,
                            'end': end,
                        })

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        seq = self.sequences[chunk['seq_idx']]
        start, end = chunk['start'], chunk['end']
        length = end - start

        orientations = torch.from_numpy(seq['orientations'][start:end])
        actions = torch.from_numpy(seq['actions'][start:end])

        # =====================================================================
        # Body frame velocities
        # =====================================================================
        velocities_world = torch.from_numpy(seq['velocities'][start:end])
        initial_prev_vel_world = torch.from_numpy(seq['velocities'][start - 1])

        velocities_gt = transform_worldvel_to_bodyvel(velocities_world, orientations)

        initial_prev_vel_quat = torch.from_numpy(seq['orientations'][start - 1])
        initial_prev_vel = transform_worldvel_to_bodyvel(initial_prev_vel_world, initial_prev_vel_quat)

        # =====================================================================
        # Acceleration (body frame) — same as v1
        # =====================================================================
        vel_prev = np.concatenate([
            seq['velocities'][start - 1:start],
            seq['velocities'][start:end - 1],
        ], axis=0)
        vel_curr = seq['velocities'][start:end]
        accel_world = (vel_curr - vel_prev) / self.dt

        accel_gt = transform_worldvel_to_bodyvel(
            torch.from_numpy(accel_world.astype(np.float32)),
            orientations,
        ).numpy()

        accel_aug = self.accel_aug(accel_gt)

        # =====================================================================
        # Gyroscope (body frame angular velocity)
        # =====================================================================
        gyro_gt = seq['angular_velocity_body'][start:end].copy()
        gyro_aug = self.gyro_aug(gyro_gt)

        # =====================================================================
        # Angular velocity GT (body frame)
        # =====================================================================
        angular_velocity_gt = torch.from_numpy(gyro_gt.astype(np.float32))

        # =====================================================================
        # Translation direction + scale + confidence
        # =====================================================================
        vel_body_np = velocities_gt.numpy()
        vel_norms = np.linalg.norm(vel_body_np, axis=1, keepdims=True)  # (L, 1)

        # Direction: normalized velocity, default [1,0,0] for near-zero
        direction_gt = np.zeros_like(vel_body_np)
        valid_mask = vel_norms.squeeze() > self.min_vel_threshold
        direction_gt[valid_mask] = vel_body_np[valid_mask] / vel_norms[valid_mask]
        direction_gt[~valid_mask] = np.array([1.0, 0.0, 0.0])  # forward default

        # Scale: velocity magnitude
        scale_gt = vel_norms  # (L, 1)

        # Confidence: 1 when moving, 0 when near-zero
        confidence_gt = (vel_norms > self.min_vel_threshold).astype(np.float32)  # (L, 1)

        frame_indices = list(range(start, end))

        return {
            'seq_path': seq['path'],
            'start_idx': start,
            'length': length,
            'frame_indices': frame_indices,
            'orientations': orientations,
            'actions': actions,
            # Body frame velocity
            'velocities_gt': velocities_gt,
            'initial_prev_vel': initial_prev_vel,
            # Acceleration (body frame)
            'accel_gt': torch.from_numpy(accel_gt.astype(np.float32)),
            'accel_aug': torch.from_numpy(accel_aug),
            # Gyroscope (body frame)
            'gyro_gt': torch.from_numpy(gyro_gt.astype(np.float32)),
            'gyro_aug': torch.from_numpy(gyro_aug),
            # Geometry GT
            'angular_velocity_gt': angular_velocity_gt,
            'translation_direction_gt': torch.from_numpy(direction_gt.astype(np.float32)),
            'translation_scale_gt': torch.from_numpy(scale_gt.astype(np.float32)),
            'confidence_gt': torch.from_numpy(confidence_gt),
        }


def collate_sequences_v2(batch: List[dict]) -> Dict[str, any]:
    """Collate function for variable-length sequences (returns lists)."""
    return {
        'seq_paths': [b['seq_path'] for b in batch],
        'start_indices': [b['start_idx'] for b in batch],
        'lengths': [b['length'] for b in batch],
        'frame_indices': [b['frame_indices'] for b in batch],
        'orientations': [b['orientations'] for b in batch],
        'actions': [b['actions'] for b in batch],
        # Body frame velocity
        'velocities_gt': [b['velocities_gt'] for b in batch],
        'initial_prev_vels': [b['initial_prev_vel'] for b in batch],
        # Acceleration
        'accel_gt': [b['accel_gt'] for b in batch],
        'accel_aug': [b['accel_aug'] for b in batch],
        # Gyroscope
        'gyro_gt': [b['gyro_gt'] for b in batch],
        'gyro_aug': [b['gyro_aug'] for b in batch],
        # Geometry GT
        'angular_velocity_gt': [b['angular_velocity_gt'] for b in batch],
        'translation_direction_gt': [b['translation_direction_gt'] for b in batch],
        'translation_scale_gt': [b['translation_scale_gt'] for b in batch],
        'confidence_gt': [b['confidence_gt'] for b in batch],
    }


def create_dataloaders_v2(
    data_dir: str,
    seq_length: Optional[int] = 64,
    stride: Optional[int] = 32,
    batch_size: int = 8,
    val_ratio: float = 0.1,
    imu_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to sequences directory
        seq_length: Length of sequence chunks (None = full)
        stride: Stride between chunks
        batch_size: Batch size
        val_ratio: Fraction for validation
        imu_augmentation: Apply IMU noise (training only)

    Returns:
        (train_loader, val_loader)
    """
    # Training dataset with IMU augmentation
    train_full = VelNetV2Dataset(data_dir, seq_length, stride, imu_augmentation=imu_augmentation)
    # Validation dataset without augmentation
    val_full = VelNetV2Dataset(data_dir, seq_length, stride, imu_augmentation=False)

    n_total = len(train_full)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    torch.manual_seed(42)
    perm = torch.randperm(n_total).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sequences_v2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequences_v2,
        drop_last=False,
    )

    print(f"[DataLoaders V2] Train: {n_train} chunks, Val: {n_val} chunks")
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/vel_net/sequences')
    args = parser.parse_args()

    print("Testing VelNetV2Dataset...")

    try:
        dataset = VelNetV2Dataset(args.data_dir, seq_length=64, stride=32, imu_augmentation=True)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample keys: {sorted(sample.keys())}")
        print(f"  length: {sample['length']}")
        print(f"  velocities_gt: {sample['velocities_gt'].shape}")
        print(f"  accel_gt: {sample['accel_gt'].shape}")
        print(f"  gyro_gt: {sample['gyro_gt'].shape}")
        print(f"  angular_velocity_gt: {sample['angular_velocity_gt'].shape}")
        print(f"  translation_direction_gt: {sample['translation_direction_gt'].shape}")
        print(f"  translation_scale_gt: {sample['translation_scale_gt'].shape}")
        print(f"  confidence_gt: {sample['confidence_gt'].shape}")

        # Check direction is unit vector
        dir_norms = torch.norm(sample['translation_direction_gt'], dim=1)
        print(f"  direction norms (should be ~1): min={dir_norms.min():.4f}, max={dir_norms.max():.4f}")

        print("\nDataset test passed!")
    except Exception as e:
        import traceback
        print(f"Dataset test failed: {e}")
        traceback.print_exc()
