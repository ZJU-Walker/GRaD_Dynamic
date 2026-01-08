"""
PyTorch Dataset for Velocity Network Training.

Sequence-based dataset for scheduled sampling training.
Returns full sequences (or fixed-length chunks) instead of single frames.
This enables step-by-step training where predicted velocities can be
fed back as prev_vel (scheduled sampling).

IMU FUSION MODE:
- Computes acceleration from velocity derivatives
- Applies realistic IMU noise augmentation (bias, scale, noise, dropout)
- The network learns to correct corrupted IMU integration using vision
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

from models.vel_net.vel_obs_utils import quaternion_to_rot6d


class IMUAugmentation:
    """
    IMU noise augmentation to force the network to use vision for correction.

    If IMU is perfect, the network will do nothing. By corrupting the IMU data,
    we force the network to learn visual corrections.

    Augmentations (per sequence):
    1. Bias Drift: Constant offset added to all frames
    2. Scale Error: Multiplicative scale factor
    3. White Noise: Per-frame Gaussian noise (vibration)
    4. Sensor Dropout: Randomly zero out entire acceleration vector
    """

    def __init__(
        self,
        bias_range: float = 0.2,      # Uniform(-bias_range, bias_range) per axis
        scale_range: float = 0.05,    # Scale in [1-scale_range, 1+scale_range]
        noise_std: float = 0.1,       # White noise std
        dropout_prob: float = 0.1,    # Probability of zeroing accel
        enabled: bool = True,
    ):
        self.bias_range = bias_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.enabled = enabled

    def __call__(self, accel_gt: np.ndarray) -> np.ndarray:
        """
        Apply IMU augmentation to ground truth acceleration.

        Args:
            accel_gt: Ground truth acceleration (seq_len, 3)

        Returns:
            accel_aug: Corrupted acceleration (seq_len, 3)
        """
        if not self.enabled:
            return accel_gt.copy()

        seq_len = accel_gt.shape[0]
        accel_aug = accel_gt.copy()

        # 1. Bias drift: constant offset per sequence
        bias = np.random.uniform(-self.bias_range, self.bias_range, size=3)
        accel_aug = accel_aug + bias

        # 2. Scale error: multiplicative factor per sequence
        scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range, size=3)
        accel_aug = accel_aug * scale

        # 3. White noise: per-frame Gaussian noise
        noise = np.random.normal(0, self.noise_std, size=accel_aug.shape)
        accel_aug = accel_aug + noise

        # 4. Sensor dropout: randomly zero out entire frames
        dropout_mask = np.random.random(seq_len) < self.dropout_prob
        accel_aug[dropout_mask] = 0.0

        return accel_aug.astype(np.float32)


class VelNetDataset(Dataset):
    """
    Sequence-based dataset for scheduled sampling training with IMU fusion.

    Returns full sequences (or fixed-length chunks) instead of single frames.
    This enables step-by-step training where predicted velocities can be
    fed back as prev_vel (scheduled sampling).

    IMU FUSION: Computes acceleration from velocity derivatives and applies
    realistic noise augmentation to force the network to use vision.

    Args:
        data_dir: Path to sequences directory
        seq_length: Length of sequence chunks (None = full sequence)
        stride: Stride between sequence chunks (for overlapping)
        imu_augmentation: Whether to apply IMU noise augmentation
        dt: Time step for acceleration computation (default: 1/30 = 33ms)
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
        imu_augmentation: bool = True,
        dt: float = 1.0 / 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        self.dt = dt

        # IMU augmentation (disabled during validation)
        self.imu_aug = IMUAugmentation(enabled=imu_augmentation)

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

        print(f"[VelNetDataset] {len(self.sequences)} sequences, {len(self.chunks)} chunks")

    def _load_sequences(self):
        """Load all sequence data into memory."""
        for seq_dir in self.sequence_dirs:
            seq_path = Path(seq_dir)

            telemetry = np.load(seq_path / "telemetry.npz")

            seq_data = {
                'path': str(seq_path),
                'timestamps': telemetry['timestamps'].astype(np.float32),
                'positions': telemetry['positions'].astype(np.float32),
                'velocities': telemetry['velocities'].astype(np.float32),
                'orientations': telemetry['orientations'].astype(np.float32),
                'actions': telemetry['actions'].astype(np.float32),
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
                # Create overlapping chunks
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
        """
        Get a sequence chunk with IMU data.

        Returns:
            Dict with:
                - seq_path: path to sequence directory
                - start_idx: starting frame index
                - length: number of frames in chunk
                - orientations: (L, 4) quaternions xyzw
                - actions: (L, 4) actions
                - prev_actions: (L, 4) previous actions
                - velocities_gt: (L, 3) ground truth velocities
                - initial_prev_vel: (3,) GT prev_vel for first frame
                - accel_gt: (L, 3) ground truth acceleration (from velocity derivative)
                - accel_aug: (L, 3) augmented/corrupted acceleration (for training)
        """
        chunk = self.chunks[idx]
        seq = self.sequences[chunk['seq_idx']]
        start, end = chunk['start'], chunk['end']
        length = end - start

        orientations = torch.from_numpy(seq['orientations'][start:end])
        actions = torch.from_numpy(seq['actions'][start:end])
        prev_actions = torch.from_numpy(seq['actions'][start-1:end-1])
        velocities_gt = torch.from_numpy(seq['velocities'][start:end])
        initial_prev_vel = torch.from_numpy(seq['velocities'][start-1])

        # Compute ground truth acceleration from velocity derivative
        # accel[t] = (vel[t] - vel[t-1]) / dt
        vel_prev = np.concatenate([
            seq['velocities'][start-1:start],  # initial prev_vel
            seq['velocities'][start:end-1]     # vel[t-1] for rest
        ], axis=0)
        vel_curr = seq['velocities'][start:end]
        accel_gt = (vel_curr - vel_prev) / self.dt  # (L, 3)

        # Apply IMU augmentation (corruption)
        accel_aug = self.imu_aug(accel_gt)

        frame_indices = list(range(start, end))

        return {
            'seq_path': seq['path'],
            'start_idx': start,
            'length': length,
            'frame_indices': frame_indices,
            'orientations': orientations,
            'actions': actions,
            'prev_actions': prev_actions,
            'velocities_gt': velocities_gt,
            'initial_prev_vel': initial_prev_vel,
            'accel_gt': torch.from_numpy(accel_gt.astype(np.float32)),
            'accel_aug': torch.from_numpy(accel_aug),
        }


def collate_sequences(batch: List[dict]) -> Dict[str, any]:
    """
    Collate function for sequence dataset.

    Handles variable-length sequences by returning lists.
    """
    return {
        'seq_paths': [b['seq_path'] for b in batch],
        'start_indices': [b['start_idx'] for b in batch],
        'lengths': [b['length'] for b in batch],
        'frame_indices': [b['frame_indices'] for b in batch],
        'orientations': [b['orientations'] for b in batch],
        'actions': [b['actions'] for b in batch],
        'prev_actions': [b['prev_actions'] for b in batch],
        'velocities_gt': [b['velocities_gt'] for b in batch],
        'initial_prev_vels': [b['initial_prev_vel'] for b in batch],
        'accel_gt': [b['accel_gt'] for b in batch],
        'accel_aug': [b['accel_aug'] for b in batch],
    }


def create_dataloaders(
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
        seq_length: Length of sequence chunks (None = full sequence)
        stride: Stride between chunks
        batch_size: Batch size (number of sequences per batch)
        val_ratio: Fraction for validation
        imu_augmentation: Whether to apply IMU noise augmentation (training only)

    Returns:
        (train_loader, val_loader)
    """
    # Training dataset with IMU augmentation
    train_full_dataset = VelNetDataset(data_dir, seq_length, stride, imu_augmentation=imu_augmentation)

    # Validation dataset without IMU augmentation (clean data)
    val_full_dataset = VelNetDataset(data_dir, seq_length, stride, imu_augmentation=False)

    n_total = len(train_full_dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    # Split indices
    indices = list(range(n_total))
    torch.manual_seed(42)
    perm = torch.randperm(n_total).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_sequences,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequences,
        drop_last=False,
    )

    print(f"[DataLoaders] Train: {n_train} chunks, Val: {n_val} chunks")
    print(f"  Seq length: {seq_length}, Stride: {stride}, Batch size: {batch_size}")

    return train_loader, val_loader


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/vel_net/sequences')
    args = parser.parse_args()

    print("Testing VelNetDataset with IMU fusion...")

    try:
        dataset = VelNetDataset(args.data_dir, seq_length=64, stride=32, imu_augmentation=True)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  seq_path: {sample['seq_path']}")
        print(f"  length: {sample['length']}")
        print(f"  orientations: {sample['orientations'].shape}")
        print(f"  velocities_gt: {sample['velocities_gt'].shape}")
        print(f"  accel_gt: {sample['accel_gt'].shape}")
        print(f"  accel_aug: {sample['accel_aug'].shape}")

        # Show IMU augmentation effect
        accel_gt = sample['accel_gt'].numpy()
        accel_aug = sample['accel_aug'].numpy()
        print(f"\n  IMU Augmentation effect (first frame):")
        print(f"    accel_gt:  {accel_gt[0]}")
        print(f"    accel_aug: {accel_aug[0]}")
        print(f"    diff:      {accel_aug[0] - accel_gt[0]}")

        train_loader, val_loader = create_dataloaders(
            args.data_dir, seq_length=64, stride=32, batch_size=4
        )

        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"  seq_paths: {len(batch['seq_paths'])} sequences")
        print(f"  lengths: {batch['lengths']}")
        print(f"  accel_aug shapes: {[a.shape for a in batch['accel_aug']]}")

        print("\nDataset test passed!")

    except Exception as e:
        import traceback
        print(f"Dataset test failed: {e}")
        traceback.print_exc()
