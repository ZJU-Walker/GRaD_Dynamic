"""
PyTorch Dataset for Velocity Network Training.

Loads raw images and telemetry, constructs 81-dim observations.
Uses teacher forcing: prev_vel = GT velocity from previous timestep.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, List, Tuple, Dict
import glob

from models.vel_net.vel_obs_utils import quaternion_to_rot6d, build_vel_observation
from models.vel_net.visual_encoder import DualEncoder


class VelNetDataset(Dataset):
    """
    Dataset for velocity network training.

    Loads sequences from disk and constructs 81-dim observations:
    - Rot6D (6): from quaternion
    - Action (4): current action
    - Prev Action (4): previous action
    - Prev Velocity (3): GT velocity from t-1 (teacher forcing)
    - RGB Features (32): encoded from image
    - Depth Features (32): encoded from image

    Args:
        data_dir: Path to sequences directory
        encoder: DualEncoder for RGB/depth encoding (optional, for on-the-fly encoding)
        precomputed_features: If True, load precomputed features instead of raw images
        transform: Optional image transform
        device: Device for encoding
    """

    def __init__(
        self,
        data_dir: str,
        encoder: Optional[DualEncoder] = None,
        precomputed_features: bool = False,
        device: str = 'cuda:0',
    ):
        self.data_dir = Path(data_dir)
        self.encoder = encoder
        self.precomputed_features = precomputed_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Find all sequences
        self.sequences = sorted(glob.glob(str(self.data_dir / "seq_*")))
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {data_dir}")

        # Load all data into memory (for speed during training)
        self.samples = []
        self._load_all_sequences()

        print(f"[VelNetDataset] Loaded {len(self.samples)} samples from {len(self.sequences)} sequences")

    def _load_all_sequences(self):
        """Load all sequences into memory."""
        for seq_path in self.sequences:
            seq_path = Path(seq_path)

            # Load telemetry
            telemetry = np.load(seq_path / "telemetry.npz")
            timestamps = telemetry['timestamps']
            positions = telemetry['positions']
            velocities = telemetry['velocities']  # Ground truth target
            orientations = telemetry['orientations']  # xyzw format
            actions = telemetry['actions']

            n_frames = len(timestamps)

            # Process each frame (skip first frame since we need prev_action and prev_vel)
            for t in range(1, n_frames):
                sample = {
                    'seq_path': str(seq_path),
                    'frame_idx': t,
                    'timestamp': timestamps[t],
                    'position': positions[t].astype(np.float32),
                    'velocity': velocities[t].astype(np.float32),  # GT target
                    'orientation': orientations[t].astype(np.float32),  # xyzw
                    'action': actions[t].astype(np.float32),
                    'prev_action': actions[t-1].astype(np.float32),
                    'prev_velocity': velocities[t-1].astype(np.float32),  # Teacher forcing
                }
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with:
                - observation: (81,) observation vector
                - velocity_gt: (3,) ground truth velocity
                - rgb_path: path to RGB image (for debugging)
                - depth_path: path to depth image (for debugging)
        """
        sample = self.samples[idx]
        seq_path = Path(sample['seq_path'])
        frame_idx = sample['frame_idx']

        # Load images
        rgb_path = seq_path / "rgb" / f"{frame_idx:06d}.png"
        depth_path = seq_path / "depth" / f"{frame_idx:06d}.npy"

        rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0  # (H, W, 3)
        depth = np.load(depth_path).astype(np.float32)  # (H, W)

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # Encode features (if encoder provided)
        if self.encoder is not None:
            with torch.no_grad():
                # Move to device for encoding
                rgb_batch = rgb_tensor.unsqueeze(0).to(self.device)
                depth_batch = depth_tensor.unsqueeze(0).to(self.device)

                rgb_feat, depth_feat = self.encoder(rgb_batch, depth_batch)
                rgb_feat = rgb_feat.squeeze(0).cpu()  # (32,)
                depth_feat = depth_feat.squeeze(0).cpu()  # (32,)
        else:
            # Placeholder zeros (encoder will be applied in collate or training loop)
            rgb_feat = torch.zeros(32)
            depth_feat = torch.zeros(32)

        # Build observation
        quat = torch.from_numpy(sample['orientation']).unsqueeze(0)  # (1, 4) xyzw
        rot6d = quaternion_to_rot6d(quat).squeeze(0)  # (6,)

        action = torch.from_numpy(sample['action'])  # (4,)
        prev_action = torch.from_numpy(sample['prev_action'])  # (4,)
        prev_vel = torch.from_numpy(sample['prev_velocity'])  # (3,)

        # Concatenate to build 81-dim observation
        observation = torch.cat([
            rot6d,       # 6
            action,      # 4
            prev_action, # 4
            prev_vel,    # 3
            rgb_feat,    # 32
            depth_feat,  # 32
        ], dim=0)  # (81,)

        velocity_gt = torch.from_numpy(sample['velocity'])  # (3,)

        return {
            'observation': observation,
            'velocity_gt': velocity_gt,
            'rgb': rgb_tensor,  # For on-the-fly encoding in training loop
            'depth': depth_tensor,
            'frame_idx': frame_idx,
        }


class VelNetDatasetWithEncoder(Dataset):
    """
    Dataset that applies encoder during data loading.

    This version keeps raw images and applies encoder in __getitem__.
    The encoder's FC layer is trainable, so gradients flow during training.

    For training: Use this dataset with encoder.train() mode.
    For validation: Use this dataset with encoder.eval() mode.
    """

    def __init__(
        self,
        data_dir: str,
        encoder: DualEncoder,
        device: str = 'cuda:0',
    ):
        self.data_dir = Path(data_dir)
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Find all sequences
        self.sequences = sorted(glob.glob(str(self.data_dir / "seq_*")))
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {data_dir}")

        # Build sample index
        self.samples = []
        self._index_sequences()

        print(f"[VelNetDatasetWithEncoder] Indexed {len(self.samples)} samples from {len(self.sequences)} sequences")

    def _index_sequences(self):
        """Index all sequences (don't load images yet)."""
        for seq_path in self.sequences:
            seq_path = Path(seq_path)

            # Load telemetry
            telemetry = np.load(seq_path / "telemetry.npz")
            n_frames = len(telemetry['timestamps'])

            # Index frames (skip first)
            for t in range(1, n_frames):
                self.samples.append({
                    'seq_path': str(seq_path),
                    'frame_idx': t,
                })

        # Cache telemetry for faster access
        self._telemetry_cache = {}

    def _get_telemetry(self, seq_path: str) -> dict:
        """Get cached telemetry data."""
        if seq_path not in self._telemetry_cache:
            self._telemetry_cache[seq_path] = dict(np.load(Path(seq_path) / "telemetry.npz"))
        return self._telemetry_cache[seq_path]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        seq_path = sample['seq_path']
        t = sample['frame_idx']

        # Get telemetry
        telemetry = self._get_telemetry(seq_path)

        # Load images
        rgb_path = Path(seq_path) / "rgb" / f"{t:06d}.png"
        depth_path = Path(seq_path) / "depth" / f"{t:06d}.npy"

        rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
        depth = np.load(depth_path).astype(np.float32)

        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)

        # Build non-visual observation components
        quat = torch.from_numpy(telemetry['orientations'][t].astype(np.float32)).unsqueeze(0)
        rot6d = quaternion_to_rot6d(quat).squeeze(0)  # (6,)

        action = torch.from_numpy(telemetry['actions'][t].astype(np.float32))
        prev_action = torch.from_numpy(telemetry['actions'][t-1].astype(np.float32))
        prev_vel = torch.from_numpy(telemetry['velocities'][t-1].astype(np.float32))

        velocity_gt = torch.from_numpy(telemetry['velocities'][t].astype(np.float32))

        return {
            'rot6d': rot6d,  # (6,)
            'action': action,  # (4,)
            'prev_action': prev_action,  # (4,)
            'prev_vel': prev_vel,  # (3,)
            'rgb': rgb_tensor,  # (3, H, W)
            'depth': depth_tensor,  # (1, H, W)
            'velocity_gt': velocity_gt,  # (3,)
        }


def collate_with_encoder(batch: List[dict], encoder: DualEncoder, device: str) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that applies encoder to images.

    This is called during training to batch samples and encode images.
    The encoder's FC layer receives gradients.

    Args:
        batch: List of sample dicts
        encoder: DualEncoder to apply
        device: Device for encoding

    Returns:
        Batched dict with observations and targets
    """
    B = len(batch)

    # Stack non-visual components
    rot6d = torch.stack([b['rot6d'] for b in batch])  # (B, 6)
    action = torch.stack([b['action'] for b in batch])  # (B, 4)
    prev_action = torch.stack([b['prev_action'] for b in batch])  # (B, 4)
    prev_vel = torch.stack([b['prev_vel'] for b in batch])  # (B, 3)
    velocity_gt = torch.stack([b['velocity_gt'] for b in batch])  # (B, 3)

    # Stack images and move to device
    rgb = torch.stack([b['rgb'] for b in batch]).to(device)  # (B, 3, H, W)
    depth = torch.stack([b['depth'] for b in batch]).to(device)  # (B, 1, H, W)

    # Encode images (FC layer is trainable, gradients flow)
    rgb_feat, depth_feat = encoder(rgb, depth)  # (B, 32), (B, 32)

    # Move other components to device
    rot6d = rot6d.to(device)
    action = action.to(device)
    prev_action = prev_action.to(device)
    prev_vel = prev_vel.to(device)
    velocity_gt = velocity_gt.to(device)

    # Build full observation
    observation = torch.cat([
        rot6d,       # 6
        action,      # 4
        prev_action, # 4
        prev_vel,    # 3
        rgb_feat,    # 32
        depth_feat,  # 32
    ], dim=1)  # (B, 81)

    return {
        'observation': observation,  # (B, 81)
        'velocity_gt': velocity_gt,  # (B, 3)
    }


def create_dataloaders(
    data_dir: str,
    encoder: DualEncoder,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    device: str = 'cuda:0',
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to sequences directory
        encoder: DualEncoder instance
        batch_size: Batch size
        val_ratio: Fraction for validation
        num_workers: DataLoader workers
        device: Device for encoding

    Returns:
        (train_loader, val_loader)
    """
    # Create dataset
    dataset = VelNetDatasetWithEncoder(data_dir, encoder, device)

    # Split into train/val
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create collate function with encoder
    def collate_fn(batch):
        return collate_with_encoder(batch, encoder, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # Images go through encoder, not directly to GPU
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    print(f"[DataLoaders] Train: {n_train} samples, Val: {n_val} samples")

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/vel_net/sequences')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print("Testing VelNetDataset...")

    # Create encoder
    encoder = DualEncoder(rgb_dim=32, depth_dim=32)
    encoder.to(args.device)

    # Create dataset
    try:
        dataset = VelNetDatasetWithEncoder(args.data_dir, encoder, args.device)
        print(f"Dataset size: {len(dataset)}")

        # Test single sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  rot6d: {sample['rot6d'].shape}")
        print(f"  action: {sample['action'].shape}")
        print(f"  rgb: {sample['rgb'].shape}")
        print(f"  velocity_gt: {sample['velocity_gt'].shape}")

        # Test dataloader
        train_loader, val_loader = create_dataloaders(
            args.data_dir, encoder,
            batch_size=8, device=args.device
        )

        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"  observation: {batch['observation'].shape}")
        print(f"  velocity_gt: {batch['velocity_gt'].shape}")

        print("\nDataset test passed!")

    except Exception as e:
        print(f"Dataset test failed (expected if no data collected yet): {e}")
