"""
PyTorch Dataset for Velocity Network Training.

Sequence-based dataset for scheduled sampling training.
Returns full sequences (or fixed-length chunks) instead of single frames.
This enables step-by-step training where predicted velocities can be
fed back as prev_vel (scheduled sampling).
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


class VelNetDataset(Dataset):
    """
    Sequence-based dataset for scheduled sampling training.

    Returns full sequences (or fixed-length chunks) instead of single frames.
    This enables step-by-step training where predicted velocities can be
    fed back as prev_vel (scheduled sampling).

    Args:
        data_dir: Path to sequences directory
        seq_length: Length of sequence chunks (None = full sequence)
        stride: Stride between sequence chunks (for overlapping)
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length

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
        Get a sequence chunk.

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
    }


def create_dataloaders(
    data_dir: str,
    seq_length: Optional[int] = 64,
    stride: Optional[int] = 32,
    batch_size: int = 8,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Path to sequences directory
        seq_length: Length of sequence chunks (None = full sequence)
        stride: Stride between chunks
        batch_size: Batch size (number of sequences per batch)
        val_ratio: Fraction for validation

    Returns:
        (train_loader, val_loader)
    """
    dataset = VelNetDataset(data_dir, seq_length, stride)

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

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

    print("Testing VelNetDataset...")

    try:
        dataset = VelNetDataset(args.data_dir, seq_length=64, stride=32)
        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"  seq_path: {sample['seq_path']}")
        print(f"  length: {sample['length']}")
        print(f"  orientations: {sample['orientations'].shape}")
        print(f"  velocities_gt: {sample['velocities_gt'].shape}")

        train_loader, val_loader = create_dataloaders(
            args.data_dir, seq_length=64, stride=32, batch_size=4
        )

        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"  seq_paths: {len(batch['seq_paths'])} sequences")
        print(f"  lengths: {batch['lengths']}")

        print("\nDataset test passed!")

    except Exception as e:
        print(f"Dataset test failed: {e}")
