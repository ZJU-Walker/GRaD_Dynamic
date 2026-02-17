#!/usr/bin/env python3
"""
Pre-compute MobileNetV3 backbone features for faster training.

This script extracts 576-dim backbone features from all images and saves them.
During training, only the FC layer (576 → 32) needs to run, which is ~10x faster.

Usage:
    python training/vel_net/precompute_features.py \
        --data_dir data/vel_net/sequences_0106 \
        --device cuda:0
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.vel_net.visual_encoder import DualEncoder


def precompute_sequence_features(
    seq_path: Path,
    encoder: DualEncoder,
    device: str,
    batch_size: int = 32,
) -> dict:
    """
    Pre-compute backbone features for a single sequence.

    Args:
        seq_path: Path to sequence folder
        encoder: DualEncoder with backbone
        device: PyTorch device
        batch_size: Batch size for processing

    Returns:
        Dict with rgb_features and depth_features arrays
    """
    rgb_dir = seq_path / "rgb"
    depth_dir = seq_path / "depth"

    # Get all frame indices
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.png")))
    n_frames = len(rgb_files)

    if n_frames == 0:
        return None

    # Pre-allocate arrays
    rgb_features = np.zeros((n_frames, 576), dtype=np.float32)
    depth_features = np.zeros((n_frames, 576), dtype=np.float32)

    # Process in batches
    encoder.eval()
    with torch.no_grad():
        for start_idx in range(0, n_frames, batch_size):
            end_idx = min(start_idx + batch_size, n_frames)
            batch_rgb = []
            batch_depth = []

            for idx in range(start_idx, end_idx):
                frame_idx = idx  # Assuming 0-indexed filenames
                rgb_path = rgb_dir / f"{frame_idx:06d}.png"
                depth_path = depth_dir / f"{frame_idx:06d}.npy"

                # Load RGB
                rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
                batch_rgb.append(rgb_tensor)

                # Load Depth
                depth = np.load(depth_path).astype(np.float32).squeeze()  # (H, W)
                depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
                batch_depth.append(depth_tensor)

            # Stack and move to device
            rgb_batch = torch.stack(batch_rgb).to(device)  # (B, 3, H, W)
            depth_batch = torch.stack(batch_depth).to(device)  # (B, 1, H, W)

            # Extract backbone features
            rgb_feat, depth_feat = encoder.extract_backbone_features(rgb_batch, depth_batch)

            # Store
            rgb_features[start_idx:end_idx] = rgb_feat.cpu().numpy()
            depth_features[start_idx:end_idx] = depth_feat.cpu().numpy()

    return {
        'rgb_features': rgb_features,
        'depth_features': depth_features,
    }


def precompute_all_features(
    data_dir: str,
    device: str = 'cuda:0',
    batch_size: int = 32,
):
    """
    Pre-compute backbone features for all sequences.

    Args:
        data_dir: Path to sequences directory
        device: PyTorch device
        batch_size: Batch size for processing
    """
    data_path = Path(data_dir)
    sequences = sorted(glob.glob(str(data_path / "seq_*")))

    if len(sequences) == 0:
        print(f"No sequences found in {data_dir}")
        return

    print(f"Found {len(sequences)} sequences in {data_dir}")
    print(f"Device: {device}, Batch size: {batch_size}")

    # Create encoder
    encoder = DualEncoder(rgb_dim=32, depth_dim=32).to(device)
    encoder.eval()

    # Process each sequence
    for seq_path in tqdm(sequences, desc="Pre-computing features"):
        seq_path = Path(seq_path)
        output_file = seq_path / "backbone_features.npz"

        # Skip if already computed
        if output_file.exists():
            continue

        # Compute features
        features = precompute_sequence_features(seq_path, encoder, device, batch_size)

        if features is not None:
            # Save features
            np.savez_compressed(
                output_file,
                rgb_features=features['rgb_features'],
                depth_features=features['depth_features'],
            )

    print(f"\nDone! Backbone features saved to each sequence folder as 'backbone_features.npz'")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute MobileNetV3 backbone features')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to sequences directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='PyTorch device')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing features')

    args = parser.parse_args()

    # Remove existing features if force
    if args.force:
        data_path = Path(args.data_dir)
        for npz_file in data_path.glob("seq_*/backbone_features.npz"):
            npz_file.unlink()
            print(f"Removed {npz_file}")

    precompute_all_features(
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
