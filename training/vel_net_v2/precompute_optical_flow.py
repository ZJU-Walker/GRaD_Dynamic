#!/usr/bin/env python3
"""
Pre-compute RAFT-Small dense optical flow for each sequence.

Analogous to precompute_features.py but for optical flow.
RAFT-Small is frozen (pretrained); the FlowEncoder CNN is trainable during training.

Usage:
    python training/vel_net_v2/precompute_optical_flow.py \
        --data_dir data/vel_net/sequences \
        --device cuda:0

Output per sequence: optical_flow.npz with key 'flows' shape (N-1, 2, H, W)
  - flows[i] = optical flow from frame i to frame i+1
  - First frame has no predecessor, so flows has N-1 entries (aligned with frames 1..N-1)
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

import torchvision.transforms.functional as TF


def load_raft_small(device: str = 'cuda:0'):
    """Load pretrained RAFT-Small model from torchvision."""
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device)
    model.eval()

    # Get the transforms from weights
    transforms = weights.transforms()

    return model, transforms


def precompute_sequence_flow(
    seq_path: Path,
    model,
    transforms,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Pre-compute optical flow for a single sequence.

    Args:
        seq_path: Path to sequence folder
        model: RAFT-Small model
        transforms: Preprocessing transforms for RAFT
        device: PyTorch device
        batch_size: Batch size for RAFT inference

    Returns:
        flows: (N-1, 2, H, W) optical flow arrays, or None if no frames
    """
    rgb_dir = seq_path / "rgb"
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.png")))
    n_frames = len(rgb_files)

    if n_frames < 2:
        return None

    # Load all RGB frames
    frames = []
    for rgb_file in rgb_files:
        img = Image.open(rgb_file).convert('RGB')
        img_tensor = TF.to_tensor(img)  # (3, H, W), [0, 1]
        frames.append(img_tensor)

    # Compute flow between consecutive frames
    all_flows = []

    with torch.no_grad():
        for start_idx in range(0, n_frames - 1, batch_size):
            end_idx = min(start_idx + batch_size, n_frames - 1)

            batch_img1 = []
            batch_img2 = []
            for i in range(start_idx, end_idx):
                img1 = frames[i]
                img2 = frames[i + 1]
                # Apply RAFT transforms (expects batch dim)
                img1_t, img2_t = transforms(img1.unsqueeze(0), img2.unsqueeze(0))
                batch_img1.append(img1_t.squeeze(0))
                batch_img2.append(img2_t.squeeze(0))

            batch_img1 = torch.stack(batch_img1).to(device)
            batch_img2 = torch.stack(batch_img2).to(device)

            # RAFT returns list of flow predictions (multi-scale); take the last (finest)
            flow_predictions = model(batch_img1, batch_img2)
            flow = flow_predictions[-1]  # (B, 2, H, W)

            all_flows.append(flow.cpu().numpy())

    flows = np.concatenate(all_flows, axis=0)  # (N-1, 2, H, W)
    return flows.astype(np.float32)


def precompute_all_flows(
    data_dir: str,
    device: str = 'cuda:0',
    batch_size: int = 8,
    force: bool = False,
):
    """
    Pre-compute optical flow for all sequences.

    Args:
        data_dir: Path to sequences directory
        device: PyTorch device
        batch_size: Batch size for RAFT inference
        force: Overwrite existing files
    """
    data_path = Path(data_dir)
    sequences = sorted(glob.glob(str(data_path / "seq_*")))

    if len(sequences) == 0:
        print(f"No sequences found in {data_dir}")
        return

    print(f"Found {len(sequences)} sequences in {data_dir}")
    print(f"Device: {device}, Batch size: {batch_size}")

    # Load RAFT-Small
    print("Loading RAFT-Small...")
    model, transforms = load_raft_small(device)
    print("RAFT-Small loaded.")

    # Process each sequence
    for seq_dir in tqdm(sequences, desc="Pre-computing optical flow"):
        seq_path = Path(seq_dir)
        output_file = seq_path / "optical_flow.npz"

        # Skip if already computed (unless force)
        if output_file.exists() and not force:
            continue

        flows = precompute_sequence_flow(seq_path, model, transforms, device, batch_size)

        if flows is not None:
            np.savez_compressed(output_file, flows=flows)

    print(f"\nDone! Optical flow saved to each sequence folder as 'optical_flow.npz'")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute RAFT-Small optical flow')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to sequences directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='PyTorch device')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for RAFT inference')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing flow files')

    args = parser.parse_args()

    # Remove existing files if force
    if args.force:
        data_path = Path(args.data_dir)
        for npz_file in data_path.glob("seq_*/optical_flow.npz"):
            npz_file.unlink()
            print(f"Removed {npz_file}")

    precompute_all_flows(
        data_dir=args.data_dir,
        device=args.device,
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == '__main__':
    main()
