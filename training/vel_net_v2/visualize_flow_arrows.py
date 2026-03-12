#!/usr/bin/env python3
"""
Visualize optical flow using arrow overlays on RGB images.

Draws sparse arrow grids on top of RGB frames showing both RAFT and GT flow
side by side. Useful for debugging and presentations where direction/magnitude
need to be immediately intuitive.

Usage:
    python training/vel_net_v2/visualize_flow_arrows.py \
        --data_dir /scr/irislab/ke/data/vel_net_v2_data_0309/gate_mid \
        --seq_idx 0 --max_frames 50 --step 12 --scale 3.0 \
        --output_dir eval_flow_arrows --save_video
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from training.vel_net_v2.evaluate_optical_flow import (
    compute_camera_pose,
    compute_gt_flow,
    compute_epe_metrics,
    FX, FY, CX, CY, IMG_H, IMG_W,
)


def draw_flow_arrows(
    rgb: np.ndarray,
    flow: np.ndarray,
    step: int = 12,
    scale: float = 3.0,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    mag_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw sparse flow arrows on an RGB image.

    Args:
        rgb: (H, W, 3) uint8 image
        flow: (2, H, W) optical flow [du, dv]
        step: Grid spacing in pixels
        scale: Arrow length multiplier
        color: BGR arrow color
        thickness: Arrow line thickness
        mag_threshold: Skip arrows with magnitude below this (pixels)

    Returns:
        Annotated RGB image copy (uint8)
    """
    vis = rgb.copy()

    H, W = flow.shape[1], flow.shape[2]
    for v in range(step // 2, H, step):
        for u in range(step // 2, W, step):
            du = flow[0, v, u]
            dv = flow[1, v, u]
            mag = np.sqrt(du * du + dv * dv)
            if mag < mag_threshold:
                continue
            u1 = int(round(u + du * scale))
            v1 = int(round(v + dv * scale))
            cv2.arrowedLine(vis, (u, v), (u1, v1), color,
                            thickness=thickness, tipLength=0.3,
                            line_type=cv2.LINE_AA)
    return vis


def save_arrow_comparison(
    rgb_t: np.ndarray,
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
    frame_idx: int,
    step: int = 12,
    scale: float = 3.0,
) -> np.ndarray:
    """
    Save a side-by-side comparison: [RGB + RAFT arrows | RGB + GT arrows].

    Args:
        rgb_t: (H, W, 3) uint8 RGB frame
        flow_pred: (2, H, W) RAFT predicted flow
        flow_gt: (2, H, W) ground-truth flow
        valid_mask: (H, W) bool valid pixels
        out_path: Output file path
        frame_idx: Frame index for annotation
        step: Arrow grid spacing
        scale: Arrow length multiplier

    Returns:
        The assembled comparison image (H_total, W_total, 3) uint8
    """
    H, W = rgb_t.shape[:2]

    # Darken RGB so arrows stand out
    rgb_dark = (rgb_t.astype(np.float32) * 0.6).astype(np.uint8)

    # Draw arrows
    raft_vis = draw_flow_arrows(rgb_dark, flow_pred, step=step, scale=scale,
                                color=(0, 255, 0), thickness=1)
    gt_vis = draw_flow_arrows(rgb_dark, flow_gt.astype(np.float32),
                              step=step, scale=scale,
                              color=(0, 255, 255), thickness=1)

    # Compute EPE stats
    epe_metrics = compute_epe_metrics(flow_pred, flow_gt, valid_mask)
    if epe_metrics is not None:
        mean_epe = epe_metrics['mean_epe']
        median_epe = epe_metrics['median_epe']
    else:
        mean_epe = 0.0
        median_epe = 0.0

    # Header row
    total_w = W * 2
    header_h = 22
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    labels = [
        f"RAFT Flow (green) - Frame {frame_idx}",
        f"GT Flow (cyan) - EPE: {mean_epe:.2f}px",
    ]
    for i, label in enumerate(labels):
        tw = cv2.getTextSize(label, font, 0.4, 1)[0][0]
        x = i * W + (W - tw) // 2
        cv2.putText(header, label, (x, 16), font, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Main row
    main_row = np.concatenate([raft_vis, gt_vis], axis=1)

    # Stats footer
    footer_h = 18
    footer = np.zeros((footer_h, total_w, 3), dtype=np.uint8)
    stats_text = (f"Mean EPE: {mean_epe:.2f}px | "
                  f"Median EPE: {median_epe:.2f}px | "
                  f"Step: {step}px | Scale: {scale:.1f}x")
    cv2.putText(footer, stats_text, (8, 13), font, 0.33,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Assemble
    full = np.concatenate([header, main_row, footer], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    return full


def process_sequence(
    seq_path: Path,
    max_frames: int = None,
    step: int = 12,
    scale: float = 3.0,
    output_dir: Path = None,
    save_video: bool = False,
):
    """Process a single sequence: load data, draw arrows, save outputs."""

    # Load telemetry
    telemetry = np.load(seq_path / "telemetry.npz")
    positions = telemetry['positions']       # (N, 3)
    orientations = telemetry['orientations']  # (N, 4) xyzw

    # Load RAFT flows
    flow_data = np.load(seq_path / "optical_flow.npz")
    flows_pred = flow_data['flows']  # (N-1, 2, H, W)

    n_pairs = len(flows_pred)
    if max_frames is not None:
        n_pairs = min(n_pairs, max_frames)

    # Precompute camera poses
    poses = []
    for i in range(n_pairs + 1):
        poses.append(compute_camera_pose(positions[i], orientations[i]))

    depth_dir = seq_path / "depth"
    rgb_dir = seq_path / "rgb"

    vis_dir = output_dir / seq_path.name
    vis_dir.mkdir(parents=True, exist_ok=True)

    frames_for_video = []

    for i in tqdm(range(n_pairs), desc=f"Processing {seq_path.name}"):
        depth_file = depth_dir / f"{i:06d}.npy"
        rgb_file = rgb_dir / f"{i:06d}.png"

        if not depth_file.exists() or not rgb_file.exists():
            continue

        depth_t = np.load(depth_file)
        rgb_t = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)

        # GT flow
        flow_gt, valid_mask = compute_gt_flow(depth_t, poses[i], poses[i + 1])

        # RAFT flow
        flow_pred = flows_pred[i]  # (2, H, W)

        # Save comparison
        out_path = vis_dir / f"frame_{i:04d}.png"
        frame_img = save_arrow_comparison(
            rgb_t, flow_pred, flow_gt, valid_mask,
            out_path, i, step=step, scale=scale,
        )

        if save_video:
            # RAFT-only frame for video
            rgb_dark = (rgb_t.astype(np.float32) * 0.6).astype(np.uint8)
            raft_frame = draw_flow_arrows(rgb_dark, flow_pred, step=step, scale=scale,
                                           color=(0, 255, 0), thickness=1)
            cv2.putText(raft_frame, f"Frame {i}", (5, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            frames_for_video.append(raft_frame)

    # Stitch into video
    if save_video and frames_for_video:
        video_path = output_dir / f"{seq_path.name}_arrows.avi"
        h, w = frames_for_video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
        for frame in frames_for_video:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved to {video_path}")

    print(f"Saved {n_pairs} frames to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optical flow with arrow overlays"
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing seq_* folders')
    parser.add_argument('--seq_idx', type=int, default=0,
                        help='Which sequence index to visualize')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames to process')
    parser.add_argument('--step', type=int, default=12,
                        help='Arrow grid spacing in pixels')
    parser.add_argument('--scale', type=float, default=3.0,
                        help='Arrow length multiplier')
    parser.add_argument('--output_dir', type=str, default='eval_flow_arrows',
                        help='Where to save output')
    parser.add_argument('--save_video', action='store_true',
                        help='Also save AVI video')

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    sequences = sorted(glob.glob(str(data_path / "seq_*")))

    if len(sequences) == 0:
        print(f"No sequences found in {args.data_dir}")
        return

    if args.seq_idx >= len(sequences):
        print(f"seq_idx {args.seq_idx} out of range (found {len(sequences)} sequences)")
        return

    seq_path = Path(sequences[args.seq_idx])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {seq_path.name} from {args.data_dir}")
    print(f"Arrow step: {args.step}px, scale: {args.scale}x")

    process_sequence(
        seq_path,
        max_frames=args.max_frames,
        step=args.step,
        scale=args.scale,
        output_dir=output_dir,
        save_video=args.save_video,
    )


if __name__ == '__main__':
    main()
