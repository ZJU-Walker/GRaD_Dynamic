#!/usr/bin/env python3
"""
Evaluate optical flow quality by comparing RAFT-Small predictions against
pose-based ground-truth flow computed via depth + camera poses + intrinsics.

For each frame pair (t, t+1):
  1. Unproject frame t pixels to 3D using depth + intrinsics
  2. Transform 3D points from camera t to camera t+1 using poses
  3. Reproject to frame t+1 pixel coordinates
  4. GT flow = projected coords - original coords
  5. Compare with RAFT flow using EPE and photometric metrics

Usage:
    python training/vel_net_v2/evaluate_optical_flow.py \
        --data_dir /scr/irislab/ke/data/vel_net_v2_data_0309/gate_mid \
        --device cuda:0 --num_sequences 5 --save_vis --output_dir eval_flow_output
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

# ---------------------------------------------------------------------------
# Camera intrinsics (gs_local.py:52-53 scaled by 0.4 for 256x144)
# ---------------------------------------------------------------------------

FX = 185.182
FY = 185.201
CX = 129.230
CY = 72.474
IMG_H = 144
IMG_W = 256

# ---------------------------------------------------------------------------
# Fixed transform matrices (from utils/gs_local.py:238-257)
# ---------------------------------------------------------------------------

# Camera → ROS body (axis swap)
T_c2r = np.array([
    [0.0,  0.0, -1.0, 0.0],
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)

# ROS body → drone body (~8° tilt + offset)
T_r2d = np.array([
    [ 0.990, 0.000, 0.140, 0.152],
    [ 0.000, 1.000, 0.000, -0.031],
    [-0.140, 0.000, 0.990, -0.012],
    [ 0.000, 0.000, 0.000,  1.000],
], dtype=np.float64)


def _quaternion_to_rotation_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix (numpy, float64)."""
    q = quat / (np.linalg.norm(quat) + 1e-12)
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_camera_pose(position: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
    """
    Compute 4x4 camera-to-world transform from telemetry pose.

    Args:
        position: (3,) world-frame position [x, y, z]
        orientation_xyzw: (4,) quaternion in xyzw format

    Returns:
        T_cam_to_world: (4, 4) float64 transform
    """
    R = _quaternion_to_rotation_matrix_np(orientation_xyzw.astype(np.float64))

    # Build T_d2f (drone body → world) from telemetry
    T_d2f = np.eye(4, dtype=np.float64)
    T_d2f[:3, :3] = R
    T_d2f[:3, 3] = position.astype(np.float64)

    # Camera → world: T_d2f @ T_r2d @ T_c2r
    T_cam_to_world = T_d2f @ T_r2d @ T_c2r
    return T_cam_to_world


def compute_gt_flow(
    depth_t: np.ndarray,
    T_cam_world_t: np.ndarray,
    T_cam_world_t1: np.ndarray,
    fx: float = FX, fy: float = FY,
    cx: float = CX, cy: float = CY,
) -> tuple:
    """
    Compute ground-truth optical flow via depth unprojection and reprojection.

    Args:
        depth_t: (H, W) or (H, W, 1) z-buffer depth at frame t
        T_cam_world_t: (4, 4) camera-to-world transform at t
        T_cam_world_t1: (4, 4) camera-to-world transform at t+1
        fx, fy, cx, cy: Camera intrinsics

    Returns:
        flow_gt: (2, H, W) float64 ground-truth flow [dx, dy]
        valid_mask: (H, W) bool — valid pixels for evaluation
    """
    if depth_t.ndim == 3:
        depth_t = depth_t[:, :, 0]

    H, W = depth_t.shape
    depth = depth_t.astype(np.float64)

    # Pixel grid
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (H, W) each

    # Unproject to 3D in camera t frame
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    # Stack into (H*W, 4) homogeneous points
    ones = np.ones_like(Z)
    pts_cam_t = np.stack([X, Y, Z, ones], axis=-1).reshape(-1, 4)  # (H*W, 4)

    # Transform: camera t → world → camera t+1
    T_world_to_cam_t1 = np.linalg.inv(T_cam_world_t1)
    T_t_to_t1 = T_world_to_cam_t1 @ T_cam_world_t  # camera t → camera t+1

    # Convert from OpenGL camera convention to CV convention
    # T_cam_to_world uses OpenGL axes (x=right, y=up, z=backward) via T_c2r
    # But pixel unprojection/projection uses CV axes (x=right, y=down, z=forward)
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    T_t_to_t1 = F @ T_t_to_t1 @ F

    pts_cam_t1 = (T_t_to_t1 @ pts_cam_t.T).T  # (H*W, 4)

    # Project to pixel coords in camera t+1
    X1 = pts_cam_t1[:, 0]
    Y1 = pts_cam_t1[:, 1]
    Z1 = pts_cam_t1[:, 2]

    # Avoid division by zero
    Z1_safe = np.where(np.abs(Z1) < 1e-6, 1e-6, Z1)
    u1 = fx * X1 / Z1_safe + cx
    v1 = fy * Y1 / Z1_safe + cy

    # Reshape back to (H, W)
    u1 = u1.reshape(H, W)
    v1 = v1.reshape(H, W)
    Z1 = Z1.reshape(H, W)

    # GT flow
    flow_u = u1 - uu
    flow_v = v1 - vv
    flow_gt = np.stack([flow_u, flow_v], axis=0)  # (2, H, W)

    # Valid mask: positive depth, positive z in camera t+1, projected in bounds
    valid = (depth > 0) & (Z1 > 0.01)
    valid &= (u1 >= 0) & (u1 < W) & (v1 >= 0) & (v1 < H)

    return flow_gt, valid


def compute_epe_metrics(
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """
    Compute end-point error metrics between predicted and GT flow.

    Args:
        flow_pred: (2, H, W) predicted flow
        flow_gt: (2, H, W) ground-truth flow
        valid_mask: (H, W) bool

    Returns:
        Dict with mean_epe, median_epe, outlier_rate, and per-bin metrics.
    """
    if valid_mask.sum() == 0:
        return None

    pred = flow_pred[:, valid_mask].astype(np.float64)  # (2, N)
    gt = flow_gt[:, valid_mask].astype(np.float64)       # (2, N)

    epe = np.sqrt(((pred - gt) ** 2).sum(axis=0))  # (N,)
    gt_mag = np.sqrt((gt ** 2).sum(axis=0))          # (N,)

    # Outlier: EPE > 3px AND EPE > 5% of GT magnitude
    outliers = (epe > 3.0) & (epe > 0.05 * gt_mag)

    metrics = {
        'mean_epe': float(np.mean(epe)),
        'median_epe': float(np.median(epe)),
        'outlier_rate': float(np.mean(outliers)),
        'n_valid': int(valid_mask.sum()),
    }

    # Per motion-magnitude bin
    slow = gt_mag < 2.0
    medium = (gt_mag >= 2.0) & (gt_mag < 10.0)
    fast = gt_mag >= 10.0

    for name, mask in [('slow', slow), ('medium', medium), ('fast', fast)]:
        if mask.sum() > 0:
            metrics[f'{name}_epe'] = float(np.mean(epe[mask]))
            metrics[f'{name}_frac'] = float(np.mean(mask))
        else:
            metrics[f'{name}_epe'] = float('nan')
            metrics[f'{name}_frac'] = 0.0

    return metrics


def compute_photometric_error(
    rgb_t: np.ndarray,
    rgb_t1: np.ndarray,
    flow: np.ndarray,
) -> dict:
    """
    Backward-warp rgb_t1 to frame t using flow, compute L1 error.

    Args:
        rgb_t: (H, W, 3) uint8 frame t
        rgb_t1: (H, W, 3) uint8 frame t+1
        flow: (2, H, W) optical flow from t to t+1

    Returns:
        Dict with mean_l1 photometric error.
    """
    H, W = flow.shape[1], flow.shape[2]

    # Build remap coordinates: for each pixel (u,v) in frame t,
    # the corresponding pixel in frame t+1 is (u + flow_u, v + flow_v)
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    map_x = (uu + flow[0]).astype(np.float32)
    map_y = (vv + flow[1]).astype(np.float32)

    # Warp rgb_t1 to frame t
    warped = cv2.remap(rgb_t1, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # L1 error (normalized to [0, 1])
    diff = np.abs(rgb_t.astype(np.float32) - warped.astype(np.float32)) / 255.0

    # Mask out border pixels where warp goes out of bounds
    in_bounds = (map_x >= 0) & (map_x < W) & (map_y >= 0) & (map_y < H)

    if in_bounds.sum() > 0:
        mean_l1 = float(diff[in_bounds].mean())
    else:
        mean_l1 = float('nan')

    return {'mean_l1': mean_l1}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray, max_mag: float = None, gamma: float = 0.7,
                mag_threshold: float = 0.0) -> np.ndarray:
    """Convert (2, H, W) flow to (H, W, 3) uint8 RGB using HSV encoding.

    Args:
        gamma: Power curve for magnitude. Higher = less amplification of small flows.
               0.4 is standard but exaggerates sub-pixel flows; 0.7 is more honest.
        mag_threshold: Flow magnitudes below this (in px) are shown as black.
    """
    fx, fy = flow[0], flow[1]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)

    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255

    if max_mag is None or max_mag <= 0:
        max_mag = magnitude.max()
    if max_mag > 0:
        normed = np.clip(magnitude / max_mag, 0, 1)
        hsv[..., 2] = (np.power(normed, gamma) * 255).astype(np.uint8)

    # Suppress near-zero flows so direction noise doesn't dominate
    if mag_threshold > 0:
        hsv[magnitude < mag_threshold] = 0

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _make_color_wheel(size: int = 80) -> np.ndarray:
    """Generate a color wheel legend showing flow direction → color mapping."""
    cy, cx = size // 2, size // 2
    radius = size // 2 - 14

    ys, xs = np.mgrid[:size, :size]
    dy = ys - cy
    dx = xs - cx
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)

    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(dist / radius, 0, 1) * 255).astype(np.uint8)
    hsv[dist > radius, :] = 0

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Axis labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    s, t, c = 0.35, 1, (255, 255, 255)
    cv2.putText(rgb, "R", (size - 12, cy + 4), font, s, c, t, cv2.LINE_AA)
    cv2.putText(rgb, "L", (2, cy + 4), font, s, c, t, cv2.LINE_AA)
    cv2.putText(rgb, "D", (cx - 3, size - 3), font, s, c, t, cv2.LINE_AA)
    cv2.putText(rgb, "U", (cx - 3, 11), font, s, c, t, cv2.LINE_AA)

    return rgb


def _make_epe_colorbar(height: int, width: int = 20, vmin: float = 0,
                       vmax: float = 10) -> np.ndarray:
    """Generate a vertical JET colorbar for EPE values."""
    bar = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    bar = np.tile(bar, (1, width))
    bar_colored = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    bar_rgb = cv2.cvtColor(bar_colored, cv2.COLOR_BGR2RGB)

    # Add to a wider canvas with tick labels
    canvas_w = width + 35
    canvas = np.zeros((height, canvas_w, 3), dtype=np.uint8)
    canvas[:, :width] = bar_rgb

    font = cv2.FONT_HERSHEY_SIMPLEX
    n_ticks = 5
    for i in range(n_ticks + 1):
        y = int(i / n_ticks * (height - 1))
        val = vmin + (vmax - vmin) * i / n_ticks
        cv2.putText(canvas, f"{val:.0f}", (width + 2, y + 4), font, 0.3,
                    (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def save_comparison_image(
    rgb_t: np.ndarray,
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    valid_mask: np.ndarray,
    epe_map: np.ndarray,
    out_path: Path,
    frame_idx: int,
):
    """
    Save a comparison image with legends:
      Row 1 (header): column labels
      Row 2 (main):   [RGB | RAFT flow | GT flow | EPE heatmap]
      Row 3 (legend):  color wheel + EPE colorbar + stats text
    """
    H, W = rgb_t.shape[:2]

    # Shared magnitude scale
    pred_mag_map = np.sqrt(flow_pred[0]**2 + flow_pred[1]**2)
    gt_mag_map = np.sqrt(flow_gt[0]**2 + flow_gt[1]**2)
    max_mag = max(float(pred_mag_map.max()), float(gt_mag_map.max()), 1.0)

    mag_thresh = 0.5
    pred_vis = flow_to_rgb(flow_pred, max_mag=max_mag, gamma=0.7,
                           mag_threshold=mag_thresh)
    gt_vis = flow_to_rgb(flow_gt, max_mag=max_mag, gamma=0.7,
                         mag_threshold=mag_thresh)

    # EPE heatmap (masked)
    epe_vis = np.zeros((H, W, 3), dtype=np.uint8)
    if valid_mask.any():
        epe_clipped = np.clip(epe_map, 0, 10)
        normed = (epe_clipped / 10.0 * 255).astype(np.uint8)
        epe_colored = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        epe_colored = cv2.cvtColor(epe_colored, cv2.COLOR_BGR2RGB)
        epe_vis[valid_mask] = epe_colored[valid_mask]

    # --- Header row ---
    total_w = W * 4
    header_h = 22
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_labels = ["RGB (frame t)", "RAFT Flow (t->t+1)", "GT Flow (pose-based)",
                  "EPE Error (px)"]
    for i, label in enumerate(col_labels):
        tw = cv2.getTextSize(label, font, 0.4, 1)[0][0]
        x = i * W + (W - tw) // 2
        cv2.putText(header, label, (x, 16), font, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # --- Main row ---
    main_row = np.concatenate([rgb_t, pred_vis, gt_vis, epe_vis], axis=1)

    # --- Legend row ---
    legend_h = 100
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8)

    # Color wheel (for flow columns)
    wheel = _make_color_wheel(size=80)
    wh, ww = wheel.shape[:2]
    y0 = (legend_h - wh) // 2
    legend[y0:y0+wh, 10:10+ww] = wheel

    # Color wheel label
    cv2.putText(legend, "Flow Direction", (10, y0 - 4), font, 0.35,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(legend, f"Brightness = magnitude", (10, y0 + wh + 12), font,
                0.3, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(legend, f"(max: {max_mag:.1f}px)", (10, y0 + wh + 24), font,
                0.3, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(legend, f"Black = <{mag_thresh}px", (10, y0 + wh + 36), font,
                0.3, (160, 160, 160), 1, cv2.LINE_AA)

    # EPE colorbar (for EPE column)
    epe_bar_x = 3 * W + 10
    colorbar = _make_epe_colorbar(height=80, width=14, vmin=0, vmax=10)
    cb_h, cb_w = colorbar.shape[:2]
    y0_cb = (legend_h - cb_h) // 2
    legend[y0_cb:y0_cb+cb_h, epe_bar_x:epe_bar_x+cb_w] = colorbar
    cv2.putText(legend, "EPE (px)", (epe_bar_x, y0_cb - 4), font, 0.35,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(legend, "Black = invalid", (epe_bar_x, y0_cb + cb_h + 12),
                font, 0.3, (160, 160, 160), 1, cv2.LINE_AA)

    # Stats text in the middle of legend
    mean_epe = float(epe_map[valid_mask].mean()) if valid_mask.any() else 0
    median_epe = float(np.median(epe_map[valid_mask])) if valid_mask.any() else 0
    pct_valid = float(valid_mask.mean()) * 100

    stats_x = W + ww + 40
    stats = [
        f"Frame {frame_idx} -> {frame_idx+1}",
        f"Mean EPE:   {mean_epe:.2f} px",
        f"Median EPE: {median_epe:.2f} px",
        f"Max flow:   {max_mag:.1f} px",
        f"Valid:      {pct_valid:.0f}% pixels",
    ]
    for j, line in enumerate(stats):
        cv2.putText(legend, line, (stats_x, 16 + j * 16), font, 0.35,
                    (220, 220, 220), 1, cv2.LINE_AA)

    # Assemble
    full = np.concatenate([header, main_row, legend], axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Sequence and dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    seq_path: Path,
    max_frames: int = None,
    save_vis: bool = False,
    output_dir: Path = None,
) -> dict:
    """
    Evaluate optical flow for a single sequence.

    Args:
        seq_path: Path to sequence directory
        max_frames: Max frame pairs to evaluate (None = all)
        save_vis: Whether to save comparison images
        output_dir: Directory for visualizations

    Returns:
        Dict with aggregated metrics for this sequence.
    """
    # Load telemetry
    telemetry = np.load(seq_path / "telemetry.npz")
    positions = telemetry['positions']      # (N, 3)
    orientations = telemetry['orientations']  # (N, 4) xyzw

    # Load RAFT flows
    flow_data = np.load(seq_path / "optical_flow.npz")
    flows_pred = flow_data['flows']  # (N-1, 2, H, W)

    n_pairs = len(flows_pred)
    if max_frames is not None:
        n_pairs = min(n_pairs, max_frames)

    # Precompute all camera poses
    poses = []
    for i in range(n_pairs + 1):
        poses.append(compute_camera_pose(positions[i], orientations[i]))

    depth_dir = seq_path / "depth"
    rgb_dir = seq_path / "rgb"

    all_metrics = []

    for i in range(n_pairs):
        # Load depth for frame i (one at a time to save memory)
        depth_file = depth_dir / f"{i:06d}.npy"
        if not depth_file.exists():
            continue
        depth_t = np.load(depth_file)  # (H, W) or (H, W, 1)

        # Compute GT flow
        flow_gt, valid_mask = compute_gt_flow(
            depth_t, poses[i], poses[i + 1],
        )

        # Compute EPE metrics
        flow_pred = flows_pred[i]  # (2, H, W)
        epe_metrics = compute_epe_metrics(flow_pred, flow_gt, valid_mask)
        if epe_metrics is None:
            continue

        # Compute photometric error for RAFT flow
        rgb_t_file = rgb_dir / f"{i:06d}.png"
        rgb_t1_file = rgb_dir / f"{i+1:06d}.png"
        if rgb_t_file.exists() and rgb_t1_file.exists():
            rgb_t = cv2.cvtColor(cv2.imread(str(rgb_t_file)), cv2.COLOR_BGR2RGB)
            rgb_t1 = cv2.cvtColor(cv2.imread(str(rgb_t1_file)), cv2.COLOR_BGR2RGB)

            photo_pred = compute_photometric_error(rgb_t, rgb_t1, flow_pred)
            photo_gt = compute_photometric_error(rgb_t, rgb_t1, flow_gt.astype(np.float32))
            epe_metrics['photo_pred_l1'] = photo_pred['mean_l1']
            epe_metrics['photo_gt_l1'] = photo_gt['mean_l1']

            # Save visualization for sampled frames
            if save_vis and output_dir is not None and i % max(1, n_pairs // 5) == 0:
                epe_map = np.sqrt(((flow_pred.astype(np.float64) - flow_gt) ** 2).sum(axis=0))
                vis_dir = output_dir / seq_path.name
                vis_dir.mkdir(parents=True, exist_ok=True)
                save_comparison_image(
                    rgb_t, flow_pred, flow_gt.astype(np.float32),
                    valid_mask, epe_map, vis_dir / f"frame_{i:04d}.png", i,
                )

        all_metrics.append(epe_metrics)

    if len(all_metrics) == 0:
        return None

    # Aggregate per-sequence
    result = {
        'seq': seq_path.name,
        'n_pairs': len(all_metrics),
        'mean_epe': float(np.mean([m['mean_epe'] for m in all_metrics])),
        'median_epe': float(np.mean([m['median_epe'] for m in all_metrics])),
        'outlier_rate': float(np.mean([m['outlier_rate'] for m in all_metrics])),
    }

    for bin_name in ['slow', 'medium', 'fast']:
        valid_epes = [m[f'{bin_name}_epe'] for m in all_metrics
                      if not np.isnan(m[f'{bin_name}_epe'])]
        valid_fracs = [m[f'{bin_name}_frac'] for m in all_metrics]
        result[f'{bin_name}_epe'] = float(np.mean(valid_epes)) if valid_epes else float('nan')
        result[f'{bin_name}_frac'] = float(np.mean(valid_fracs))

    # Photometric
    pred_l1s = [m['photo_pred_l1'] for m in all_metrics if 'photo_pred_l1' in m]
    gt_l1s = [m['photo_gt_l1'] for m in all_metrics if 'photo_gt_l1' in m]
    result['photo_pred_l1'] = float(np.mean(pred_l1s)) if pred_l1s else float('nan')
    result['photo_gt_l1'] = float(np.mean(gt_l1s)) if gt_l1s else float('nan')

    return result


def evaluate_all(
    data_dir: str,
    num_sequences: int = None,
    max_frames: int = None,
    save_vis: bool = False,
    output_dir: str = None,
):
    """
    Evaluate optical flow across all sequences in a data directory.
    """
    data_path = Path(data_dir)
    sequences = sorted(glob.glob(str(data_path / "seq_*")))

    if len(sequences) == 0:
        print(f"No sequences found in {data_dir}")
        return

    if num_sequences is not None:
        sequences = sequences[:num_sequences]

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(sequences)} sequences from {data_dir}")
    print(f"Max frames per sequence: {max_frames or 'all'}")
    print()

    all_results = []
    for seq_dir in tqdm(sequences, desc="Evaluating sequences"):
        result = evaluate_sequence(
            Path(seq_dir),
            max_frames=max_frames,
            save_vis=save_vis,
            output_dir=out_path,
        )
        if result is not None:
            all_results.append(result)

    if len(all_results) == 0:
        print("No valid results.")
        return

    # Aggregate across all sequences
    total_pairs = sum(r['n_pairs'] for r in all_results)
    mean_epe = np.average(
        [r['mean_epe'] for r in all_results],
        weights=[r['n_pairs'] for r in all_results],
    )
    median_epe = np.average(
        [r['median_epe'] for r in all_results],
        weights=[r['n_pairs'] for r in all_results],
    )
    outlier_rate = np.average(
        [r['outlier_rate'] for r in all_results],
        weights=[r['n_pairs'] for r in all_results],
    )

    print()
    print("=" * 50)
    print("Optical Flow Evaluation Summary")
    print("=" * 50)
    print(f"Sequences: {len(all_results)} | Frame pairs: {total_pairs}")
    print()
    print("EPE (RAFT vs Pose-GT):")
    print(f"  Mean:    {mean_epe:.2f} px | Median: {median_epe:.2f} px | Outlier: {outlier_rate*100:.1f}%")

    for bin_name, label in [('slow', '<2px'), ('medium', '2-10px'), ('fast', '>10px')]:
        epes = [r[f'{bin_name}_epe'] for r in all_results
                if not np.isnan(r[f'{bin_name}_epe'])]
        fracs = [r[f'{bin_name}_frac'] for r in all_results]
        if epes:
            print(f"  {label:12s} {np.mean(epes):.2f} px ({np.mean(fracs)*100:.0f}%)")

    # Photometric
    pred_l1s = [r['photo_pred_l1'] for r in all_results
                if not np.isnan(r.get('photo_pred_l1', float('nan')))]
    gt_l1s = [r['photo_gt_l1'] for r in all_results
              if not np.isnan(r.get('photo_gt_l1', float('nan')))]
    if pred_l1s:
        print()
        print("Photometric:")
        print(f"  RAFT warp L1: {np.mean(pred_l1s):.4f}")
        print(f"  GT warp L1:   {np.mean(gt_l1s):.4f}")

    print()

    # Per-sequence breakdown
    print("Per-sequence breakdown:")
    print(f"  {'Sequence':<12s} {'Pairs':>6s} {'Mean EPE':>10s} {'Outlier':>8s} {'Photo L1':>10s}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
    for r in all_results:
        photo = f"{r.get('photo_pred_l1', float('nan')):.4f}"
        print(f"  {r['seq']:<12s} {r['n_pairs']:>6d} {r['mean_epe']:>10.2f} "
              f"{r['outlier_rate']*100:>7.1f}% {photo:>10s}")

    # Save results
    if out_path:
        results_file = out_path / "results.npz"
        np.savez(results_file, results=all_results)
        print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optical flow quality against pose-based GT"
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing seq_* folders')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (unused currently, CPU-only evaluation)')
    parser.add_argument('--num_sequences', type=int, default=None,
                        help='Max number of sequences to evaluate')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frame pairs per sequence')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for output visualizations and results')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save comparison visualizations')

    args = parser.parse_args()

    evaluate_all(
        data_dir=args.data_dir,
        num_sequences=args.num_sequences,
        max_frames=args.max_frames,
        save_vis=args.save_vis,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
