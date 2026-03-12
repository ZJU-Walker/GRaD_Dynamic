#!/usr/bin/env python3
"""
Visualize precomputed RAFT-Small optical flow.

Loads optical_flow.npz from a sequence directory and produces either a
grid image (default) or a video showing RGB frames alongside flow
color-coded with the standard HSV convention (angle → hue, magnitude → value).

The grid shows three columns per row:
  [RGB frame | Flow direction (HSV) | Flow magnitude (grayscale)]

All flow panels use global magnitude normalization so brightness is
comparable across frames.  A color wheel legend and per-row annotations
(frame index, max/mean magnitude) are included for readability.

Usage:
    # Grid image (saved as flow_visualization.png in the sequence dir)
    python training/vel_net_v2/visualize_optical_flow.py \
        --seq_dir /path/to/seq_0000

    # Video output
    python training/vel_net_v2/visualize_optical_flow.py \
        --seq_dir /path/to/seq_0000 --save_video

    # Custom sample count
    python training/vel_net_v2/visualize_optical_flow.py \
        --seq_dir /path/to/seq_0000 --num_samples 8
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color wheel legend
# ---------------------------------------------------------------------------

def make_color_wheel(size: int = 100) -> np.ndarray:
    """
    Generate a color wheel showing the HSV direction→color mapping.

    Returns an (size, size, 3) uint8 RGB image with:
      - Hue   = angle around center
      - Value = radial distance from center (normalized)
      - Sat   = 255
    Axis labels (U/D/L/R) are drawn around the wheel.
    """
    cy, cx = size // 2, size // 2
    radius = size // 2 - 12  # leave room for labels

    # Build coordinate grids relative to center
    ys, xs = np.mgrid[:size, :size]
    dy = ys - cy
    dx = xs - cx
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)  # [-pi, pi]

    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(dist / radius, 0, 1) * 255).astype(np.uint8)

    # Mask outside the circle to black
    hsv[dist > radius, :] = 0

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Draw axis labels (work in BGR for cv2.putText, then convert)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thick = 1
    color = (255, 255, 255)
    # Right (+x)
    cv2.putText(rgb, "R", (size - 11, cy + 4), font, scale, color, thick, cv2.LINE_AA)
    # Left (-x)
    cv2.putText(rgb, "L", (1, cy + 4), font, scale, color, thick, cv2.LINE_AA)
    # Down (+y)
    cv2.putText(rgb, "D", (cx - 4, size - 2), font, scale, color, thick, cv2.LINE_AA)
    # Up (-y)
    cv2.putText(rgb, "U", (cx - 4, 10), font, scale, color, thick, cv2.LINE_AA)

    return rgb


# ---------------------------------------------------------------------------
# Flow → image helpers
# ---------------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray, max_mag: float = None, gamma: float = 0.4) -> np.ndarray:
    """
    Convert a single optical flow field to an RGB color image.

    Uses the standard HSV encoding:
      - Hue   = flow angle (direction of motion)
      - Sat   = 255
      - Value = flow magnitude (clamped to [0, 1] after normalizing by max)

    Args:
        flow: (2, H, W) float array with channels [flow_x, flow_y].
        max_mag: If provided and > 0, normalize magnitudes by this value
                 instead of the per-frame maximum.  Enables cross-frame
                 brightness comparison.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    fx, fy = flow[0], flow[1]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)  # radians in [-pi, pi]

    # Build HSV image
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # H: [0, 180)
    hsv[..., 1] = 255  # S: full saturation

    if max_mag is None or max_mag <= 0:
        max_mag = magnitude.max()
    if max_mag > 0:
        normed = np.clip(magnitude / max_mag, 0, 1)
        hsv[..., 2] = (np.power(normed, gamma) * 255).astype(np.uint8)
    # else: leave V at 0 (no motion)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def flow_to_magnitude_gray(flow: np.ndarray, max_mag: float = None, gamma: float = 0.4) -> np.ndarray:
    """
    Convert a flow field to a grayscale magnitude image (bright = fast).

    Args:
        flow: (2, H, W) float array.
        max_mag: Global max for normalization.

    Returns:
        (H, W, 3) uint8 RGB image (grayscale replicated across channels).
    """
    magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    if max_mag is None or max_mag <= 0:
        max_mag = magnitude.max()
    if max_mag > 0:
        normed = np.clip(magnitude / max_mag, 0, 1)
    else:
        normed = np.zeros_like(magnitude)
    gray = (np.power(normed, gamma) * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

_ARROW_MAP = {
    "R": "\u2192", "L": "\u2190", "U": "\u2191", "D": "\u2193",
    "UR": "\u2197", "UL": "\u2196", "DR": "\u2198", "DL": "\u2199",
}


def _direction_label(mean_fx: float, mean_fy: float) -> str:
    """Return a short cardinal/ordinal label for the mean flow vector."""
    angle_deg = np.degrees(np.arctan2(mean_fy, mean_fx))  # [-180, 180]
    # Quantize to 8 compass directions
    if -22.5 <= angle_deg < 22.5:
        return "R"
    elif 22.5 <= angle_deg < 67.5:
        return "DR"
    elif 67.5 <= angle_deg < 112.5:
        return "D"
    elif 112.5 <= angle_deg < 157.5:
        return "DL"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return "L"
    elif -157.5 <= angle_deg < -112.5:
        return "UL"
    elif -112.5 <= angle_deg < -67.5:
        return "U"
    else:
        return "UR"


def annotate_row(image: np.ndarray, frame_idx: int, flow: np.ndarray) -> None:
    """
    Draw text annotations on the left side of a row image (in-place).

    Annotations:
      - Frame index (e.g., "frame 15 → 16")
      - Max magnitude in pixels
      - Mean flow direction as a cardinal label
    """
    magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    max_m = float(magnitude.max())
    mean_m = float(magnitude.mean())
    mean_fx = float(flow[0].mean())
    mean_fy = float(flow[1].mean())
    direction = _direction_label(mean_fx, mean_fy)

    lines = [
        f"frame {frame_idx} -> {frame_idx + 1}",
        f"max: {max_m:.1f} px",
        f"mean: {mean_m:.1f} px ({direction})",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    fg = (255, 255, 255)
    bg = (0, 0, 0)

    y0 = 18
    for i, text in enumerate(lines):
        y = y0 + i * 18
        # Draw black outline for readability
        cv2.putText(image, text, (5, y), font, scale, bg, thick + 1, cv2.LINE_AA)
        cv2.putText(image, text, (5, y), font, scale, fg, thick, cv2.LINE_AA)


def overlay_color_wheel(image: np.ndarray, wheel: np.ndarray, margin: int = 5) -> None:
    """Overlay the color wheel in the top-right corner of *image* (in-place)."""
    wh, ww = wheel.shape[:2]
    ih, iw = image.shape[:2]
    y0 = margin
    x0 = iw - ww - margin
    if y0 + wh > ih or x0 < 0:
        return  # image too small
    # Only write non-black pixels so the circle blends on dark backgrounds
    mask = wheel.sum(axis=-1) > 0
    roi = image[y0:y0 + wh, x0:x0 + ww]
    roi[mask] = wheel[mask]


# ---------------------------------------------------------------------------
# Grid and video builders
# ---------------------------------------------------------------------------

def load_sequence(seq_dir: Path):
    """Load RGB frames and precomputed flows from a sequence directory."""
    flow_path = seq_dir / "optical_flow.npz"
    if not flow_path.exists():
        raise FileNotFoundError(f"No optical_flow.npz in {seq_dir}")

    flows = np.load(flow_path)["flows"]  # (N-1, 2, H, W)

    rgb_dir = seq_dir / "rgb"
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.png")))
    if len(rgb_files) == 0:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")

    frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in rgb_files]
    return frames, flows


def make_grid(seq_dir: Path, num_samples: int = 6):
    """Create and save a grid image with RGB + flow color + flow magnitude."""
    frames, flows = load_sequence(seq_dir)
    n_flows = len(flows)

    # Sample indices evenly
    if num_samples >= n_flows:
        indices = list(range(n_flows))
    else:
        indices = np.linspace(0, n_flows - 1, num_samples, dtype=int).tolist()

    # Global max magnitude for consistent normalization
    global_max = float(np.max([np.sqrt(flows[i][0] ** 2 + flows[i][1] ** 2).max()
                               for i in indices]))

    wheel = make_color_wheel(size=60)

    rows = []
    for idx in indices:
        rgb = frames[idx + 1]
        flow_rgb = flow_to_rgb(flows[idx], max_mag=global_max)
        mag_gray = flow_to_magnitude_gray(flows[idx], max_mag=global_max)

        # Resize flow panels to match RGB if shapes differ
        if flow_rgb.shape[:2] != rgb.shape[:2]:
            flow_rgb = cv2.resize(flow_rgb, (rgb.shape[1], rgb.shape[0]))
            mag_gray = cv2.resize(mag_gray, (rgb.shape[1], rgb.shape[0]))

        # Three-column row: RGB | Flow HSV | Flow magnitude
        row = np.concatenate([rgb, flow_rgb, mag_gray], axis=1)

        # Annotate with frame info (drawn on the RGB portion)
        annotate_row(row, idx, flows[idx])

        # Overlay color wheel on the flow HSV column (top-right of that column)
        col_w = rgb.shape[1]
        flow_col = row[:, col_w:2 * col_w]
        overlay_color_wheel(flow_col, wheel)

        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    # Add column headers
    col_w = frames[0].shape[1]
    header_h = 24
    header = np.zeros((header_h, col_w * 3, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    fg = (255, 255, 255)
    labels = ["RGB", "Flow Direction (HSV)", "Flow Magnitude"]
    for i, label in enumerate(labels):
        tw = cv2.getTextSize(label, font, scale, thick)[0][0]
        x = i * col_w + (col_w - tw) // 2
        cv2.putText(header, label, (x, 17), font, scale, fg, thick, cv2.LINE_AA)

    grid = np.concatenate([header, grid], axis=0)

    # Add global-max info at bottom
    footer_h = 20
    footer = np.zeros((footer_h, col_w * 3, 3), dtype=np.uint8)
    info = f"Global max magnitude: {global_max:.1f} px"
    cv2.putText(footer, info, (5, 14), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    grid = np.concatenate([grid, footer], axis=0)

    out_path = seq_dir / "flow_visualization.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved grid image ({len(indices)} samples) -> {out_path}")


def make_video(seq_dir: Path, fps: int = 10):
    """Create and save a video with RGB | flow HSV | flow magnitude."""
    frames, flows = load_sequence(seq_dir)

    # Global max magnitude across all frames
    global_max = float(np.max([np.sqrt(flows[i][0] ** 2 + flows[i][1] ** 2).max()
                               for i in range(len(flows))]))

    h, w = frames[0].shape[:2]
    wheel = make_color_wheel(size=min(60, h // 3))

    out_path = seq_dir / "flow_visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * 3, h))

    for i in range(len(flows)):
        rgb = frames[i + 1]
        flow_rgb = flow_to_rgb(flows[i], max_mag=global_max)
        mag_gray = flow_to_magnitude_gray(flows[i], max_mag=global_max)
        if flow_rgb.shape[:2] != (h, w):
            flow_rgb = cv2.resize(flow_rgb, (w, h))
            mag_gray = cv2.resize(mag_gray, (w, h))

        row = np.concatenate([rgb, flow_rgb, mag_gray], axis=1)
        annotate_row(row, i, flows[i])

        flow_col = row[:, w:2 * w]
        overlay_color_wheel(flow_col, wheel)

        writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved video ({len(flows)} frames, {fps} fps) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize precomputed optical flow for a sequence."
    )
    parser.add_argument(
        "--seq_dir", type=str, required=True,
        help="Path to a sequence directory containing optical_flow.npz",
    )
    parser.add_argument(
        "--num_samples", type=int, default=6,
        help="Number of evenly-spaced samples for the grid image (default: 6)",
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Save a video instead of (in addition to) the grid image",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for video output (default: 10)",
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    if not seq_dir.is_dir():
        raise ValueError(f"Not a directory: {seq_dir}")

    make_grid(seq_dir, num_samples=args.num_samples)

    if args.save_video:
        make_video(seq_dir, fps=args.fps)


if __name__ == "__main__":
    main()
