#!/usr/bin/env python3
"""
Waypoint Navigation Script

Main script for drone waypoint navigation with the following functions:
1. define_waypoints - Define or load navigation waypoints
2. get_path - Plan collision-free path using A* algorithm
3. fly - Fly drone through path using PD controller
4. render_and_save - Save video from collected frames

Usage:
    python waypoint_nav.py --map gate_mid --output output/nav_video.mp4
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import cv2
import torch

from envs.drone_env import SimpleDroneEnv
from utils.traj_planner_global import TrajectoryPlanner
from utils.common import get_time_stamp, print_info, print_ok, print_warning
from drone_pd_controller.pd_controller import SimplePDController, WaypointFollower


def define_waypoints(
    waypoints_list: list = None,
    map_name: str = 'gate_mid',
    start_pos: list = None,
    destination: list = None,
) -> tuple:
    """
    Define navigation waypoints.

    Can load from provided list or use default map waypoints.

    Args:
        waypoints_list: Optional list of [x, y, z] waypoints
        map_name: Name of the map (used for default waypoints)
        start_pos: Start position [x, y, z]
        destination: Destination position [x, y, z]

    Returns:
        start_pos: Starting position
        waypoints: List of intermediate waypoints
        destination: Final destination
    """
    # Default configurations per map
    map_configs = {
        "gate_mid": {
            "start": [-6.0, 0.0, 1.25],
            "waypoints": [
                [-0.2, -0.1, 1.4],
                [1.6, 0.7, 1.1],
                [3.7, 1.5, 0.7],
                [5.8, 0.0, 1.2],
            ],
            "destination": [7.0, -2.0, 1.3],
        },
        "gate_left": {
            "start": [-6.0, 0.0, 1.2],
            "waypoints": [
                [-0.2, 1.2, 1.4],
                [3.7, 1.2, 0.6],
                [5.8, 0.0, 1.2],
            ],
            "destination": [7.0, -2.0, 1.2],
        },
        "gate_right": {
            "start": [-6.0, 0.0, 1.3],
            "waypoints": [
                [-3.0, -1.0, 1.3],
                [0.0, -1.4, 1.5],
                [1.8, 0.6, 1.1],
                [3.7, 1.4, 0.7],
                [5.8, 0.0, 1.3],
            ],
            "destination": [7.0, -2.0, 1.3],
        },
        "simple_hover": {
            "start": [-6.0, 0.0, 1.2],
            "waypoints": [],
            "destination": [-6.0, 0.0, 1.2],
        },
    }

    config = map_configs.get(map_name, map_configs["gate_mid"])

    # Use provided values or defaults
    if start_pos is None:
        start_pos = config["start"]
    if waypoints_list is None:
        waypoints_list = config["waypoints"]
    if destination is None:
        destination = config["destination"]

    print_info(f"Waypoints defined for map '{map_name}':")
    print(f"  Start: {start_pos}")
    print(f"  Waypoints: {len(waypoints_list)} points")
    for i, wp in enumerate(waypoints_list):
        print(f"    {i+1}. {wp}")
    print(f"  Destination: {destination}")

    return start_pos, waypoints_list, destination


def get_path(
    current_pos: list,
    waypoints: list,
    destination: list,
    ply_file: str,
    safety_distance: float = 0.15,
    wp_distance: float = 2.0,
) -> tuple:
    """
    Plan collision-free path using A* algorithm.

    Args:
        current_pos: Current drone position [x, y, z] (in point cloud space)
        waypoints: Intermediate waypoints to pass through (in point cloud space)
        destination: Final destination [x, y, z] (in point cloud space)
        ply_file: Path to point cloud PLY file
        safety_distance: Safety margin around obstacles
        wp_distance: Distance between resampled waypoints

    Returns:
        path: List of [x, y, z] points forming the path (in point cloud space)
        planner: TrajectoryPlanner instance (for visualization)
    """
    print_info("Planning path with A* algorithm...")

    # Initialize trajectory planner
    planner = TrajectoryPlanner(
        ply_file=ply_file,
        safety_distance=safety_distance,
        batch_size=1,
        wp_distance=wp_distance,
        verbose=True,
    )

    # Print point cloud bounds
    print(f"  Point cloud X range: [{planner.points[:, 0].min():.2f}, {planner.points[:, 0].max():.2f}]")
    print(f"  Point cloud Y range: [{planner.points[:, 1].min():.2f}, {planner.points[:, 1].max():.2f}]")
    print(f"  Point cloud Z range: [{planner.points[:, 2].min():.2f}, {planner.points[:, 2].max():.2f}]")

    # Convert to tensors
    current_pos_tensor = torch.tensor([current_pos], device='cuda:0')
    destination_tensor = torch.tensor([destination], device='cuda:0')
    waypoints_tensor = [torch.tensor(waypoints, device='cuda:0')]

    # Plan trajectory
    import time
    start_time = time.time()
    trajectories = planner.plan_trajectories(
        current_pos_tensor,
        destination_tensor,
        waypoints_tensor,
    )
    elapsed = time.time() - start_time

    if trajectories[0] is None:
        print_warning("Path planning failed! Using direct waypoints.")
        path = [current_pos] + waypoints + [destination]
    else:
        path = trajectories[0]
        print_ok(f"Path planned successfully in {elapsed:.2f}s")
        print(f"  Path length: {len(path)} points")

    return path, planner


def fly(
    env: SimpleDroneEnv,
    controller: SimplePDController,
    path: list,
    max_steps: int = 2000,
    frame_interval: int = 2,
    waypoint_threshold: float = 0.5,
) -> tuple:
    """
    Fly drone through path using PD controller.

    Args:
        env: SimpleDroneEnv instance
        controller: SimplePDController instance
        path: List of [x, y, z] waypoints to follow
        max_steps: Maximum number of steps
        frame_interval: Record frame every N steps
        waypoint_threshold: Distance to waypoint to consider it reached

    Returns:
        frames: List of RGB frames (numpy arrays)
        trajectory: List of position records
    """
    print_info("Starting flight...")

    # Create waypoint follower
    follower = WaypointFollower(controller, waypoint_threshold=waypoint_threshold)
    follower.set_waypoints(path)

    frames = []
    trajectory = []
    goal_reached = False

    for step in range(max_steps):
        # Get current state
        state = env.get_state()
        pos = state['position'][0].cpu().numpy()
        vel = state['velocity'][0].cpu().numpy()

        # Record trajectory
        trajectory.append(pos.copy())

        # Compute action
        action, goal_reached = follower.compute_action(pos, vel)

        # Execute step
        _, rgb, depth, done, info = env.step(action)

        # Record frame
        if step % frame_interval == 0 and rgb is not None:
            # Convert to uint8 numpy array
            frame = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

        # Progress update
        if step % 50 == 0:
            wp_idx, total_wp = follower.get_progress()
            target = follower.get_current_target()
            dist = np.linalg.norm(target - pos) if target is not None else 0
            print(f"  Step {step}/{max_steps} | WP {wp_idx+1}/{total_wp} | "
                  f"Pos [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                  f"Dist {dist:.2f}m")

        # Check termination
        if goal_reached:
            print_ok(f"Goal reached at step {step}!")
            break

        if done or info.get('collision', False):
            print_warning(f"Episode ended at step {step} (collision or timeout)")
            break

    print(f"  Total frames collected: {len(frames)}")
    print(f"  Trajectory points: {len(trajectory)}")

    return frames, trajectory


def render_and_save(
    frames: list,
    output_path: str,
    fps: int = 20,
) -> None:
    """
    Save video from collected frames using H.264 codec for VS Code compatibility.

    Uses ffmpeg subprocess for H.264 encoding (piping frames directly).

    Args:
        frames: List of RGB numpy arrays (H, W, 3)
        output_path: Path to save video file
        fps: Frames per second
    """
    import subprocess
    import shutil

    if len(frames) == 0:
        print_warning("No frames to save!")
        return

    print_info(f"Saving video to {output_path}...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path:
        # Use ffmpeg with H.264 encoding (pipe frames via stdin)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            # Write frames to ffmpeg
            for frame in frames:
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()

            if process.returncode == 0:
                print_ok(f"Video saved successfully with H.264!")
                print(f"  Resolution: {width}x{height}")
                print(f"  Duration: {len(frames)/fps:.1f} seconds")
                print(f"  FPS: {fps}")
                print(f"  Codec: H.264 (libx264)")
                print(f"  Path: {output_path}")
                return
            else:
                stderr = process.stderr.read().decode()
                print_warning(f"ffmpeg failed: {stderr[:200]}")
                print("Falling back to OpenCV...")
        except Exception as e:
            print_warning(f"ffmpeg error: {e}")
            print("Falling back to OpenCV...")

    # Fallback to OpenCV with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print_warning("Could not open video writer!")
        return

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

    print_ok(f"Video saved (mp4v codec - may not preview in VS Code)")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {len(frames)/fps:.1f} seconds")
    print(f"  FPS: {fps}")
    print(f"  Path: {output_path}")
    print_warning("To convert to H.264 for VS Code preview, run:")
    print(f"  ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 22 {output_path.replace('.mp4', '_h264.mp4')}")


def main():
    """Main function for waypoint navigation."""
    parser = argparse.ArgumentParser(description='Drone Waypoint Navigation')
    parser.add_argument('--map', type=str, default='gate_mid',
                        choices=['gate_mid', 'gate_left', 'gate_right', 'simple_hover'],
                        help='Map name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum simulation steps')
    parser.add_argument('--fps', type=int, default=20,
                        help='Output video FPS')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--visualize_path', action='store_true',
                        help='Visualize planned path before flying')
    args = parser.parse_args()

    print("=" * 80)
    print("Drone Waypoint Navigation")
    print("=" * 80)

    # Set default output path if not specified
    if args.output is None:
        timestamp = get_time_stamp()
        args.output = f"output/{args.map}_{timestamp}.mp4"

    # Coordinate offset: point cloud space vs simulation space
    # Point cloud: x from -6 to +7, GS/simulation: x from 0 to +13
    PC_TO_SIM_OFFSET = np.array([6.0, 0.0, 0.0])

    # 1. Define waypoints (in point cloud space for path planning)
    start_pos_pc, waypoints_pc, destination_pc = define_waypoints(map_name=args.map)

    # 2. Get path (in point cloud space)
    ply_file = Path(__file__).parent.parent / "envs" / "assets" / "point_cloud" / f"sv_1007_gate_mid.ply"
    path_pc, planner = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=1.0,  # Reduced from 2.0 for more path points
    )

    # Optionally visualize path (in point cloud space)
    if args.visualize_path:
        print_info("Visualizing planned path (close window to continue)...")
        print("  - Gray points: obstacles (point cloud)")
        print("  - Green tube: planned path")
        print("  - Blue spheres: waypoints")
        print("  - Red sphere: destination")
        planner.visualize_trajectories()

    # 3. Convert path to simulation space (add offset)
    print_info("Converting path to simulation coordinates...")
    path_sim = [np.array(p) + PC_TO_SIM_OFFSET for p in path_pc]
    start_pos_sim = np.array(start_pos_pc) + PC_TO_SIM_OFFSET
    print(f"  Start (sim space): {start_pos_sim.tolist()}")
    print(f"  Path points: {len(path_sim)}")

    # 4. Initialize environment
    print_info("Initializing environment...")
    env = SimpleDroneEnv(
        map_name=args.map,
        device='cuda:0',
        num_envs=1,
        episode_length=args.max_steps,
        render_resolution=1.0,  # Full resolution 640x360
    )

    # Reset environment with simulation space start position
    env.reset(start_position=start_pos_sim.tolist())

    # 4. Initialize controller (separate gains for X, Y, Z)
    controller = SimplePDController(
        hover_thrust=env.get_hover_thrust(),
        kp_x=0.15,    # X (forward) position gain
        kp_y=0.05,    # Y (lateral) position gain - lower to reduce oscillation
        kp_z=0.1,     # Z (altitude) position gain

        kd_x=0.3,     # X velocity damping
        kd_y=0.5,    # Y velocity damping - higher to reduce oscillation
        kd_z=0.08,    # Z velocity damping
        rate_limit=0.15,
    )

    # 5. Fly through path (using simulation space coordinates)
    frames, trajectory = fly(
        env=env,
        controller=controller,
        path=path_sim,
        max_steps=args.max_steps,
        frame_interval=2,
    )

    # 6. Render and save video
    render_and_save(frames, args.output, fps=args.fps)

    # Cleanup
    env.close()

    print("=" * 80)
    print("Navigation complete!")
    print(f"Output: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
