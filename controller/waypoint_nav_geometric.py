#!/usr/bin/env python3
"""
Waypoint Navigation with MinSnap Trajectory + Geometric Controller

Main script for drone waypoint navigation using:
1. A* path planning for collision-free paths
2. MinSnap trajectory generation for smooth motion
3. SE(3) Geometric Controller with feedforward for accurate tracking

Usage:
    python waypoint_nav_geometric.py --map gate_mid --output output/geometric_nav.mp4
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import time

from envs.drone_env import SimpleDroneEnv
from utils.traj_planner_global import TrajectoryPlanner
from utils.common import get_time_stamp, print_info, print_ok, print_warning
from trajectory.bspline_trajectory import generate_bspline_trajectory, BSplineTrajectorySampler
from controller.geometric_controller import GeometricController


def define_waypoints(
    waypoints_list: list = None,
    map_name: str = 'gate_mid',
    start_pos: list = None,
    destination: list = None,
) -> tuple:
    """
    Define navigation waypoints.

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
            "start": [-6.0, 0.0, 1.2],
            "waypoints": [
                [-0.2, -0.1, 1.2],
                [1.6, 0.7, 1.2],
                [3.7, 1.5, 1.2],
                [5.8, 0.0, 1.2],
            ],
            "destination": [7.0, -2.0, 1.2],
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
    wp_distance: float = 0.5,
) -> tuple:
    """
    Plan collision-free path using A* algorithm.

    Args:
        current_pos: Current drone position [x, y, z]
        waypoints: Intermediate waypoints to pass through
        destination: Final destination [x, y, z]
        ply_file: Path to point cloud PLY file
        safety_distance: Safety margin around obstacles
        wp_distance: Distance between resampled waypoints

    Returns:
        path: List of [x, y, z] points forming the path
        planner: TrajectoryPlanner instance
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

    # Convert to tensors
    current_pos_tensor = torch.tensor([current_pos], device='cuda:0')
    destination_tensor = torch.tensor([destination], device='cuda:0')
    waypoints_tensor = [torch.tensor(waypoints, device='cuda:0')]

    # Plan trajectory
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


def generate_bspline_trajectory_from_path(
    path: list,
    v_avg: float = 1.5,
    corner_smoothing: float = 0.1,
) -> BSplineTrajectorySampler:
    """
    Generate B-Spline trajectory from path.

    B-Splines create:
    - Straight lines when waypoints are aligned
    - Smooth corner rounding (no overshoot)
    - Waypoints act as "magnets", path may not pass exactly through them

    Args:
        path: List of [x, y, z] waypoints
        v_avg: Average velocity (m/s)
        corner_smoothing: Corner smoothing factor (0 = sharp corners, 0.5+ = smoother)

    Returns:
        BSplineTrajectorySampler for querying trajectory
    """
    print_info("Generating B-Spline trajectory...")

    path_array = np.array(path)
    print(f"  Input waypoints: {len(path_array)}")

    # Generate B-spline trajectory
    sampler = generate_bspline_trajectory(
        waypoints=path_array,
        v_avg=v_avg,
        corner_smoothing=corner_smoothing,
    )

    print_ok(f"B-Spline trajectory generated: {sampler.total_time:.2f}s duration")
    print(f"  Corner smoothing: {corner_smoothing}")

    return sampler


def save_trajectory_profile(
    sampler: BSplineTrajectorySampler,
    output_path: str,
    dt: float = 0.02,
) -> dict:
    """
    Save and visualize the trajectory profile before flying.

    Args:
        sampler: BSplineTrajectorySampler with generated trajectory
        output_path: Path to save the plot
        dt: Time step for sampling (default 0.02s = 50Hz)

    Returns:
        Dictionary with trajectory data (time, pos, vel, acc)
    """
    import matplotlib.pyplot as plt

    print_info("Saving trajectory profile...")

    # Sample trajectory
    times = np.arange(0, sampler.total_time, dt)
    positions = []
    velocities = []
    accelerations = []

    for t in times:
        pos, vel, acc, _ = sampler.sample(t)
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    # Create figure with subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    # Row 0: 3D trajectory
    ax_3d = fig.add_subplot(4, 3, (1, 3), projection='3d')
    ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax_3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax_3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory')
    ax_3d.legend()

    # Clear the first row subplots (we used them for 3D)
    for i in range(3):
        axes[0, i].set_visible(False)

    # Row 1: Position vs time
    for i in range(3):
        axes[1, i].plot(times, positions[:, i], colors[i], linewidth=1.5)
        axes[1, i].set_ylabel(f'Pos {labels[i]} (m)')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].grid(True)
        axes[1, i].set_title(f'Position {labels[i]}')

    # Row 2: Velocity vs time
    for i in range(3):
        axes[2, i].plot(times, velocities[:, i], colors[i], linewidth=1.5)
        axes[2, i].set_ylabel(f'Vel {labels[i]} (m/s)')
        axes[2, i].set_xlabel('Time (s)')
        axes[2, i].grid(True)
        axes[2, i].set_title(f'Velocity {labels[i]}')

    # Row 3: Acceleration vs time (THIS IS KEY FOR OSCILLATION DIAGNOSIS)
    for i in range(3):
        axes[3, i].plot(times, accelerations[:, i], colors[i], linewidth=1.5)
        axes[3, i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # Mark high acceleration regions
        max_acc = np.max(np.abs(accelerations[:, i]))
        axes[3, i].axhline(y=2.0, color='orange', linestyle=':', alpha=0.5, label='±2 m/s²')
        axes[3, i].axhline(y=-2.0, color='orange', linestyle=':', alpha=0.5)
        axes[3, i].set_ylabel(f'Acc {labels[i]} (m/s²)')
        axes[3, i].set_xlabel('Time (s)')
        axes[3, i].grid(True)
        axes[3, i].set_title(f'Acceleration {labels[i]} (max: {max_acc:.2f})')
        axes[3, i].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print_ok(f"Trajectory profile saved: {output_path}")
    plt.close()

    # Print summary statistics
    print_info("Trajectory Statistics:")
    print(f"  Duration: {sampler.total_time:.2f}s")
    print(f"  Position range:")
    print(f"    X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"    Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
    print(f"    Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
    print(f"  Max velocity: [{np.max(np.abs(velocities[:, 0])):.2f}, {np.max(np.abs(velocities[:, 1])):.2f}, {np.max(np.abs(velocities[:, 2])):.2f}] m/s")
    print(f"  Max acceleration: [{np.max(np.abs(accelerations[:, 0])):.2f}, {np.max(np.abs(accelerations[:, 1])):.2f}, {np.max(np.abs(accelerations[:, 2])):.2f}] m/s²")

    # Warn if accelerations are too high
    max_acc_y = np.max(np.abs(accelerations[:, 1]))
    if max_acc_y > 2.0:
        print_warning(f"  ⚠ Y acceleration ({max_acc_y:.2f} m/s²) is high - may cause oscillation!")

    return {
        'time': times,
        'position': positions,
        'velocity': velocities,
        'acceleration': accelerations,
    }


def save_trajectory_topdown(
    sampler: BSplineTrajectorySampler,
    astar_path: list,
    waypoints: list,
    output_path: str,
    pc_offset: np.ndarray = None,
    dt: float = 0.02,
):
    """
    Save top-down (XY plane), side view (XZ plane), and 3D views comparing A* path and B-spline trajectory.

    Args:
        sampler: BSplineTrajectorySampler with generated trajectory
        astar_path: Raw A* path points (in point cloud space)
        waypoints: Waypoints in point cloud space
        output_path: Path to save the image
        pc_offset: Offset from point cloud to simulation space
        dt: Time step for sampling trajectory
    """
    import matplotlib.pyplot as plt

    if pc_offset is None:
        pc_offset = np.array([0.0, 0.0, 0.0])

    # Sample B-spline trajectory
    times = np.arange(0, sampler.total_time, dt)
    traj_points = []
    for t in times:
        pos, _, _, _ = sampler.sample(t)
        pos_pc = pos - pc_offset
        traj_points.append(pos_pc)
    traj_points = np.array(traj_points)

    # Convert A* path to array
    astar_points = np.array(astar_path)

    waypoints_arr = np.array(waypoints)

    # Create figure with 3x2 subplots (row1: XY, row2: XZ, row3: 3D)
    fig = plt.figure(figsize=(16, 18))

    # Row 1 - Top-down view (XY plane)
    # Top-left: A* raw path (XY)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(astar_points[:, 0], astar_points[:, 1], 'gray', linewidth=1, alpha=0.7)
    ax1.scatter(astar_points[:, 0], astar_points[:, 1], c='gray', s=10, alpha=0.5, label=f'A* ({len(astar_points)} pts)')
    ax1.scatter(waypoints_arr[:, 0], waypoints_arr[:, 1], c='orange', s=100, zorder=5, label='Waypoints')
    ax1.scatter(astar_points[0, 0], astar_points[0, 1], c='green', s=200, marker='o', zorder=6, label='Start')
    ax1.scatter(astar_points[-1, 0], astar_points[-1, 1], c='red', s=200, marker='x', zorder=6, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('A* Raw Path - Top Down (XY)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Top-right: B-spline trajectory (XY)
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(astar_points[:, 0], astar_points[:, 1], 'gray', linewidth=1, alpha=0.3, label='A* path')
    ax2.plot(traj_points[:, 0], traj_points[:, 1], 'b-', linewidth=2, label='B-spline')
    ax2.scatter(waypoints_arr[:, 0], waypoints_arr[:, 1], c='orange', s=100, zorder=5, label='Waypoints')
    ax2.scatter(traj_points[0, 0], traj_points[0, 1], c='green', s=200, marker='o', zorder=6, label='Start')
    ax2.scatter(traj_points[-1, 0], traj_points[-1, 1], c='red', s=200, marker='x', zorder=6, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('B-Spline Trajectory - Top Down (XY)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # Row 2 - Side view (XZ plane)
    # Middle-left: A* raw path (XZ)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(astar_points[:, 0], astar_points[:, 2], 'gray', linewidth=1, alpha=0.7)
    ax3.scatter(astar_points[:, 0], astar_points[:, 2], c='gray', s=10, alpha=0.5, label=f'A* ({len(astar_points)} pts)')
    ax3.scatter(waypoints_arr[:, 0], waypoints_arr[:, 2], c='orange', s=100, zorder=5, label='Waypoints')
    ax3.scatter(astar_points[0, 0], astar_points[0, 2], c='green', s=200, marker='o', zorder=6, label='Start')
    ax3.scatter(astar_points[-1, 0], astar_points[-1, 2], c='red', s=200, marker='x', zorder=6, label='End')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('A* Raw Path - Side View (XZ)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Middle-right: B-spline trajectory (XZ)
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(astar_points[:, 0], astar_points[:, 2], 'gray', linewidth=1, alpha=0.3, label='A* path')
    ax4.plot(traj_points[:, 0], traj_points[:, 2], 'b-', linewidth=2, label='B-spline')
    ax4.scatter(waypoints_arr[:, 0], waypoints_arr[:, 2], c='orange', s=100, zorder=5, label='Waypoints')
    ax4.scatter(traj_points[0, 0], traj_points[0, 2], c='green', s=200, marker='o', zorder=6, label='Start')
    ax4.scatter(traj_points[-1, 0], traj_points[-1, 2], c='red', s=200, marker='x', zorder=6, label='End')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('B-Spline Trajectory - Side View (XZ)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')

    # Row 3 - 3D views
    # Bottom-left: A* raw path (3D)
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    ax5.plot(astar_points[:, 0], astar_points[:, 1], astar_points[:, 2], 'gray', linewidth=1, alpha=0.7)
    ax5.scatter(astar_points[:, 0], astar_points[:, 1], astar_points[:, 2], c='gray', s=5, alpha=0.3)
    ax5.scatter(waypoints_arr[:, 0], waypoints_arr[:, 1], waypoints_arr[:, 2], c='orange', s=100, label='Waypoints')
    ax5.scatter(astar_points[0, 0], astar_points[0, 1], astar_points[0, 2], c='green', s=200, marker='o', label='Start')
    ax5.scatter(astar_points[-1, 0], astar_points[-1, 1], astar_points[-1, 2], c='red', s=200, marker='x', label='End')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_zlabel('Z (m)')
    ax5.set_title('A* Raw Path - 3D View')
    ax5.legend(fontsize=8)

    # Bottom-right: B-spline trajectory (3D)
    ax6 = fig.add_subplot(3, 2, 6, projection='3d')
    ax6.plot(astar_points[:, 0], astar_points[:, 1], astar_points[:, 2], 'gray', linewidth=1, alpha=0.3, label='A* path')
    ax6.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], 'b-', linewidth=2, label='B-spline')
    ax6.scatter(waypoints_arr[:, 0], waypoints_arr[:, 1], waypoints_arr[:, 2], c='orange', s=100, label='Waypoints')
    ax6.scatter(traj_points[0, 0], traj_points[0, 1], traj_points[0, 2], c='green', s=200, marker='o', label='Start')
    ax6.scatter(traj_points[-1, 0], traj_points[-1, 1], traj_points[-1, 2], c='red', s=200, marker='x', label='End')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Z (m)')
    ax6.set_title('B-Spline Trajectory - 3D View')
    ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print_ok(f"Trajectory views saved: {output_path}")


def fly_trajectory(
    env: SimpleDroneEnv,
    controller: GeometricController,
    sampler: BSplineTrajectorySampler,
    max_steps: int = 3000,
    frame_interval: int = 2,
    extra_hover_time: float = 2.0,
) -> tuple:
    """
    Fly drone through trajectory using Geometric Controller.

    Args:
        env: SimpleDroneEnv instance
        controller: GeometricController instance
        sampler: BSplineTrajectorySampler with generated trajectory
        max_steps: Maximum number of steps
        frame_interval: Record frame every N steps
        extra_hover_time: Extra time to hover at goal after trajectory ends

    Returns:
        frames: List of RGB frames
        trajectory_actual: List of actual positions
        trajectory_desired: List of desired positions
        errors: List of tracking errors
    """
    print_info("Starting trajectory tracking flight...")

    frames = []
    trajectory_actual = []
    trajectory_desired = []
    errors = []

    dt = 0.02  # Environment step time (50 Hz)
    total_time = sampler.total_time + extra_hover_time

    # Get final position for hovering after trajectory
    final_pos, _, _, _ = sampler.sample(sampler.total_time)

    # Stabilization period - hover at start for a moment
    stabilize_steps = 50
    start_pos_d, _, _, _ = sampler.sample(0.0)

    for step in range(max_steps):
        t = max(0, (step - stabilize_steps) * dt)

        # Check if we've completed the trajectory + hover time
        if t > total_time:
            print_ok(f"Trajectory completed at step {step}")
            break

        # Get current state
        state = env.get_state()
        pos = state['position'][0].cpu().numpy()
        vel = state['velocity'][0].cpu().numpy()
        quat_xyzw = state['orientation'][0].cpu().numpy()
        omega = state['angular_velocity'][0].cpu().numpy()

        # Convert quaternion from [x,y,z,w] to [w,x,y,z] for controller
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Sample desired trajectory
        if step < stabilize_steps:
            # Stabilization phase - hover at start
            pos_d = start_pos_d
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
            yaw_d = 0.0
        elif t <= sampler.total_time:
            pos_d, vel_d, acc_d, _ = sampler.sample(t)
            yaw_d = sampler.get_yaw_from_velocity(vel_d, default_yaw=0.0)
        else:
            # Hover at final position
            pos_d = final_pos
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
            yaw_d = 0.0

        # Compute control with feedforward
        thrust, omega_cmd, info = controller.compute_from_quaternion(
            pos, vel, quat_wxyz, omega,
            pos_d, vel_d, acc_d, yaw_d
        )

        # Convert to environment action format
        # Environment expects: [roll_rate, pitch_rate, yaw_rate, thrust]
        # Body rates in [-1, 1] (scaled to [-0.5, 0.5] rad/s internally)
        # Thrust in [0, 1]
        #
        # Note: The environment uses a convention where:
        # - Positive pitch rate = nose up (backward acceleration)
        # - Positive roll rate = right wing down (rightward acceleration)
        # The geometric controller uses standard aerospace convention, so we need to
        # negate pitch and roll to match the environment's simplified model.
        rate_scale = 0.5  # Environment scales by 0.5
        action = np.array([
            np.clip(omega_cmd[0] / rate_scale, -1.0, 1.0),   # roll rate (Y motion)
            np.clip(-omega_cmd[1] / rate_scale, -1.0, 1.0),  # pitch rate (negated for X motion)
            np.clip(omega_cmd[2] / rate_scale, -1.0, 1.0),   # yaw rate
            np.clip(thrust / controller.max_thrust, 0.0, 1.0),  # normalized thrust
        ])

        # Debug Y-axis control - show correction vs feedforward
        if step < 2000 and step % 50 == 0:
            e_p = pos - pos_d
            e_v = vel - vel_d
            correction_y = -controller.Kp[1] * e_p[1] - controller.Kv[1] * e_v[1]
            feedforward_y = controller.m * acc_d[1]
            print(f"  [Y-DEBUG] step={step}: e_p_y={e_p[1]:.3f} e_v_y={e_v[1]:.3f} acc_d_y={acc_d[1]:.2f}")
            print(f"            correction={correction_y:.3f} feedforward={feedforward_y:.3f} => F_des_y={info['F_des'][1]:.3f}")

        # Execute step
        _, rgb, depth, done, step_info = env.step(action)

        # Record data
        trajectory_actual.append(pos.copy())
        trajectory_desired.append(pos_d.copy())
        error = np.linalg.norm(pos - pos_d)
        errors.append(error)

        # Record frame
        if step % frame_interval == 0 and rgb is not None:
            frame = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

        # Progress update
        if step % 100 == 0:
            progress = min(t / sampler.total_time * 100, 100)
            print(f"  Step {step:4d} | t={t:.2f}s | Progress: {progress:.0f}% | "
                  f"Pos [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                  f"Error: {error*100:.1f}cm")

        # Check termination
        if done or step_info.get('collision', False):
            print_warning(f"Episode ended at step {step} (collision or timeout)")
            print(f"  Final pos: {pos}, vel: {vel}")
            break

    # Print statistics
    errors = np.array(errors)
    print_info("Tracking Statistics:")
    print(f"  Mean error: {np.mean(errors)*100:.2f} cm")
    print(f"  Max error:  {np.max(errors)*100:.2f} cm")
    print(f"  RMS error:  {np.sqrt(np.mean(errors**2))*100:.2f} cm")
    print(f"  Total frames: {len(frames)}")

    return frames, trajectory_actual, trajectory_desired, errors


def render_and_save(
    frames: list,
    output_path: str,
    fps: int = 25,
) -> None:
    """Save video from collected frames using H.264 codec."""
    import subprocess
    import shutil
    import cv2

    if len(frames) == 0:
        print_warning("No frames to save!")
        return

    print_info(f"Saving video to {output_path}...")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    height, width = frames[0].shape[:2]

    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        try:
            process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()

            if process.returncode == 0:
                print_ok(f"Video saved: {output_path}")
                print(f"  Resolution: {width}x{height}, Duration: {len(frames)/fps:.1f}s")
                return
        except Exception as e:
            print_warning(f"ffmpeg error: {e}")

    # Fallback to OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print_ok(f"Video saved (mp4v): {output_path}")


def save_trajectory_plot(
    trajectory_actual: list,
    trajectory_desired: list,
    errors: list,
    output_path: str,
) -> None:
    """Save trajectory comparison plot."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    actual = np.array(trajectory_actual)
    desired = np.array(trajectory_desired)
    errors = np.array(errors)

    fig = plt.figure(figsize=(15, 5))

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(desired[:, 0], desired[:, 1], desired[:, 2], 'b--', linewidth=2, label='Desired')
    ax1.plot(actual[:, 0], actual[:, 1], actual[:, 2], 'r-', linewidth=1.5, label='Actual')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # XY plane
    ax2 = fig.add_subplot(132)
    ax2.plot(desired[:, 0], desired[:, 1], 'b--', linewidth=2, label='Desired')
    ax2.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=1.5, label='Actual')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Trajectory')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True)

    # Error over time
    ax3 = fig.add_subplot(133)
    times = np.arange(len(errors)) * 0.02
    ax3.plot(times, errors * 100, 'k-', linewidth=1.5)
    ax3.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5cm threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (cm)')
    ax3.set_title(f'Tracking Error (Mean: {np.mean(errors)*100:.1f}cm)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print_ok(f"Trajectory plot saved: {output_path}")
    plt.close()


def main():
    """Main function for waypoint navigation with MinSnap + Geometric Controller."""
    parser = argparse.ArgumentParser(description='Drone Navigation with MinSnap + Geometric Controller')
    parser.add_argument('--map', type=str, default='gate_mid',
                        choices=['gate_mid', 'gate_left', 'gate_right', 'simple_hover'],
                        help='Map name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--max_steps', type=int, default=3000,
                        help='Maximum simulation steps')
    parser.add_argument('--fps', type=int, default=25,
                        help='Output video FPS')
    parser.add_argument('--v_avg', type=float, default=1.5,
                        help='Average velocity for trajectory (m/s)')
    parser.add_argument('--corner_smoothing', type=float, default=0.1,
                        help='B-spline corner smoothing (0=sharp, 0.5=moderate, 1+=very smooth)')
    parser.add_argument('--visualize_path', action='store_true',
                        help='Visualize planned path before flying')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save trajectory comparison plot')
    parser.add_argument('--save_traj_profile', action='store_true',
                        help='Save trajectory profile (pos/vel/acc) before flying')
    parser.add_argument('--traj_only', action='store_true',
                        help='Only generate and save trajectory, do not fly')
    args = parser.parse_args()

    print("=" * 80)
    print("Drone Navigation: MinSnap Trajectory + Geometric SE(3) Controller")
    print("=" * 80)

    # Set default output path
    if args.output is None:
        timestamp = get_time_stamp()
        args.output = f"output/geometric_{args.map}_{timestamp}.mp4"

    # Coordinate offset: point cloud space vs simulation space
    PC_TO_SIM_OFFSET = np.array([6.0, 0.0, 0.0])

    # 1. Define waypoints (in point cloud space)
    start_pos_pc, waypoints_pc, destination_pc = define_waypoints(map_name=args.map)

    # 2. Get A* path (in point cloud space)
    ply_file = Path(__file__).parent.parent / "envs" / "assets" / "point_cloud" / "sv_1007_gate_mid.ply"
    path_pc, planner = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=0.2,  # Denser path for smoother Z
    )

    # Visualize path if requested
    if args.visualize_path:
        print_info("Visualizing planned path...")
        planner.visualize_trajectories()

    # 3. Convert path to simulation space
    path_sim = [np.array(p) + PC_TO_SIM_OFFSET for p in path_pc]
    start_pos_sim = np.array(start_pos_pc) + PC_TO_SIM_OFFSET

    print_info("Converted to simulation coordinates")
    print(f"  Start (sim): {start_pos_sim.tolist()}")
    print(f"  Path points: {len(path_sim)}")

    # 4. Generate B-Spline trajectory (straight lines + smooth corners)
    sampler = generate_bspline_trajectory_from_path(
        path=path_sim,
        v_avg=args.v_avg,
        corner_smoothing=args.corner_smoothing,
    )

    # 4.5 Save trajectory profile and visualize if requested
    if args.save_traj_profile or args.traj_only:
        traj_profile_path = args.output.replace('.mp4', '_traj_profile.png')
        save_trajectory_profile(sampler, traj_profile_path)

        if args.traj_only:
            # Save top-down XY view comparing A* and B-spline
            topdown_path = args.output.replace('.mp4', '_topdown.png')
            save_trajectory_topdown(
                sampler=sampler,
                astar_path=path_pc,
                waypoints=[start_pos_pc] + waypoints_pc + [destination_pc],
                output_path=topdown_path,
                pc_offset=PC_TO_SIM_OFFSET,
            )

            print("=" * 80)
            print("Trajectory generation complete (--traj_only mode)")
            print(f"Profile: {traj_profile_path}")
            print(f"Top-down: {topdown_path}")
            print("=" * 80)
            return

    # 5. Initialize environment
    print_info("Initializing environment...")
    env = SimpleDroneEnv(
        map_name=args.map,
        device='cuda:0',
        num_envs=1,
        episode_length=args.max_steps,
        render_resolution=1.0,
    )
    env.reset(start_position=start_pos_sim.tolist())

    # 6. Initialize Geometric Controller
    # Conservative gains to prevent oscillation and flipping
    controller = GeometricController(
        mass=env.mass,
        gravity=9.81,
        # Position gains
        Kp=np.array([2.5, 3.0, 0.5]),   # [X, Y, Z]
        Kv=np.array([2.0, 2.0, 2.5]),   # [X, Y, Z] - higher Kv_y for damping
        # Lower attitude gains for stability
        Kr=np.array([3.0, 3.0, 2.0]),   # Attitude gains
        Kw=np.array([0.8, 0.8, 0.5]),   # Angular velocity gains
        max_thrust=env.max_thrust,
        min_thrust=2.0,
        max_rate=1.0,  # rad/s
    )

    print_info("Controller initialized")
    print(f"  Mass: {controller.m} kg")
    print(f"  Kp: {controller.Kp}")
    print(f"  Kv: {controller.Kv}")

    # 7. Fly through trajectory
    frames, traj_actual, traj_desired, errors = fly_trajectory(
        env=env,
        controller=controller,
        sampler=sampler,
        max_steps=args.max_steps,
        frame_interval=2,
        extra_hover_time=2.0,
    )

    # 8. Save video
    render_and_save(frames, args.output, fps=args.fps)

    # 9. Save trajectory plot if requested
    if args.save_plot:
        plot_path = args.output.replace('.mp4', '_trajectory.png')
        save_trajectory_plot(traj_actual, traj_desired, errors, plot_path)

    # Cleanup
    env.close()

    print("=" * 80)
    print("Navigation complete!")
    print(f"Video: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
