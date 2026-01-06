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

from envs.drone_env import SimpleDroneEnv
from utils.common import get_time_stamp, print_info, print_ok
from controller.geometric_controller import GeometricController
from controller.nav_helpers import (
    get_path,
    generate_bspline_trajectory_from_path,
    save_trajectory_profile,
    save_trajectory_topdown,
    fly_trajectory,
    render_and_save,
    save_trajectory_plot,
    save_trajectory_3d_plot,
    save_trajectory_data,
)


# =============================================================================
# Waypoint Configurations
# =============================================================================

MAP_CONFIGS = {
    "gate_mid": {
        "start": [-6.0, 0.0, 1.2],
        "waypoints": [
            [-0.2, -0.1, 1.2],
            [1.6, 0.7, 1.1],
            [3.7, 1.5, 0.7],
            [5.8, 0.0, 0.9],
        ],
        "destination": [7.5, -2.0, 1.2],
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

# Coordinate offset: point cloud space vs simulation space
PC_TO_SIM_OFFSET = np.array([6.0, 0.0, 0.0])


# =============================================================================
# Waypoint Definition
# =============================================================================

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
    config = MAP_CONFIGS.get(map_name, MAP_CONFIGS["gate_mid"])

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


# =============================================================================
# Main Function
# =============================================================================

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
    parser.add_argument('--corner_smoothing', type=float, default=0.18,
                        help='B-spline corner smoothing (0=sharp, 0.5=moderate, 1+=very smooth)')
    parser.add_argument('--visualize_path', action='store_true',
                        help='Visualize planned path before flying')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save trajectory comparison plot')
    parser.add_argument('--save_traj_profile', action='store_true',
                        help='Save trajectory profile (pos/vel/acc) before flying')
    parser.add_argument('--save_3d_plot', action='store_true',
                        help='Save 3D trajectory plot with multiple views')
    parser.add_argument('--save_traj_data', action='store_true',
                        help='Save trajectory data to file (.npz)')
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

    # 1. Define waypoints (in point cloud space)
    start_pos_pc, waypoints_pc, destination_pc = define_waypoints(map_name=args.map)

    # 2. Get A* path (in point cloud space)
    ply_file = Path(__file__).parent.parent / "envs" / "assets" / "point_cloud" / "sv_1007_gate_mid.ply"
    path_pc, planner = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=0.5,  # Denser path for smoother Z
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
        Kp=np.array([1.5, 1.5, 0.5]),   # [X, Y, Z]
        Kv=np.array([2.0, 2.0, 1.5]),   # [X, Y, Z] - higher Kv_y for damping
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

    # 10. Save 3D trajectory plot if requested
    if args.save_3d_plot:
        plot_3d_path = args.output.replace('.mp4', '_3d_trajectory.png')
        # Convert waypoints to simulation space for visualization
        waypoints_sim = [np.array(start_pos_pc) + PC_TO_SIM_OFFSET] + \
                        [np.array(wp) + PC_TO_SIM_OFFSET for wp in waypoints_pc] + \
                        [np.array(destination_pc) + PC_TO_SIM_OFFSET]
        save_trajectory_3d_plot(
            trajectory_actual=traj_actual,
            trajectory_desired=traj_desired,
            waypoints=waypoints_sim,
            output_path=plot_3d_path,
        )

    # 11. Save trajectory data if requested
    if args.save_traj_data:
        data_path = args.output.replace('.mp4', '_trajectory_data.npz')
        save_trajectory_data(traj_actual, traj_desired, errors, data_path)

    # Cleanup
    env.close()

    print("=" * 80)
    print("Navigation complete!")
    print(f"Video: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
