"""
Data Collector for Velocity Network Training.

Follows the same workflow as waypoint_nav_geometric.py:
1. Define waypoints in point cloud space
2. A* path planning in point cloud space
3. Convert to simulation space (+PC_TO_SIM_OFFSET)
4. Generate B-spline trajectory in simulation space
5. Save planning plots (A* path, B-spline trajectory, velocity/acceleration profiles)
6. Fly and collect data

Output structure:
    data/vel_net/sequences/
    ├── seq_0000/
    │   ├── telemetry.npz         # timestamps, positions, velocities, orientations, actions
    │   ├── rgb/                  # RGB images
    │   ├── depth/                # Depth images
    │   ├── astar_bspline.png     # A* path vs B-spline comparison (saved BEFORE flying)
    │   ├── trajectory_profile.png # Position, velocity, acceleration profiles (saved BEFORE flying)
    │   └── trajectory.png        # Actual vs desired trajectory (saved AFTER flying)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from PIL import Image
from typing import Optional, List, Dict, Tuple
import time
from tqdm import tqdm

from envs.drone_env import SimpleDroneEnv
from controller.geometric_controller import GeometricController
from controller.nav_helpers import (
    get_path,
    generate_bspline_trajectory_from_path,
    save_trajectory_3d_plot,
    save_trajectory_topdown,
    save_trajectory_profile,
)
from trajectory.bspline_trajectory import BSplineTrajectorySampler


# Coordinate offset: point cloud space -> simulation space
# Point cloud: start = [-6, 0, 1.2]
# Simulation:  start = [0, 0, 1.2]
PC_TO_SIM_OFFSET = np.array([6.0, 0.0, 0.0])

# Map configurations (in POINT CLOUD space, same as waypoint_nav_geometric.py)
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
}


class DataCollector:
    """
    Collects synchronized telemetry and images during flight.
    """

    def __init__(
        self,
        env: SimpleDroneEnv,
        controller: GeometricController,
        output_dir: str = "data/vel_net/sequences",
        collection_freq: float = 30.0,
    ):
        self.env = env
        self.controller = controller
        self.output_dir = Path(output_dir)
        self.collection_freq = collection_freq

        # Sim runs at 50 Hz, we collect at specified freq
        self.sim_freq = env.sim_freq
        self.collection_interval = max(1, int(self.sim_freq / self.collection_freq))

        self.reset_buffers()

    def reset_buffers(self):
        """Reset all data buffers."""
        self.timestamps = []
        self.positions = []
        self.velocities = []
        self.orientations = []
        self.actions = []
        self.rgb_images = []
        self.depth_images = []
        self.trajectory_actual = []
        self.trajectory_desired = []

    def collect_step(
        self,
        state: dict,
        action: np.ndarray,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        timestamp: float,
        pos_desired: np.ndarray,
    ):
        """Record a single timestep of data."""
        self.timestamps.append(timestamp)
        self.positions.append(state['position'][0].cpu().numpy())
        self.velocities.append(state['velocity'][0].cpu().numpy())
        self.orientations.append(state['orientation'][0].cpu().numpy())  # xyzw
        self.actions.append(action.copy())
        self.rgb_images.append(rgb[0].cpu().numpy())
        self.depth_images.append(depth[0].cpu().numpy())
        self.trajectory_actual.append(state['position'][0].cpu().numpy().copy())
        self.trajectory_desired.append(pos_desired.copy())

    def save_sequence(self, seq_idx: int, waypoints_sim: List[np.ndarray]) -> str:
        """Save collected data to disk."""
        seq_dir = self.output_dir / f"seq_{seq_idx:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Save telemetry
        np.savez_compressed(
            seq_dir / "telemetry.npz",
            timestamps=np.array(self.timestamps),
            positions=np.array(self.positions),
            velocities=np.array(self.velocities),
            orientations=np.array(self.orientations),
            actions=np.array(self.actions),
        )

        # Save RGB images
        rgb_dir = seq_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        for i, rgb in enumerate(self.rgb_images):
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            Image.fromarray(rgb).save(rgb_dir / f"{i:06d}.png")

        # Save depth images
        depth_dir = seq_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for i, depth in enumerate(self.depth_images):
            np.save(depth_dir / f"{i:06d}.npy", depth)

        # Save trajectory plot
        if len(self.trajectory_actual) > 0 and len(self.trajectory_desired) > 0:
            try:
                save_trajectory_3d_plot(
                    trajectory_actual=self.trajectory_actual,
                    trajectory_desired=self.trajectory_desired,
                    waypoints=waypoints_sim,
                    output_path=str(seq_dir / "trajectory.png"),
                )
            except Exception:
                pass

        self.reset_buffers()
        return str(seq_dir)

    def fly_and_collect(
        self,
        sampler: BSplineTrajectorySampler,
        start_pos_sim: np.ndarray,
        seq_idx: int,
        waypoints_sim: List[np.ndarray],
        max_steps: int = 3000,
        pbar: tqdm = None,
    ) -> Tuple[str, dict]:
        """
        Fly trajectory and collect data (follows fly_trajectory from nav_helpers.py).
        """
        # Reset environment to start position (in SIMULATION space)
        self.env.reset(start_position=start_pos_sim.tolist())

        dt = 1.0 / self.sim_freq
        total_time = sampler.total_time
        stabilize_steps = 50  # Hover at start briefly

        # Get start position for stabilization
        start_pos_d, _, _, _ = sampler.sample(0.0)

        collection_count = 0
        fail_reason = None  # None = success, 'collision' or 'timeout'
        t = 0.0

        for step in range(max_steps):
            t = max(0, (step - stabilize_steps) * dt)

            if t > total_time:
                break

            # Update progress bar
            if pbar is not None:
                progress = min(t / total_time, 1.0)
                pbar.n = int(progress * 100)
                pbar.refresh()

            # Get current state (in SIMULATION space)
            state = self.env.get_state()
            pos = state['position'][0].cpu().numpy()
            vel = state['velocity'][0].cpu().numpy()
            quat_xyzw = state['orientation'][0].cpu().numpy()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            omega = state['angular_velocity'][0].cpu().numpy()

            # Sample desired trajectory (in SIMULATION space)
            if step < stabilize_steps:
                pos_d = start_pos_d
                vel_d = np.zeros(3)
                acc_d = np.zeros(3)
                yaw_d = 0.0
            else:
                pos_d, vel_d, acc_d, _ = sampler.sample(t)
                yaw_d = -sampler.get_yaw_from_velocity(vel_d, default_yaw=0.0)

            # Compute control
            thrust, omega_cmd, _ = self.controller.compute_from_quaternion(
                pos, vel, quat_wxyz, omega,
                pos_d, vel_d, acc_d, yaw_d
            )

            # Convert to env action format (same as fly_trajectory)
            rate_scale = 0.5
            action = np.array([
                np.clip(omega_cmd[0] / rate_scale, -1.0, 1.0),
                np.clip(-omega_cmd[1] / rate_scale, -1.0, 1.0),  # Negated!
                np.clip(omega_cmd[2] / rate_scale, -1.0, 1.0),
                np.clip(thrust / self.controller.max_thrust, 0.0, 1.0),
            ])

            # Step environment (renders internally)
            _, rgb, depth, done, info = self.env.step(action)

            # Collect at specified frequency
            if step % self.collection_interval == 0 and step >= stabilize_steps:
                self.collect_step(state, action, rgb, depth, t, pos_d)
                collection_count += 1

            # Check for failure
            if info.get('collision', False):
                fail_reason = 'collision'
                break
            elif done:
                fail_reason = 'timeout'
                break
        else:
            # Loop completed without break - check if trajectory finished
            if t < total_time:
                fail_reason = 'timeout'  # max_steps reached before trajectory completed

        # Complete progress bar
        if pbar is not None:
            pbar.n = 100
            pbar.refresh()

        # Save sequence
        seq_path = self.save_sequence(seq_idx, waypoints_sim)

        stats = {
            'seq_idx': seq_idx,
            'frames': collection_count,
            'duration': t,
            'fail_reason': fail_reason,  # None = success, 'collision' or 'timeout'
        }

        return seq_path, stats


def plan_and_generate_trajectory(
    map_name: str,
    v_avg: float = 0.5,
    corner_smoothing: float = 0.018,
) -> Tuple[BSplineTrajectorySampler, np.ndarray, List[np.ndarray], List, List]:
    """
    Plan A* path and generate B-spline trajectory.

    Returns:
        sampler: BSplineTrajectorySampler (trajectory in SIMULATION space)
        start_pos_sim: Start position in simulation space
        waypoints_sim: Waypoints in simulation space (for plotting)
        path_pc: A* path in point cloud space (for plotting)
        waypoints_pc: Waypoints in point cloud space (for plotting)
    """
    config = MAP_CONFIGS[map_name]

    # 1. Waypoints in POINT CLOUD space
    start_pos_pc = config["start"]
    waypoints_pc = config["waypoints"]
    destination_pc = config["destination"]

    # 2. A* path planning (in POINT CLOUD space)
    ply_file = Path(__file__).parent.parent.parent / "envs" / "assets" / "point_cloud" / "sv_1007_gate_mid.ply"
    path_pc, _ = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=0.5,
    )

    # 3. Convert to SIMULATION space
    path_sim = [np.array(p) + PC_TO_SIM_OFFSET for p in path_pc]
    start_pos_sim = np.array(start_pos_pc) + PC_TO_SIM_OFFSET

    # 4. Generate B-spline trajectory (in SIMULATION space)
    sampler = generate_bspline_trajectory_from_path(
        path=path_sim,
        v_avg=v_avg,
        corner_smoothing=corner_smoothing,
    )

    # Waypoints in simulation space for plotting
    waypoints_sim = [np.array(start_pos_pc) + PC_TO_SIM_OFFSET] + \
                    [np.array(wp) + PC_TO_SIM_OFFSET for wp in waypoints_pc] + \
                    [np.array(destination_pc) + PC_TO_SIM_OFFSET]

    # Full waypoints in PC space (start + waypoints + destination)
    full_waypoints_pc = [start_pos_pc] + waypoints_pc + [destination_pc]

    return sampler, start_pos_sim, waypoints_sim, path_pc, full_waypoints_pc


def save_planning_plots(
    seq_dir: Path,
    sampler: BSplineTrajectorySampler,
    path_pc: List,
    waypoints_pc: List,
):
    """
    Save A* path and B-spline trajectory plots before flying.

    Saves:
        - astar_bspline.png: A* path vs B-spline comparison (XY, XZ, 3D views)
        - trajectory_profile.png: Position, velocity, acceleration profiles
    """
    # Save A* vs B-spline comparison (in point cloud space)
    try:
        save_trajectory_topdown(
            sampler=sampler,
            astar_path=path_pc,
            waypoints=waypoints_pc,
            output_path=str(seq_dir / "astar_bspline.png"),
            pc_offset=PC_TO_SIM_OFFSET,  # Convert sampler (sim space) back to PC space for comparison
        )
    except Exception:
        pass

    # Save trajectory profile (position, velocity, acceleration vs time)
    try:
        save_trajectory_profile(
            sampler=sampler,
            output_path=str(seq_dir / "trajectory_profile.png"),
        )
    except Exception:
        pass


def collect_sequences(
    output_dir: str = "data/vel_net/sequences",
    map_name: str = "gate_mid",
    n_sequences: int = 30,
    collection_freq: float = 30.0,
    v_min: float = 0.5,
    v_max: float = 2.0,
    smoothing: float = 0.018,  # B-spline corner smoothing
    device: str = "cuda:0",
) -> List[dict]:
    """Collect multiple sequences for training.

    Args:
        v_min: Minimum velocity (m/s)
        v_max: Maximum velocity (m/s). If v_min == v_max, fixed velocity is used.
               Otherwise, random velocity is sampled uniformly from [v_min, v_max].
    """
    # Determine velocity mode
    vary_velocity = (v_min != v_max)

    # Header
    print(f"\n{'='*60}")
    print(f"Velocity Network Data Collection")
    print(f"{'='*60}")
    print(f"  Map: {map_name}")
    print(f"  Sequences: {n_sequences}")
    print(f"  Collection freq: {collection_freq} Hz")
    if vary_velocity:
        print(f"  Velocity: random in [{v_min:.2f}, {v_max:.2f}] m/s")
    else:
        print(f"  Velocity: fixed at {v_min:.2f} m/s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Initialize environment
    env = SimpleDroneEnv(
        map_name=map_name,
        device=device,
        num_envs=1,
        episode_length=5000,
        render_resolution=0.4,  # Lower res for faster collection
    )

    # Initialize controller (same as waypoint_nav_geometric.py)
    controller = GeometricController(
        mass=env.mass,
        gravity=9.81,
        Kp=np.array([1.5, 1.5, 0.5]),
        Kv=np.array([2.0, 2.0, 1.5]),
        Kr=np.array([3.0, 3.0, 2.0]),
        Kw=np.array([0.8, 0.8, 0.5]),
        max_thrust=env.max_thrust,
        min_thrust=2.0,
        max_rate=1.0,
    )

    # Initialize collector
    collector = DataCollector(
        env=env,
        controller=controller,
        output_dir=output_dir,
        collection_freq=collection_freq,
    )

    all_stats = []
    start_time = time.time()

    for seq_idx in range(n_sequences):
        # Sample velocity for this sequence
        if vary_velocity:
            current_v_avg = np.random.uniform(v_min, v_max)
        else:
            current_v_avg = v_min

        # Plan and generate trajectory
        sampler, start_pos_sim, waypoints_sim, path_pc, waypoints_pc = plan_and_generate_trajectory(
            map_name=map_name,
            v_avg=current_v_avg,
            corner_smoothing=smoothing,
        )

        # Create sequence directory and save planning plots BEFORE flying
        seq_dir = Path(output_dir) / f"seq_{seq_idx:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        save_planning_plots(
            seq_dir=seq_dir,
            sampler=sampler,
            path_pc=path_pc,
            waypoints_pc=waypoints_pc,
        )

        # Create progress bar for this sequence
        desc = f"[{seq_idx+1:2d}/{n_sequences}] v={current_v_avg:.2f}m/s"
        pbar = tqdm(total=100, desc=desc, bar_format='{l_bar}{bar:20}{r_bar}', leave=True)

        # Fly and collect
        seq_path, stats = collector.fly_and_collect(
            sampler=sampler,
            start_pos_sim=start_pos_sim,
            seq_idx=seq_idx,
            waypoints_sim=waypoints_sim,
            pbar=pbar,
        )

        # Update progress bar with result
        if stats['fail_reason'] is None:
            status = "OK"
        else:
            status = stats['fail_reason'].upper()  # COLLISION or TIMEOUT
        pbar.set_postfix_str(f"{status} | {stats['frames']} frames | {stats['duration']:.1f}s")
        pbar.close()

        stats['v_avg'] = current_v_avg
        stats['path'] = seq_path
        all_stats.append(stats)

    # Summary
    total_time = time.time() - start_time
    total_frames = sum(s['frames'] for s in all_stats)
    n_collisions = sum(1 for s in all_stats if s['fail_reason'] == 'collision')
    n_timeouts = sum(1 for s in all_stats if s['fail_reason'] == 'timeout')
    n_success = len(all_stats) - n_collisions - n_timeouts

    print(f"\n{'='*60}")
    print(f"Collection Complete!")
    print(f"{'='*60}")
    print(f"  Sequences: {len(all_stats)} ({n_success} OK, {n_collisions} collision, {n_timeouts} timeout)")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg frames/seq: {total_frames / len(all_stats):.0f}")
    print(f"{'='*60}\n")

    env.close()
    return all_stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect vel_net training data')
    parser.add_argument('--output', type=str, default='data/vel_net/sequences')
    parser.add_argument('--map', type=str, default='gate_mid',
                        choices=['gate_mid', 'gate_left', 'gate_right'])
    parser.add_argument('--n_sequences', type=int, default=30)
    parser.add_argument('--freq', type=float, default=30.0)
    parser.add_argument('--v_min', type=float, default=0.5,
                        help='Min velocity (m/s)')
    parser.add_argument('--v_max', type=float, default=2.0,
                        help='Max velocity (m/s). If v_min != v_max, random velocity per sequence.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--smoothing', type=float, default=0.018)

    args = parser.parse_args()

    collect_sequences(
        output_dir=args.output,
        map_name=args.map,
        n_sequences=args.n_sequences,
        collection_freq=args.freq,
        v_min=args.v_min,
        v_max=args.v_max,
        smoothing=args.smoothing,
        device=args.device,
    )
