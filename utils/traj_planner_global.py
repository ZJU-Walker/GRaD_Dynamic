"""
Global Trajectory Planner using A* Algorithm

Provides collision-free path planning on point cloud occupancy grids.
"""

import numpy as np
import open3d as o3d
import heapq
from scipy.ndimage import binary_dilation
import torch
import time


class TrajectoryPlanner:
    def __init__(self, ply_file, voxel_size=0.1, safety_distance=0.2, batch_size=1, wp_distance=2.0, verbose=True):
        """
        Initialize the trajectory planner.

        Args:
            ply_file: Path to point cloud PLY file
            voxel_size: Size of voxels for occupancy grid
            safety_distance: Safety margin around obstacles
            batch_size: Number of trajectories to plan
            wp_distance: Distance between resampled waypoints
            verbose: Print debug info
        """
        self.verbose = verbose
        self.ply_file = ply_file
        self.voxel_size = voxel_size
        self.safety_distance = safety_distance
        self.batch_size = batch_size
        self.points = self.load_point_cloud()
        self.occupancy_grid, self.min_bound = self.create_occupancy_grid()
        self.trajectory_batches = []
        self.waypoints_list = None
        self.destination_positions = None
        self.wp_distance = wp_distance

    def load_point_cloud(self):
        """Load point cloud data from PLY file."""
        pcd = o3d.io.read_point_cloud(str(self.ply_file))
        if self.verbose:
            print("Point cloud loaded with {} points.".format(len(pcd.points)))
        return np.asarray(pcd.points)

    def create_occupancy_grid(self):
        """Create binary occupancy grid from point cloud."""
        min_bound = self.points.min(axis=0) - self.safety_distance
        max_bound = self.points.max(axis=0) + self.safety_distance
        grid_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
        occupancy_grid = np.zeros(grid_shape, dtype=bool)

        indices = np.floor((self.points - min_bound) / self.voxel_size).astype(int)
        occupancy_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Inflate obstacles by safety distance
        occupancy_grid = binary_dilation(
            occupancy_grid, iterations=int(self.safety_distance / self.voxel_size)
        )
        if self.verbose:
            print("Occupancy grid created with shape {}.".format(occupancy_grid.shape))
        return occupancy_grid, min_bound

    def heuristic(self, a, b):
        """Euclidean distance heuristic for A*."""
        return np.linalg.norm(a - b)

    def astar(self, start, goal):
        """
        A* pathfinding algorithm.

        Args:
            start: Start position (x, y, z)
            goal: Goal position (x, y, z)

        Returns:
            List of path points, or None if no path found
        """
        start_idx = np.floor((start - self.min_bound) / self.voxel_size).astype(int)
        goal_idx = np.floor((goal - self.min_bound) / self.voxel_size).astype(int)

        grid_shape = self.occupancy_grid.shape
        visited = np.full(grid_shape, False, dtype=bool)
        came_from = {}
        g_score = np.full(grid_shape, np.inf)
        g_score[tuple(start_idx)] = 0
        f_score = np.full(grid_shape, np.inf)
        f_score[tuple(start_idx)] = self.heuristic(start_idx, goal_idx)

        open_set = []
        heapq.heappush(open_set, (f_score[tuple(start_idx)], tuple(start_idx)))

        neighbors = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]

        while open_set:
            current = heapq.heappop(open_set)[1]
            if np.array_equal(current, goal_idx):
                path = []
                while current in came_from:
                    path.append(
                        np.array(current) * self.voxel_size + self.min_bound
                    )
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            visited[current] = True
            for neighbor in neighbors:
                neighbor_idx = np.array(current) + neighbor
                if (
                    0 <= neighbor_idx[0] < grid_shape[0]
                    and 0 <= neighbor_idx[1] < grid_shape[1]
                    and 0 <= neighbor_idx[2] < grid_shape[2]
                ):
                    if (
                        self.occupancy_grid[tuple(neighbor_idx)]
                        or visited[tuple(neighbor_idx)]
                    ):
                        continue
                    tentative_g_score = g_score[current] + self.heuristic(
                        np.array(current), neighbor_idx
                    )
                    if tentative_g_score < g_score[tuple(neighbor_idx)]:
                        came_from[tuple(neighbor_idx)] = current
                        g_score[tuple(neighbor_idx)] = tentative_g_score
                        f_score[tuple(neighbor_idx)] = tentative_g_score + self.heuristic(
                            neighbor_idx, goal_idx
                        )
                        heapq.heappush(
                            open_set, (f_score[tuple(neighbor_idx)], tuple(neighbor_idx))
                        )
        return None

    def interpolate_z_between_waypoints(self, path, waypoints):
        """
        Interpolate Z values linearly between waypoints based on XY distance progress.

        This fixes the voxel-snapping issue where Z drops to grid centers.
        Keeps A* XY path for obstacle avoidance, but smooths Z between exact waypoints.

        Args:
            path: Full A* path as numpy array (N, 3)
            waypoints: List of waypoints with exact Z values (includes start and destination)

        Returns:
            Path with interpolated Z values
        """
        path = np.array(path)
        waypoints = np.array(waypoints)

        if len(path) < 2 or len(waypoints) < 2:
            return path

        # Find which waypoint each path point is closest to (to determine segment)
        # Map each path point to a segment between consecutive waypoints

        # First, find the path indices closest to each waypoint
        waypoint_path_indices = []
        for wp in waypoints:
            distances = np.linalg.norm(path[:, :2] - wp[:2], axis=1)  # XY distance only
            idx = np.argmin(distances)
            waypoint_path_indices.append(idx)

        # Ensure indices are sorted and unique
        waypoint_path_indices = sorted(set(waypoint_path_indices))

        # Make sure first and last path points are included
        if waypoint_path_indices[0] != 0:
            waypoint_path_indices.insert(0, 0)
        if waypoint_path_indices[-1] != len(path) - 1:
            waypoint_path_indices.append(len(path) - 1)

        # Get the exact Z values at waypoint positions
        waypoint_z_values = []
        for idx in waypoint_path_indices:
            # Find the closest original waypoint to get exact Z
            distances = np.linalg.norm(waypoints[:, :2] - path[idx, :2], axis=1)
            closest_wp_idx = np.argmin(distances)
            waypoint_z_values.append(waypoints[closest_wp_idx, 2])

        # Interpolate Z for each segment
        interpolated_path = path.copy()

        for seg_idx in range(len(waypoint_path_indices) - 1):
            start_path_idx = waypoint_path_indices[seg_idx]
            end_path_idx = waypoint_path_indices[seg_idx + 1]

            start_z = waypoint_z_values[seg_idx]
            end_z = waypoint_z_values[seg_idx + 1]

            # Get segment points
            segment = path[start_path_idx:end_path_idx + 1]

            if len(segment) < 2:
                continue

            # Calculate cumulative XY distance along segment
            xy_diffs = np.diff(segment[:, :2], axis=0)
            xy_distances = np.linalg.norm(xy_diffs, axis=1)
            cumulative_dist = np.insert(np.cumsum(xy_distances), 0, 0)
            total_dist = cumulative_dist[-1]

            if total_dist > 0:
                # Normalize to [0, 1] progress
                progress = cumulative_dist / total_dist
                # Linear interpolation of Z
                interpolated_z = start_z + progress * (end_z - start_z)
                interpolated_path[start_path_idx:end_path_idx + 1, 2] = interpolated_z
            else:
                # No XY movement, just use start Z
                interpolated_path[start_path_idx:end_path_idx + 1, 2] = start_z

        return interpolated_path

    def resample_trajectory_with_waypoints(self, full_path, waypoints, step_size=2.0):
        """
        Resample trajectory to ensure waypoints are included.

        Args:
            full_path: Original path
            waypoints: List of waypoints to include
            step_size: Distance between resampled points

        Returns:
            Resampled trajectory as numpy array
        """
        trajectory_points = np.array(full_path)
        resampled_trajectory = []

        waypoint_indices = []
        # Find indices of waypoints in the trajectory
        for wp in waypoints:
            distances = np.linalg.norm(trajectory_points - wp, axis=1)
            idx = np.argmin(distances)
            waypoint_indices.append(idx)

        # Sort waypoints indices to ensure order
        waypoint_indices = sorted(set(waypoint_indices))

        # Resample between waypoints
        for i in range(len(waypoint_indices) - 1):
            start_idx = waypoint_indices[i]
            end_idx = waypoint_indices[i + 1]

            segment = trajectory_points[start_idx:end_idx + 1]

            # Compute cumulative distances along the segment
            segment_lengths = np.linalg.norm(np.diff(segment, axis=0), axis=1)
            cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
            total_length = cumulative_lengths[-1]
            num_samples = max(int(np.ceil(total_length / step_size)), 1) + 1
            sample_distances = np.linspace(0, total_length, num_samples)

            # Interpolate the segment at the sample distances
            resampled_segment = np.zeros((num_samples, 3))
            for j in range(3):
                resampled_segment[:, j] = np.interp(
                    sample_distances, cumulative_lengths, segment[:, j]
                )

            # Remove the last point to avoid duplicates except for the final segment
            if i < len(waypoint_indices) - 2:
                resampled_segment = resampled_segment[:-1]

            resampled_trajectory.append(resampled_segment)

        # Concatenate all resampled segments
        resampled_trajectory = np.vstack(resampled_trajectory)
        return resampled_trajectory

    def ensure_increasing_x_with_waypoints(self, trajectory_points, waypoints):
        """Ensure trajectory has only increasing x values while preserving waypoints."""
        x = trajectory_points[:, 0]
        dx = np.diff(x)
        # Identify indices where x is not decreasing
        valid_indices = np.where(dx >= 0)[0] + 1
        valid_indices = np.insert(valid_indices, 0, 0)

        # Ensure waypoints are included
        waypoint_indices = []
        for wp in waypoints:
            distances = np.linalg.norm(trajectory_points - wp, axis=1)
            idx = np.argmin(distances)
            waypoint_indices.append(idx)
        waypoint_indices = set(waypoint_indices)

        # Combine valid indices and waypoint indices
        combined_indices = sorted(set(valid_indices) | waypoint_indices)

        # Filter the trajectory
        filtered_trajectory = trajectory_points[combined_indices]
        return filtered_trajectory

    def plan_single_trajectory(self, idx, current_pos, destination_pos, waypoints):
        """
        Plan a single trajectory from current position to destination via waypoints.

        Args:
            idx: Trajectory index
            current_pos: Start position
            destination_pos: Goal position
            waypoints: Intermediate waypoints

        Returns:
            Resampled trajectory as list, or None if planning failed
        """
        if self.verbose:
            print(f"Planning trajectory {idx+1}/{self.batch_size}...")
        full_path = []
        waypoints = [current_pos] + list(waypoints) + [destination_pos]
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            if self.verbose:
                print(f"Trajectory {idx+1}: Path from {start} to {goal}...")
            path = self.astar(start, goal)
            if path is None:
                if self.verbose:
                    print(f"Trajectory {idx+1}: Path from {start} to {goal} not found.")
                return None
            if full_path and np.array_equal(full_path[-1], path[0]):
                full_path.extend(path[1:])
            else:
                full_path.extend(path)
        if self.verbose:
            print(f"Trajectory {idx+1} planning completed.")

        # Convert the full path to a NumPy array
        trajectory_points = np.array(full_path)

        # Interpolate Z values between waypoints to fix voxel-snapping artifacts
        # This keeps A* XY path for obstacle avoidance but gives smooth/exact Z values
        trajectory_points = self.interpolate_z_between_waypoints(trajectory_points, waypoints)
        full_path = trajectory_points.tolist()

        if self.verbose:
            print(f"Trajectory {idx+1}: Z values interpolated between waypoints.")

        # Resample the trajectory between waypoints to ensure waypoints are included
        resampled_trajectory = self.resample_trajectory_with_waypoints(full_path, waypoints, step_size=self.wp_distance)

        # Ensure that x is only increasing, but preserve waypoints
        resampled_trajectory = self.ensure_increasing_x_with_waypoints(resampled_trajectory, waypoints)

        return resampled_trajectory.tolist()

    def plan_trajectories(self, current_positions, destination_positions, waypoints_list):
        """
        Plan trajectories for multiple drones.

        Args:
            current_positions: Current positions [B, 3] (tensor or numpy)
            destination_positions: Destination positions [B, 3] (tensor or numpy)
            waypoints_list: List of waypoint arrays for each trajectory

        Returns:
            List of trajectories (each trajectory is a list of [x, y, z] points)
        """
        # Handle data type conversion inside the method
        if isinstance(current_positions, torch.Tensor):
            current_positions = current_positions.clone().detach().cpu().numpy()
        if isinstance(destination_positions, torch.Tensor):
            destination_positions = destination_positions.clone().detach().cpu().numpy()
        if isinstance(waypoints_list, list):
            converted_waypoints_list = []
            for waypoints in waypoints_list:
                if isinstance(waypoints, torch.Tensor):
                    waypoints = waypoints.clone().detach().cpu().numpy()
                converted_waypoints_list.append(waypoints)
            self.waypoints_list = converted_waypoints_list
        else:
            raise ValueError("waypoints_list must be a list of torch.tensor or numpy arrays.")

        # Ensure current_positions and destination_positions are NumPy arrays
        current_positions = np.asarray(current_positions)
        destination_positions = np.asarray(destination_positions)

        if not (len(current_positions) == len(destination_positions) == len(self.waypoints_list) == self.batch_size):
            raise ValueError("Input lists must have the same length as batch_size.")

        # Store destination_positions for visualization
        self.destination_positions = destination_positions

        self.trajectory_batches = [None] * self.batch_size

        # Sequential implementation
        for i in range(self.batch_size):
            current_pos = current_positions[i]
            destination_pos = destination_positions[i]
            waypoints = self.waypoints_list[i]

            try:
                result = self.plan_single_trajectory(i, current_pos, destination_pos, waypoints)
                self.trajectory_batches[i] = result
            except Exception as e:
                print(f'Trajectory for env {i} planning failed: {e}')

        return self.trajectory_batches

    def visualize_trajectories(self):
        """Visualize planned trajectories with Open3D."""
        if not self.trajectory_batches:
            if self.verbose:
                print("No trajectories to visualize. Please run plan_trajectories first.")
            return

        def create_tube(trajectory_points, radius, color):
            """Creates a tube (cylinder segments) connecting trajectory points."""
            tube_mesh = o3d.geometry.TriangleMesh()
            for i in range(len(trajectory_points) - 1):
                start = trajectory_points[i]
                end = trajectory_points[i + 1]
                direction = end - start
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                direction /= length

                # Create cylinder
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

                # Calculate rotation to align with direction
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                rotation_angle = np.arccos(np.dot(z_axis, direction))
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis /= np.linalg.norm(rotation_axis)
                    cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle), center=(0, 0, 0))

                # Translate cylinder to the middle point between start and end
                cylinder.translate((start + end) / 2)
                cylinder.paint_uniform_color(color)
                tube_mesh += cylinder
            return tube_mesh

        # Create geometries list
        geometries = []

        # Add PointCloud (obstacles)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        geometries.append(pcd)

        # Define colors for trajectories
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]

        # Add trajectories as tubes
        for idx, trajectory in enumerate(self.trajectory_batches):
            if trajectory is None or len(trajectory) == 0:
                if self.verbose:
                    print(f"Trajectory {idx+1} is empty or not found.")
                continue

            trajectory_points = np.array(trajectory, dtype=np.float64)
            if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
                if self.verbose:
                    print(f"Trajectory {idx+1} has incorrect shape: {trajectory_points.shape}")
                continue

            color = colors[idx % len(colors)]
            tube_mesh = create_tube(trajectory_points, radius=0.025, color=colors[1])
            geometries.append(tube_mesh)

        # Add waypoints
        if self.waypoints_list:
            for idx, waypoints in enumerate(self.waypoints_list):
                for waypoint in waypoints:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                    sphere.translate(waypoint)
                    sphere.paint_uniform_color(colors[2])
                    geometries.append(sphere)

        # Add destination positions
        if self.destination_positions is not None:
            for idx, dest in enumerate(self.destination_positions):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
                sphere.translate(dest)
                sphere.paint_uniform_color(colors[0])
                geometries.append(sphere)

        # Visualize all geometries
        o3d.visualization.draw_geometries(geometries)
