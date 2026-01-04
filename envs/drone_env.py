"""
Simplified Drone Environment for Waypoint Navigation

A lightweight environment for drone simulation with:
- 3D Gaussian Splatting rendering
- Quadrotor physics simulation
- State management (position, velocity, orientation)

No ML components, dynamic objects, or policy networks.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gs_local import GS, get_gs
from utils.point_cloud_util import ObstacleDistanceCalculator
from utils.rotation import quaternion_to_euler
from envs.assets.quadrotor_dynamics import QuadrotorSimulator


class SimpleDroneEnv:
    """
    Simplified drone environment for waypoint navigation using PD control.

    This environment provides:
    - GS-based scene rendering
    - Quadrotor physics simulation
    - Collision detection via point cloud
    - State access for position, velocity, orientation
    """

    def __init__(
        self,
        map_name: str = 'gate_mid',
        device: str = 'cuda:0',
        num_envs: int = 1,
        episode_length: int = 2000,
        render_resolution: float = 0.4,
    ):
        """
        Initialize the drone environment.

        Args:
            map_name: Name of the map to load (gate_mid, gate_left, gate_right, etc.)
            device: PyTorch device to use
            num_envs: Number of parallel environments (typically 1 for waypoint nav)
            episode_length: Maximum steps per episode
            render_resolution: GS rendering resolution quality (0.0-1.0)
        """
        self.map_name = map_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.render_resolution = render_resolution

        # Map configurations
        self.map_configs = {
            "gate_mid": {
                "gs_folder": "sv_1007_gate_mid",
                "ply_file": "sv_1007_gate_mid.ply",
                "start_pos": [-6.0, 0.0, 1.25],
                "target_pos": [7.0, -2.0, 1.3],
                "waypoints": [[-0.2, -0.1, 1.4], [3.7, 1.5, 0.7]],
            },
            "gate_left": {
                "gs_folder": "sv_917_3_left_nerfstudio",
                "ply_file": "sv_917_3_left_nerfstudio.ply",
                "start_pos": [-6.0, 0.0, 1.2],
                "target_pos": [7.0, -2.0, 1.2],
                "waypoints": [[-0.2, 1.2, 1.4], [3.7, 1.2, 0.6]],
            },
            "gate_right": {
                "gs_folder": "sv_917_3_right_nerfstudio",
                "ply_file": "sv_917_3_right_nerfstudio.ply",
                "start_pos": [-6.0, 0.0, 1.3],
                "target_pos": [7.0, -2.0, 1.3],
                "waypoints": [[0.0, -1.4, 1.5], [3.7, 1.4, 0.7]],
            },
            "simple_hover": {
                "gs_folder": "sv_917_3_right_nerfstudio",
                "ply_file": "sv_917_3_right_nerfstudio.ply",
                "start_pos": [-6.0, 0.0, 1.2],
                "target_pos": [-6.0, 0.0, 1.2],
                "waypoints": [],
            },
        }

        # Simulation parameters
        self.sim_freq = 50.0  # Hz
        self.sim_dt = 1.0 / self.sim_freq
        self.control_freq = 50.0  # Hz

        # Drone physical parameters
        self.mass = 1.15
        self.max_thrust = 25.0
        self.hover_thrust = self.mass * 9.81  # Force needed to hover

        # GS origin offset (simulation space -> GS rendering space)
        # Simulation: x from 0 to +13, GS: x from -6 to +7
        self.gs_origin_offset = torch.tensor([[-6.0, 0.0, -0.05]], device=self.device).repeat(num_envs, 1)

        # Point cloud offset for collision detection (sim space -> point cloud space)
        # Same as GS offset (both point cloud and GS use same coordinate system)
        self.point_cloud_offset = torch.tensor([[-6.0, 0.0, 0.0]], device=self.device).repeat(num_envs, 1)

        # Initialize components
        self._init_gs_renderer()
        self._init_point_cloud()
        self._init_quadrotor_dynamics()
        self._init_state()

        print(f"[SimpleDroneEnv] Initialized on {self.device}")
        print(f"  Map: {map_name}")
        print(f"  Num envs: {num_envs}")
        print(f"  Render resolution: {render_resolution}")

    def _init_gs_renderer(self):
        """Initialize the Gaussian Splatting renderer."""
        gs_path = Path(__file__).parent / "assets" / "gs_data"
        self.gs = get_gs(self.map_name, gs_path, self.render_resolution)
        print(f"  GS renderer initialized")

    def _init_point_cloud(self):
        """Initialize the point cloud for collision detection."""
        point_cloud_dir = Path(__file__).parent / "assets" / "point_cloud"
        ply_file = point_cloud_dir / self.map_configs[self.map_name]["ply_file"]
        self.point_cloud = ObstacleDistanceCalculator(ply_file=ply_file, device=self.device)
        print(f"  Point cloud loaded from {ply_file}")

    def _init_quadrotor_dynamics(self):
        """Initialize the quadrotor physics simulator."""
        self.quad_dynamics = QuadrotorSimulator(
            mass=torch.tensor([self.mass] * self.num_envs),
            inertia=torch.diag_embed(torch.tensor([[0.01, 0.012, 0.025]] * self.num_envs)),
            link_length=0.15,
            Kp=torch.tensor([[1.0, 1.2, 2.5]] * self.num_envs),
            Kd=torch.tensor([[0.001, 0.001, 0.002]] * self.num_envs),
            freq=200.0,
            max_thrust=torch.tensor([self.max_thrust] * self.num_envs),
            total_time=0.02,  # 20ms per step
            rotor_noise_std=0.01,
            br_noise_std=0.01,
            device=self.device
        )
        print(f"  Quadrotor dynamics initialized")

    def _init_state(self):
        """Initialize the drone state tensors."""
        # Position [x, y, z]
        self.position = torch.zeros((self.num_envs, 3), device=self.device)
        # Velocity [vx, vy, vz]
        self.velocity = torch.zeros((self.num_envs, 3), device=self.device)
        # Orientation quaternion [x, y, z, w]
        self.orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.orientation[:, 3] = 1.0  # Identity quaternion
        # Angular velocity [wx, wy, wz]
        self.angular_velocity = torch.zeros((self.num_envs, 3), device=self.device)

        # Last rendered images
        self.last_rgb = None
        self.last_depth = None

        # Step counter
        self.step_count = 0

        # Collision flag
        self.collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, start_position=None):
        """
        Reset the environment to initial state.

        Args:
            start_position: Optional starting position [x, y, z]. If None, uses map default.

        Returns:
            Initial state dict
        """
        # Reset state
        self._init_state()

        # Set starting position
        if start_position is not None:
            self.position[:] = torch.tensor(start_position, device=self.device)
        else:
            default_start = self.map_configs[self.map_name]["start_pos"]
            self.position[:] = torch.tensor(default_start, device=self.device)

        # Render initial view
        self._render()

        self.step_count = 0

        return self.get_state()

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: Control action [roll_rate, pitch_rate, yaw_rate, thrust]
                   All values should be in [-1, 1] range except thrust in [0, 1]

        Returns:
            state: Current state dict
            rgb: RGB image from drone camera
            depth: Depth image from drone camera
            done: Whether episode is done
            info: Additional info dict
        """
        # Ensure action is a tensor
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Parse action: [roll_rate, pitch_rate, yaw_rate, thrust]
        # Scale body rates (input is [-1, 1], output is [-0.5, 0.5] rad/s)
        body_rates = torch.clip(action[:, 0:3], -1.0, 1.0) * 0.5

        # Scale thrust (input is [0, 1], output is normalized thrust)
        thrust = torch.clip(action[:, 3:], 0.0, 1.0)

        # Prepare control input for dynamics
        control_input = (body_rates, thrust.squeeze(-1))

        # Run physics simulation
        # Get current state in dynamics format (quaternion is [w, x, y, z])
        orientation_wxyz = self.orientation[:, [3, 0, 1, 2]]

        # Run simulation
        new_position, new_velocity, new_angular_velocity, new_orientation, lin_acc, ang_acc = \
            self.quad_dynamics.run_simulation(
                position=self.position,
                velocity=self.velocity,
                orientation=orientation_wxyz,
                angular_velocity=self.angular_velocity,
                control_input=control_input
            )

        # Update state (convert quaternion back to [x, y, z, w])
        self.position = new_position
        self.velocity = new_velocity
        self.angular_velocity = new_angular_velocity
        self.orientation = new_orientation[:, [1, 2, 3, 0]]

        # Check for collisions
        self._check_collision()

        # Render new view
        self._render()

        self.step_count += 1

        # Check if done
        done = self.step_count >= self.episode_length or self.collision.any()

        # Build info dict
        info = {
            'step': self.step_count,
            'collision': self.collision.item() if self.num_envs == 1 else self.collision,
        }

        return self.get_state(), self.last_rgb, self.last_depth, done, info

    def _render(self):
        """Render the scene from the drone's perspective."""
        # Build GS pose: [x, y, z, 0, 0, 0, qx, qy, qz, qw]
        gs_pos = self.position + self.gs_origin_offset
        gs_pos[:, 1] = -gs_pos[:, 1]  # Flip y
        gs_pos[:, 2] = -gs_pos[:, 2]  # Flip z

        gs_pose = torch.zeros((self.num_envs, 10), device=self.device)
        gs_pose[:, 0:3] = gs_pos
        gs_pose[:, 6:10] = self.orientation  # [qx, qy, qz, qw]

        # Render
        depth, rgb = self.gs.render(gs_pose)

        self.last_rgb = rgb
        self.last_depth = depth

    def _check_collision(self):
        """Check for collisions with obstacles."""
        # Get quaternion in (w, x, y, z) format for point cloud util
        quat_wxyz = self.orientation[:, [3, 0, 1, 2]]

        # Transform position from simulation space to point cloud space
        pc_pos = self.position + self.point_cloud_offset

        # Compute nearest distance to obstacles
        distances = self.point_cloud.compute_nearest_distances(pc_pos, quat_wxyz)

        # Check collision (distance < threshold)
        collision_threshold = 0.15
        self.collision = distances < collision_threshold

    def get_state(self):
        """
        Get current drone state.

        Returns:
            Dict with position, velocity, orientation, angular_velocity
        """
        return {
            'position': self.position.clone(),
            'velocity': self.velocity.clone(),
            'orientation': self.orientation.clone(),
            'angular_velocity': self.angular_velocity.clone(),
            'rpy': quaternion_to_euler(self.orientation[:, [3, 0, 1, 2]]),  # Convert to (w,x,z,y) format
        }

    def get_position(self):
        """Get current position as numpy array."""
        return self.position[0].cpu().numpy()

    def get_velocity(self):
        """Get current velocity as numpy array."""
        return self.velocity[0].cpu().numpy()

    def get_hover_thrust(self):
        """Get normalized thrust required for hovering."""
        return self.hover_thrust / self.max_thrust

    def get_default_waypoints(self):
        """Get default waypoints for the current map."""
        return self.map_configs[self.map_name]["waypoints"]

    def get_target_position(self):
        """Get target/destination position for the current map."""
        return self.map_configs[self.map_name]["target_pos"]

    def close(self):
        """Clean up resources."""
        pass
