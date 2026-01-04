"""
Simple PD Position Controller

Provides proportional-derivative control for drone position tracking.
Converts position error to body rate commands and thrust.
"""

import torch
import numpy as np


class SimplePDController:
    """
    Simple PD controller for position tracking.

    Converts position error to control actions:
    - Roll rate: Based on Y position error
    - Pitch rate: Based on X position error
    - Yaw rate: Based on heading error (optional)
    - Thrust: Based on Z position error + hover compensation
    """

    def __init__(
        self,
        hover_thrust: float = 0.45,
        kp_x: float = 0.3,
        kp_y: float = 0.3,
        kp_z: float = 0.2,
        kd_x: float = 0.1,
        kd_y: float = 0.1,
        kd_z: float = 0.1,
        rate_limit: float = 0.2,
        thrust_min: float = 0.3,
        thrust_max: float = 0.7,
        device: str = 'cuda:0',
    ):
        """
        Initialize the PD controller.

        Args:
            hover_thrust: Normalized thrust required for hovering (0-1)
            kp_x: Proportional gain for X position (pitch)
            kp_y: Proportional gain for Y position (roll)
            kp_z: Proportional gain for Z position (thrust)
            kd_x: Derivative gain for X velocity
            kd_y: Derivative gain for Y velocity
            kd_z: Derivative gain for Z velocity
            rate_limit: Maximum body rate command magnitude
            thrust_min: Minimum thrust command
            thrust_max: Maximum thrust command
            device: PyTorch device
        """
        self.hover_thrust = hover_thrust
        self.kp_x = kp_x
        self.kp_y = kp_y
        self.kp_z = kp_z
        self.kd_x = kd_x
        self.kd_y = kd_y
        self.kd_z = kd_z
        self.rate_limit = rate_limit
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Previous position for derivative term
        self.prev_pos_error = None

    def compute_action(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        current_vel: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute control action to move from current position to target.

        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            current_vel: Current velocity [vx, vy, vz] (optional, for D term)

        Returns:
            action: Control action [roll_rate, pitch_rate, yaw_rate, thrust]
                   In ranges: rates in [-1, 1], thrust in [0, 1]
        """
        # Ensure numpy arrays
        current_pos = np.asarray(current_pos, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)

        # Compute position error
        pos_error = target_pos - current_pos

        # Compute velocity (derivative term)
        if current_vel is not None:
            vel = np.asarray(current_vel, dtype=np.float32)
        elif self.prev_pos_error is not None:
            vel = (pos_error - self.prev_pos_error) * 50.0  # Approx velocity from error change
        else:
            vel = np.zeros(3, dtype=np.float32)

        self.prev_pos_error = pos_error.copy()

        # Compute roll rate (controls Y motion)
        roll_rate = -self.kp_y * pos_error[1] + self.kd_y * vel[1]
        roll_rate = np.clip(roll_rate, -self.rate_limit, self.rate_limit)

        # Compute pitch rate (controls X motion)
        pitch_rate = -self.kp_x * pos_error[0] + self.kd_x * vel[0]
        pitch_rate = np.clip(pitch_rate, -self.rate_limit, self.rate_limit)

        # Yaw rate (keep at zero for now - heading control could be added)
        yaw_rate = 0.0

        # Compute thrust (controls Z motion)
        # Positive Z error -> need more thrust
        # D term opposes velocity for damping (moving up -> reduce thrust)
        thrust = self.hover_thrust + self.kp_z * pos_error[2] - self.kd_z * vel[2]
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)

        # Build action vector
        action = np.array([roll_rate, pitch_rate, yaw_rate, thrust], dtype=np.float32)

        return action

    def compute_action_batch(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        current_vel: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute control actions for a batch of drones.

        Args:
            current_pos: Current positions [B, 3]
            target_pos: Target positions [B, 3]
            current_vel: Current velocities [B, 3] (optional)

        Returns:
            actions: Control actions [B, 4]
        """
        # Compute position error
        pos_error = target_pos - current_pos

        # Velocity term
        if current_vel is None:
            current_vel = torch.zeros_like(current_pos)

        # Roll rate (controls Y motion)
        roll_rate = -self.kp_y * pos_error[:, 1] + self.kd_y * current_vel[:, 1]
        roll_rate = torch.clip(roll_rate, -self.rate_limit, self.rate_limit)

        # Pitch rate (controls X motion)
        pitch_rate = -self.kp_x * pos_error[:, 0] + self.kd_x * current_vel[:, 0]
        pitch_rate = torch.clip(pitch_rate, -self.rate_limit, self.rate_limit)

        # Yaw rate
        yaw_rate = torch.zeros_like(roll_rate)

        # Thrust (D term opposes velocity for damping)
        thrust = self.hover_thrust + self.kp_z * pos_error[:, 2] - self.kd_z * current_vel[:, 2]
        thrust = torch.clip(thrust, self.thrust_min, self.thrust_max)

        # Stack actions
        actions = torch.stack([roll_rate, pitch_rate, yaw_rate, thrust], dim=1)

        return actions

    def reset(self):
        """Reset controller state."""
        self.prev_pos_error = None


class WaypointFollower:
    """
    Waypoint following controller.

    Manages progression through a list of waypoints using the PD controller.
    """

    def __init__(
        self,
        controller: SimplePDController,
        waypoint_threshold: float = 0.5,
    ):
        """
        Initialize the waypoint follower.

        Args:
            controller: SimplePDController instance
            waypoint_threshold: Distance to waypoint to consider it reached
        """
        self.controller = controller
        self.waypoint_threshold = waypoint_threshold
        self.waypoints = []
        self.current_waypoint_idx = 0

    def set_waypoints(self, waypoints: list):
        """
        Set the list of waypoints to follow.

        Args:
            waypoints: List of [x, y, z] positions
        """
        self.waypoints = [np.asarray(wp, dtype=np.float32) for wp in waypoints]
        self.current_waypoint_idx = 0
        self.controller.reset()

    def get_current_target(self) -> np.ndarray:
        """Get the current target waypoint."""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return self.waypoints[-1] if self.waypoints else None

    def compute_action(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray = None,
    ) -> tuple:
        """
        Compute control action and check waypoint progress.

        Args:
            current_pos: Current position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]

        Returns:
            action: Control action [roll_rate, pitch_rate, yaw_rate, thrust]
            reached_goal: True if final waypoint is reached
        """
        if not self.waypoints:
            return np.array([0.0, 0.0, 0.0, 0.45]), True

        target = self.get_current_target()
        current_pos = np.asarray(current_pos, dtype=np.float32)

        # Check if current waypoint is reached
        distance = np.linalg.norm(target - current_pos)
        if distance < self.waypoint_threshold:
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
                target = self.get_current_target()
                print(f"  Reached waypoint {self.current_waypoint_idx}, moving to next: {target}")

        # Compute action
        action = self.controller.compute_action(current_pos, target, current_vel)

        # Check if goal reached
        reached_goal = (
            self.current_waypoint_idx == len(self.waypoints) - 1
            and distance < self.waypoint_threshold
        )

        return action, reached_goal

    def get_progress(self) -> tuple:
        """
        Get progress information.

        Returns:
            current_idx: Current waypoint index
            total: Total number of waypoints
        """
        return self.current_waypoint_idx, len(self.waypoints)
