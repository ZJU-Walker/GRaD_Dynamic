"""
Geometric Tracking Controller on SE(3)

Implements the nonlinear geometric controller from Lee et al. for quadrotor tracking.
Uses feedforward acceleration for improved trajectory tracking.

Reference: Lee, Leok, McClamroch - "Geometric Tracking Control of a Quadrotor UAV on SE(3)"
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


class GeometricController:
    """
    Geometric Tracking Controller on SE(3).

    Computes thrust and body rates to track desired position, velocity,
    and acceleration trajectories.
    """

    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        Kp: np.ndarray = None,
        Kv: np.ndarray = None,
        Kr: np.ndarray = None,
        Kw: np.ndarray = None,
        max_thrust: float = 20.0,
        min_thrust: float = 0.5,
        max_tilt: float = np.pi / 4,  # 45 degrees
        max_rate: float = 3.0,  # rad/s
    ):
        """
        Initialize the geometric controller.

        Args:
            mass: Quadrotor mass (kg)
            gravity: Gravitational acceleration (m/s^2)
            Kp: Position gain matrix (3,) or scalar
            Kv: Velocity gain matrix (3,) or scalar
            Kr: Attitude gain matrix (3,) or scalar
            Kw: Angular velocity gain matrix (3,) or scalar
            max_thrust: Maximum thrust (N)
            min_thrust: Minimum thrust (N)
            max_tilt: Maximum tilt angle (rad)
            max_rate: Maximum body rate (rad/s)
        """
        self.m = mass
        self.g = gravity

        # Default gains (tuned for typical quadrotor)
        self.Kp = np.array([6.0, 6.0, 8.0]) if Kp is None else np.asarray(Kp)
        self.Kv = np.array([4.0, 4.0, 5.0]) if Kv is None else np.asarray(Kv)
        self.Kr = np.array([8.0, 8.0, 4.0]) if Kr is None else np.asarray(Kr)
        self.Kw = np.array([1.5, 1.5, 1.0]) if Kw is None else np.asarray(Kw)

        # Limits
        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.max_tilt = max_tilt
        self.max_rate = max_rate

        # Gravity vector in world frame
        self.e3 = np.array([0.0, 0.0, 1.0])

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion (w, x, y, z) to rotation matrix.

        Uses same convention as quadrotor_dynamics.py for consistency.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            3x3 rotation matrix (body to world)
        """
        w, x, y, z = q

        # Match the convention in quadrotor_dynamics.py
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z - w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z + w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (w, x, y, z).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def vee_map(self, R_skew: np.ndarray) -> np.ndarray:
        """
        Vee map: Extract vector from skew-symmetric matrix.

        Args:
            R_skew: 3x3 skew-symmetric matrix

        Returns:
            3D vector
        """
        return np.array([R_skew[2, 1], R_skew[0, 2], R_skew[1, 0]])

    def hat_map(self, v: np.ndarray) -> np.ndarray:
        """
        Hat map: Create skew-symmetric matrix from vector.

        Args:
            v: 3D vector

        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def compute_desired_rotation(
        self,
        F_des: np.ndarray,
        yaw_d: float
    ) -> np.ndarray:
        """
        Compute desired rotation matrix that aligns body Z-axis with thrust vector.

        Args:
            F_des: Desired thrust vector in world frame
            yaw_d: Desired yaw angle (rad)

        Returns:
            3x3 desired rotation matrix
        """
        # Desired body Z-axis (direction of thrust)
        F_norm = np.linalg.norm(F_des)
        if F_norm < 1e-6:
            z_B_d = self.e3
        else:
            z_B_d = F_des / F_norm

        # Intermediate X-axis from yaw
        x_C = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])

        # Desired body Y-axis (perpendicular to z_B_d and x_C)
        y_B_d = np.cross(z_B_d, x_C)
        y_norm = np.linalg.norm(y_B_d)
        if y_norm < 1e-6:
            # Singularity: z_B_d parallel to x_C, use alternative
            y_B_d = np.array([-np.sin(yaw_d), np.cos(yaw_d), 0.0])
        else:
            y_B_d = y_B_d / y_norm

        # Desired body X-axis
        x_B_d = np.cross(y_B_d, z_B_d)

        # Construct rotation matrix
        R_d = np.column_stack([x_B_d, y_B_d, z_B_d])

        return R_d

    def compute_attitude_error(
        self,
        R: np.ndarray,
        R_d: np.ndarray
    ) -> np.ndarray:
        """
        Compute attitude error on SO(3) manifold.

        e_R = 0.5 * vee(R_d^T R - R^T R_d)

        Args:
            R: Current rotation matrix
            R_d: Desired rotation matrix

        Returns:
            Attitude error vector (3,)
        """
        error_matrix = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * self.vee_map(error_matrix)
        return e_R

    def compute(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        R: np.ndarray,
        omega: np.ndarray,
        pos_d: np.ndarray,
        vel_d: np.ndarray,
        acc_d: np.ndarray,
        yaw_d: float = 0.0,
        omega_d: np.ndarray = None,
    ) -> Tuple[float, np.ndarray, dict]:
        """
        Compute control outputs (thrust and body rates).

        Args:
            pos: Current position [x, y, z]
            vel: Current velocity [vx, vy, vz]
            R: Current rotation matrix (3x3)
            omega: Current angular velocity in body frame [p, q, r]
            pos_d: Desired position [x, y, z]
            vel_d: Desired velocity [vx, vy, vz]
            acc_d: Desired acceleration [ax, ay, az] (FEEDFORWARD!)
            yaw_d: Desired yaw angle (rad)
            omega_d: Desired angular velocity (optional, default zeros)

        Returns:
            thrust: Collective thrust magnitude (N)
            omega_cmd: Commanded body rates [p, q, r] (rad/s)
            info: Dictionary with debug information
        """
        if omega_d is None:
            omega_d = np.zeros(3)

        # Position and velocity errors
        e_p = pos - pos_d
        e_v = vel - vel_d

        # Desired force vector with feedforward
        # F_des = -Kp * e_p - Kv * e_v + m*g*e3 + m*a_d
        F_des = (
            -self.Kp * e_p
            - self.Kv * e_v
            + self.m * self.g * self.e3
            + self.m * acc_d  # FEEDFORWARD TERM
        )

        # Compute desired rotation
        R_d = self.compute_desired_rotation(F_des, yaw_d)

        # Thrust magnitude: project F_des onto current body Z-axis
        z_B = R[:, 2]  # Current body Z-axis in world frame
        thrust = np.dot(F_des, z_B)

        # Clamp thrust
        thrust = np.clip(thrust, self.min_thrust, self.max_thrust)

        # Attitude error
        e_R = self.compute_attitude_error(R, R_d)

        # Angular velocity error
        e_w = omega - R.T @ R_d @ omega_d

        # Compute desired angular velocity (simplified, using P control on attitude)
        # In full implementation, this would involve R_d_dot
        omega_cmd = -self.Kr * e_R - self.Kw * e_w

        # Clamp body rates
        omega_cmd = np.clip(omega_cmd, -self.max_rate, self.max_rate)

        # Debug info
        info = {
            'e_p': e_p,
            'e_v': e_v,
            'e_R': e_R,
            'e_w': e_w,
            'F_des': F_des,
            'R_d': R_d,
            'thrust_raw': np.dot(F_des, z_B),
        }

        return thrust, omega_cmd, info

    def compute_from_quaternion(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        pos_d: np.ndarray,
        vel_d: np.ndarray,
        acc_d: np.ndarray,
        yaw_d: float = 0.0,
    ) -> Tuple[float, np.ndarray, dict]:
        """
        Compute control outputs using quaternion orientation.

        Args:
            pos: Current position [x, y, z]
            vel: Current velocity [vx, vy, vz]
            quat: Current orientation quaternion [w, x, y, z]
            omega: Current angular velocity in body frame [p, q, r]
            pos_d: Desired position [x, y, z]
            vel_d: Desired velocity [vx, vy, vz]
            acc_d: Desired acceleration [ax, ay, az]
            yaw_d: Desired yaw angle (rad)

        Returns:
            thrust: Collective thrust magnitude (N)
            omega_cmd: Commanded body rates [p, q, r] (rad/s)
            info: Dictionary with debug information
        """
        R = self.quaternion_to_rotation_matrix(quat)
        return self.compute(pos, vel, R, omega, pos_d, vel_d, acc_d, yaw_d)


class GeometricControllerTorch:
    """
    Batched Geometric Controller using PyTorch.

    For use with simulation environments that use batch processing.
    """

    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        Kp: torch.Tensor = None,
        Kv: torch.Tensor = None,
        Kr: torch.Tensor = None,
        Kw: torch.Tensor = None,
        max_thrust: float = 20.0,
        min_thrust: float = 0.5,
        max_rate: float = 3.0,
        device: str = 'cuda:0',
    ):
        """
        Initialize the batched geometric controller.

        Args:
            mass: Quadrotor mass (kg)
            gravity: Gravitational acceleration (m/s^2)
            Kp: Position gains (3,)
            Kv: Velocity gains (3,)
            Kr: Attitude gains (3,)
            Kw: Angular velocity gains (3,)
            max_thrust: Maximum thrust (N)
            min_thrust: Minimum thrust (N)
            max_rate: Maximum body rate (rad/s)
            device: PyTorch device
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.m = mass
        self.g = gravity

        # Default gains
        if Kp is None:
            self.Kp = torch.tensor([6.0, 6.0, 8.0], device=self.device)
        else:
            self.Kp = Kp.to(self.device)

        if Kv is None:
            self.Kv = torch.tensor([4.0, 4.0, 5.0], device=self.device)
        else:
            self.Kv = Kv.to(self.device)

        if Kr is None:
            self.Kr = torch.tensor([8.0, 8.0, 4.0], device=self.device)
        else:
            self.Kr = Kr.to(self.device)

        if Kw is None:
            self.Kw = torch.tensor([1.5, 1.5, 1.0], device=self.device)
        else:
            self.Kw = Kw.to(self.device)

        self.max_thrust = max_thrust
        self.min_thrust = min_thrust
        self.max_rate = max_rate

        # Gravity vector
        self.e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of quaternions (w, x, y, z) to rotation matrices.

        Args:
            q: Quaternions [B, 4] in (w, x, y, z) format

        Returns:
            Rotation matrices [B, 3, 3]
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        B = q.shape[0]

        R = torch.zeros((B, 3, 3), device=self.device)

        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)

        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)

        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)

        return R

    def vee_map_batch(self, R_skew: torch.Tensor) -> torch.Tensor:
        """
        Batched vee map.

        Args:
            R_skew: Skew-symmetric matrices [B, 3, 3]

        Returns:
            Vectors [B, 3]
        """
        return torch.stack([
            R_skew[:, 2, 1],
            R_skew[:, 0, 2],
            R_skew[:, 1, 0]
        ], dim=1)

    def compute_desired_rotation_batch(
        self,
        F_des: torch.Tensor,
        yaw_d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute desired rotation matrices for batch.

        Args:
            F_des: Desired thrust vectors [B, 3]
            yaw_d: Desired yaw angles [B]

        Returns:
            Desired rotation matrices [B, 3, 3]
        """
        B = F_des.shape[0]

        # Normalize thrust vector to get z_B_d
        F_norm = torch.norm(F_des, dim=1, keepdim=True)
        F_norm = torch.clamp(F_norm, min=1e-6)
        z_B_d = F_des / F_norm

        # Intermediate X-axis from yaw
        x_C = torch.stack([
            torch.cos(yaw_d),
            torch.sin(yaw_d),
            torch.zeros_like(yaw_d)
        ], dim=1)

        # y_B_d = z_B_d x x_C (normalized)
        y_B_d = torch.cross(z_B_d, x_C, dim=1)
        y_norm = torch.norm(y_B_d, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=1e-6)
        y_B_d = y_B_d / y_norm

        # x_B_d = y_B_d x z_B_d
        x_B_d = torch.cross(y_B_d, z_B_d, dim=1)

        # Stack into rotation matrix [B, 3, 3]
        R_d = torch.stack([x_B_d, y_B_d, z_B_d], dim=2)

        return R_d

    def compute(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        quat: torch.Tensor,
        omega: torch.Tensor,
        pos_d: torch.Tensor,
        vel_d: torch.Tensor,
        acc_d: torch.Tensor,
        yaw_d: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute batched control outputs.

        Args:
            pos: Current positions [B, 3]
            vel: Current velocities [B, 3]
            quat: Current quaternions [B, 4] (w, x, y, z)
            omega: Current angular velocities [B, 3]
            pos_d: Desired positions [B, 3]
            vel_d: Desired velocities [B, 3]
            acc_d: Desired accelerations [B, 3] (FEEDFORWARD!)
            yaw_d: Desired yaw angles [B] (optional)

        Returns:
            thrust: Thrust magnitudes [B]
            omega_cmd: Commanded body rates [B, 3]
        """
        B = pos.shape[0]

        if yaw_d is None:
            yaw_d = torch.zeros(B, device=self.device)

        # Current rotation matrix
        R = self.quaternion_to_rotation_matrix(quat)

        # Position and velocity errors
        e_p = pos - pos_d
        e_v = vel - vel_d

        # Desired force with feedforward
        F_des = (
            -self.Kp * e_p
            - self.Kv * e_v
            + self.m * self.g * self.e3.unsqueeze(0)
            + self.m * acc_d
        )

        # Desired rotation
        R_d = self.compute_desired_rotation_batch(F_des, yaw_d)

        # Thrust: project F_des onto body Z-axis
        z_B = R[:, :, 2]  # [B, 3]
        thrust = torch.sum(F_des * z_B, dim=1)
        thrust = torch.clamp(thrust, self.min_thrust, self.max_thrust)

        # Attitude error: e_R = 0.5 * vee(R_d^T R - R^T R_d)
        error_matrix = torch.bmm(R_d.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_d)
        e_R = 0.5 * self.vee_map_batch(error_matrix)

        # Angular velocity error (assuming omega_d = 0)
        e_w = omega

        # Commanded body rates
        omega_cmd = -self.Kr * e_R - self.Kw * e_w
        omega_cmd = torch.clamp(omega_cmd, -self.max_rate, self.max_rate)

        return thrust, omega_cmd

    def compute_normalized(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        quat: torch.Tensor,
        omega: torch.Tensor,
        pos_d: torch.Tensor,
        vel_d: torch.Tensor,
        acc_d: torch.Tensor,
        yaw_d: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute control outputs with normalized thrust (0-1 range).

        Returns:
            thrust_normalized: Normalized thrust [B] in [0, 1]
            omega_cmd: Commanded body rates [B, 3]
        """
        thrust, omega_cmd = self.compute(
            pos, vel, quat, omega, pos_d, vel_d, acc_d, yaw_d
        )

        # Normalize thrust to [0, 1]
        thrust_normalized = thrust / self.max_thrust
        thrust_normalized = torch.clamp(thrust_normalized, 0.0, 1.0)

        return thrust_normalized, omega_cmd
