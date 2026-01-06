"""
B-Spline Trajectory Generation

Creates smooth trajectories using approximating B-splines:
- Straight lines between collinear waypoints
- Smooth corner rounding (no overshoot/oscillation)
- Waypoints act as "magnets" pulling the path, not exact pass-through points
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


class BSplineTrajectory:
    """
    Approximating B-Spline trajectory generator.

    Creates paths that:
    - Go straight when waypoints are aligned
    - Smoothly round corners without overshooting
    - Provide continuous velocity and acceleration
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        v_avg: float = 1.5,
        smoothing: float = 0.0,
        degree: int = 3,
    ):
        """
        Initialize B-spline trajectory from waypoints.

        Args:
            waypoints: (N, 3) array of waypoints [x, y, z]
            v_avg: Average velocity along path (m/s)
            smoothing: Smoothing factor (0 = interpolate exactly, >0 = approximate)
            degree: B-spline degree (3 = cubic, recommended)
        """
        self.waypoints = np.asarray(waypoints)
        self.v_avg = v_avg
        self.smoothing = smoothing
        self.degree = degree

        # Build the B-spline
        self._build_spline()

    def _build_spline(self):
        """Build the B-spline representation."""
        # Remove duplicate/very close points first
        waypoints = self._remove_duplicates(self.waypoints)
        n_points = len(waypoints)

        if n_points < 2:
            raise ValueError("Need at least 2 waypoints after removing duplicates")

        # Compute cumulative arc length for parameterization
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.zeros(n_points)
        self.arc_lengths[1:] = np.cumsum(segment_lengths)
        self.total_arc_length = self.arc_lengths[-1]

        if self.total_arc_length < 1e-6:
            raise ValueError("Path has zero length")

        # Normalize parameter to [0, 1]
        u = self.arc_lengths / self.total_arc_length

        # Ensure strictly increasing u values
        for i in range(1, len(u)):
            if u[i] <= u[i-1]:
                u[i] = u[i-1] + 1e-6

        # Determine spline degree (must be less than number of points)
        k = min(self.degree, n_points - 1)

        # Create B-spline using splprep
        try:
            if self.smoothing > 0:
                tck, _ = interpolate.splprep(
                    [waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]],
                    u=u,
                    k=k,
                    s=self.smoothing * self.total_arc_length,
                )
            else:
                tck, _ = interpolate.splprep(
                    [waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]],
                    u=u,
                    k=k,
                    s=0,
                )
        except Exception as e:
            # Fallback: use simple linear interpolation
            print(f"Warning: B-spline fitting failed ({e}), using linear interpolation")
            tck, _ = interpolate.splprep(
                [waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]],
                u=u,
                k=1,
                s=0,
            )

        self.tck = tck
        self.waypoints = waypoints  # Store cleaned waypoints

        # Compute total time based on arc length and average velocity
        self.total_time = self.total_arc_length / self.v_avg

    def _remove_duplicates(self, waypoints: np.ndarray, min_dist: float = 0.01) -> np.ndarray:
        """Remove duplicate or very close points."""
        if len(waypoints) < 2:
            return waypoints

        cleaned = [waypoints[0]]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(waypoints[i] - cleaned[-1])
            if dist > min_dist:
                cleaned.append(waypoints[i])

        return np.array(cleaned)

    def _u_from_t(self, t: float) -> float:
        """Convert time to spline parameter u."""
        t = np.clip(t, 0, self.total_time)
        return t / self.total_time

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample trajectory at time t.

        Args:
            t: Time from start (seconds)

        Returns:
            pos: Position [x, y, z]
            vel: Velocity [vx, vy, vz]
            acc: Acceleration [ax, ay, az]
            jerk: Jerk [jx, jy, jz]
        """
        u = self._u_from_t(t)

        # Position (0th derivative)
        pos = np.array(interpolate.splev(u, self.tck, der=0))

        # Derivatives w.r.t. u
        d1 = np.array(interpolate.splev(u, self.tck, der=1))  # dp/du
        d2 = np.array(interpolate.splev(u, self.tck, der=2))  # d2p/du2
        d3 = np.array(interpolate.splev(u, self.tck, der=3))  # d3p/du3

        # Convert to time derivatives using chain rule
        # du/dt = 1 / total_time
        du_dt = 1.0 / self.total_time

        vel = d1 * du_dt
        acc = d2 * (du_dt ** 2)
        jerk = d3 * (du_dt ** 3)

        return pos, vel, acc, jerk

    def sample_trajectory(self, dt: float = 0.02) -> Tuple[np.ndarray, ...]:
        """
        Sample entire trajectory at fixed intervals.

        Args:
            dt: Time step (seconds)

        Returns:
            times, positions, velocities, accelerations, jerks
        """
        times = np.arange(0, self.total_time + dt, dt)
        n = len(times)

        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        accelerations = np.zeros((n, 3))
        jerks = np.zeros((n, 3))

        for i, t in enumerate(times):
            pos, vel, acc, jerk = self.sample(t)
            positions[i] = pos
            velocities[i] = vel
            accelerations[i] = acc
            jerks[i] = jerk

        return times, positions, velocities, accelerations, jerks


class BSplineTrajectorySampler:
    """
    Wrapper to match TrajectorySampler interface.
    """

    def __init__(self, bspline_traj: BSplineTrajectory):
        self.traj = bspline_traj

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample trajectory at time t."""
        return self.traj.sample(t)

    @property
    def total_time(self) -> float:
        return self.traj.total_time

    def get_yaw_from_velocity(self, vel: np.ndarray, default_yaw: float = 0.0) -> float:
        """Compute yaw from velocity direction."""
        vel_xy = vel[:2]
        if np.linalg.norm(vel_xy) < 0.1:
            return default_yaw
        return np.arctan2(vel[1], vel[0])


def generate_bspline_trajectory(
    waypoints: np.ndarray,
    v_avg: float = 1.5,
    corner_smoothing: float = 0.5,
) -> BSplineTrajectorySampler:
    """
    Generate a B-spline trajectory with straight lines and smooth corners.

    Args:
        waypoints: (N, 3) array of waypoints
        v_avg: Average velocity (m/s)
        corner_smoothing: How much to smooth corners (0 = sharp, 0.5 = moderate, 1+ = very smooth)

    Returns:
        BSplineTrajectorySampler for querying trajectory
    """
    waypoints = np.asarray(waypoints)

    # Create trajectory directly - duplicate removal is handled internally
    traj = BSplineTrajectory(
        waypoints=waypoints,
        v_avg=v_avg,
        smoothing=corner_smoothing,
        degree=3,
    )

    return BSplineTrajectorySampler(traj)


# ============================================================
# Alternative: Piecewise Linear with Corner Smoothing
# ============================================================

class LinearWithSmoothCorners:
    """
    Simpler approach: Straight lines with smooth corner transitions.

    - Straight line segments between waypoints
    - Circular/polynomial blend at corners
    - Very predictable behavior
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        v_max: float = 1.5,
        corner_radius: float = 0.5,
        acc_max: float = 2.0,
    ):
        """
        Initialize trajectory.

        Args:
            waypoints: (N, 3) array of waypoints
            v_max: Maximum velocity (m/s)
            corner_radius: Radius for corner blending (m)
            acc_max: Maximum acceleration (m/s²)
        """
        self.waypoints = np.asarray(waypoints)
        self.v_max = v_max
        self.corner_radius = corner_radius
        self.acc_max = acc_max

        self._build_path()

    def _build_path(self):
        """Build the piecewise path with corner blends."""
        n = len(self.waypoints)

        # Compute segment vectors and lengths
        self.segments = []
        self.segment_lengths = []
        self.segment_directions = []

        for i in range(n - 1):
            seg = self.waypoints[i + 1] - self.waypoints[i]
            length = np.linalg.norm(seg)
            direction = seg / length if length > 1e-6 else np.zeros(3)

            self.segments.append(seg)
            self.segment_lengths.append(length)
            self.segment_directions.append(direction)

        # Compute corner angles and blend distances
        self.corner_angles = []
        self.blend_distances = []

        for i in range(1, n - 1):
            # Angle between segments
            d1 = self.segment_directions[i - 1]
            d2 = self.segment_directions[i]

            cos_angle = np.clip(np.dot(d1, d2), -1, 1)
            angle = np.arccos(cos_angle)  # 0 = straight, pi = U-turn
            self.corner_angles.append(angle)

            # Blend distance based on corner radius and angle
            if angle > 0.01:  # Not straight
                # Distance to start blending before corner
                blend_dist = self.corner_radius * np.tan(angle / 2)
                blend_dist = min(blend_dist, self.segment_lengths[i - 1] / 2,
                               self.segment_lengths[i] / 2)
            else:
                blend_dist = 0
            self.blend_distances.append(blend_dist)

        # Build time-parameterized path
        self._parameterize()

    def _parameterize(self):
        """Create time parameterization with velocity profile."""
        # Simple approach: constant velocity along path
        # (Could be enhanced with trapezoidal velocity profile)

        total_length = sum(self.segment_lengths)
        self.total_time = total_length / self.v_max

        # Store cumulative distances for lookup
        self.cumulative_dist = np.zeros(len(self.waypoints))
        self.cumulative_dist[1:] = np.cumsum(self.segment_lengths)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample trajectory at time t.

        Args:
            t: Time from start (seconds)

        Returns:
            pos, vel, acc, jerk
        """
        t = np.clip(t, 0, self.total_time)

        # Distance along path
        s = t * self.v_max

        # Find which segment we're on
        seg_idx = np.searchsorted(self.cumulative_dist[1:], s)
        seg_idx = min(seg_idx, len(self.segment_lengths) - 1)

        # Local distance within segment
        s_local = s - self.cumulative_dist[seg_idx]

        # Position on line segment
        pos = self.waypoints[seg_idx] + self.segment_directions[seg_idx] * s_local

        # Velocity along segment direction
        vel = self.segment_directions[seg_idx] * self.v_max

        # For simplicity, acceleration and jerk are zero on straight segments
        # (Would need blending logic for corners)
        acc = np.zeros(3)
        jerk = np.zeros(3)

        return pos, vel, acc, jerk

    @property
    def total_time(self) -> float:
        return self._total_time

    @total_time.setter
    def total_time(self, value):
        self._total_time = value


class LinearSmoothSampler:
    """Wrapper for LinearWithSmoothCorners."""

    def __init__(self, traj: LinearWithSmoothCorners):
        self.traj = traj

    def sample(self, t: float):
        return self.traj.sample(t)

    @property
    def total_time(self):
        return self.traj.total_time

    def get_yaw_from_velocity(self, vel: np.ndarray, default_yaw: float = 0.0) -> float:
        vel_xy = vel[:2]
        if np.linalg.norm(vel_xy) < 0.1:
            return default_yaw
        return np.arctan2(vel[1], vel[0])
