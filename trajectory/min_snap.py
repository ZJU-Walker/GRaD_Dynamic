"""
Minimum Snap Trajectory Generation

Implements smooth trajectory generation using 7th-order polynomial optimization.
Provides continuous position, velocity, acceleration, and jerk profiles.
"""

import numpy as np
from typing import List, Tuple, Optional


class PathProcessor:
    """
    Module A: Path Processor

    Converts raw A* waypoints into sparse, timed keyframes.
    """

    def __init__(self, epsilon: float = 0.1, v_avg: float = 1.0, turn_slowdown: float = 1.3):
        """
        Initialize path processor.

        Args:
            epsilon: RDP simplification threshold (meters)
            v_avg: Average velocity for time allocation (m/s)
            turn_slowdown: Multiplier for segments with sharp turns (>45 deg)
        """
        self.epsilon = epsilon
        self.v_avg = v_avg
        self.turn_slowdown = turn_slowdown

    def rdp_simplify(self, points: np.ndarray) -> np.ndarray:
        """
        Ramer-Douglas-Peucker path simplification.

        Reduces dense waypoints to essential corner keyframes.

        Args:
            points: Nx3 array of waypoints

        Returns:
            Simplified waypoints (Mx3 where M <= N)
        """
        if len(points) <= 2:
            return points

        # Find point with maximum distance from line segment
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return np.array([start, end])

        line_unit = line_vec / line_len

        # Calculate perpendicular distances
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(points) - 1):
            vec_to_point = points[i] - start
            proj_len = np.dot(vec_to_point, line_unit)
            proj_len = np.clip(proj_len, 0, line_len)
            proj_point = start + proj_len * line_unit
            dist = np.linalg.norm(points[i] - proj_point)

            if dist > max_dist:
                max_dist = dist
                max_idx = i

        # Recursively simplify if max distance exceeds threshold
        if max_dist > self.epsilon:
            left = self.rdp_simplify(points[:max_idx + 1])
            right = self.rdp_simplify(points[max_idx:])
            return np.vstack([left[:-1], right])
        else:
            return np.array([start, end])

    def compute_turn_angles(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Compute turning angles at each waypoint.

        Args:
            waypoints: Nx3 array of keyframes

        Returns:
            Array of angles in radians (length N-2, for interior points)
        """
        if len(waypoints) < 3:
            return np.array([])

        angles = []
        for i in range(1, len(waypoints) - 1):
            v1 = waypoints[i] - waypoints[i - 1]
            v2 = waypoints[i + 1] - waypoints[i]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-10 or norm2 < 1e-10:
                angles.append(0.0)
                continue

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)  # Angle between vectors (0 = straight, pi = 180 turn)
            turn_angle = np.pi - angle  # Convert to turning angle
            angles.append(turn_angle)

        return np.array(angles)

    def allocate_times(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Allocate time durations for each segment using trapezoidal heuristic.

        Args:
            waypoints: Nx3 array of keyframes

        Returns:
            Array of segment durations (length N-1)
        """
        n_segments = len(waypoints) - 1
        if n_segments <= 0:
            return np.array([])

        # Compute segment distances
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        # Base time allocation: T = distance / v_avg
        times = distances / self.v_avg

        # Compute turn angles at interior waypoints
        turn_angles = self.compute_turn_angles(waypoints)

        # Slow down segments before sharp turns (>45 degrees)
        angle_threshold = np.pi / 4  # 45 degrees

        for i, angle in enumerate(turn_angles):
            if angle > angle_threshold:
                # Slow down the segment leading into the turn
                times[i] *= self.turn_slowdown
                # Also slow down segment leaving the turn
                if i + 1 < len(times):
                    times[i + 1] *= self.turn_slowdown

        # Ensure minimum segment time
        times = np.maximum(times, 0.5)

        return times

    def process(self, raw_waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process raw waypoints into timed keyframes.

        Args:
            raw_waypoints: Nx3 array of raw A* waypoints

        Returns:
            keyframes: Mx3 array of simplified keyframes
            segment_times: (M-1,) array of segment durations
        """
        raw_waypoints = np.asarray(raw_waypoints)

        # Simplify path
        keyframes = self.rdp_simplify(raw_waypoints)

        # Allocate times
        segment_times = self.allocate_times(keyframes)

        return keyframes, segment_times


class MinSnapTrajectory:
    """
    Module B: Minimum Snap Trajectory Generator

    Creates smooth piecewise polynomial trajectories using closed-form optimization.
    Uses 7th-order (septic) polynomials for C4 continuity.
    """

    ORDER = 7  # Polynomial order (septic)
    N_COEFFS = 8  # Number of coefficients per segment

    def __init__(self):
        """Initialize the minimum snap trajectory generator."""
        self.keyframes = None
        self.segment_times = None
        self.coefficients = None  # Shape: (n_segments, 3, N_COEFFS) for x,y,z
        self.cumulative_times = None
        self.total_time = None

    def _poly_coeffs(self, t: float, deriv: int = 0) -> np.ndarray:
        """
        Compute polynomial basis coefficients at time t for given derivative.

        p(t) = c0 + c1*t + c2*t^2 + ... + c7*t^7

        Args:
            t: Time value
            deriv: Derivative order (0=position, 1=velocity, 2=acceleration, etc.)

        Returns:
            Array of 8 coefficients for the polynomial at time t
        """
        coeffs = np.zeros(self.N_COEFFS)

        for i in range(deriv, self.N_COEFFS):
            # Compute factorial coefficient for derivative
            factor = 1.0
            for j in range(deriv):
                factor *= (i - j)
            coeffs[i] = factor * (t ** (i - deriv))

        return coeffs

    def solve(self, keyframes: np.ndarray, segment_times: np.ndarray) -> np.ndarray:
        """
        Solve for polynomial coefficients using closed-form minimum snap.

        Constraints:
        - Position at each keyframe
        - C4 continuity at segment junctions (pos, vel, acc, jerk, snap match)
        - Zero velocity and acceleration at start and end

        Args:
            keyframes: (N,3) array of waypoint positions
            segment_times: (N-1,) array of segment durations

        Returns:
            Coefficient matrix of shape (n_segments, 3, 8)
        """
        self.keyframes = np.asarray(keyframes)
        self.segment_times = np.asarray(segment_times)

        n_segments = len(segment_times)
        n_waypoints = len(keyframes)

        if n_waypoints != n_segments + 1:
            raise ValueError(f"Expected {n_segments + 1} waypoints for {n_segments} segments")

        # Total unknowns: 8 coefficients per segment
        n_unknowns = n_segments * self.N_COEFFS

        # Build constraint matrix A and vector b for each axis
        self.coefficients = np.zeros((n_segments, 3, self.N_COEFFS))

        for axis in range(3):  # x, y, z
            A = np.zeros((n_unknowns, n_unknowns))
            b = np.zeros(n_unknowns)

            row = 0

            # Constraint 1: Start position
            # p_0(0) = keyframes[0]
            A[row, 0:self.N_COEFFS] = self._poly_coeffs(0.0, deriv=0)
            b[row] = keyframes[0, axis]
            row += 1

            # Constraint 2: Start velocity = 0
            A[row, 0:self.N_COEFFS] = self._poly_coeffs(0.0, deriv=1)
            b[row] = 0.0
            row += 1

            # Constraint 3: Start acceleration = 0
            A[row, 0:self.N_COEFFS] = self._poly_coeffs(0.0, deriv=2)
            b[row] = 0.0
            row += 1

            # Constraint 4: Start jerk = 0 (for smoother motion)
            A[row, 0:self.N_COEFFS] = self._poly_coeffs(0.0, deriv=3)
            b[row] = 0.0
            row += 1

            # Constraints for each segment junction
            for seg in range(n_segments):
                T = segment_times[seg]
                seg_start = seg * self.N_COEFFS

                # Position at end of segment = waypoint
                A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=0)
                b[row] = keyframes[seg + 1, axis]
                row += 1

                # Continuity constraints at junctions (except last segment)
                if seg < n_segments - 1:
                    next_seg_start = (seg + 1) * self.N_COEFFS

                    # C0: Position continuity
                    A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=0)
                    A[row, next_seg_start:next_seg_start + self.N_COEFFS] = -self._poly_coeffs(0.0, deriv=0)
                    b[row] = 0.0
                    row += 1

                    # C1: Velocity continuity
                    A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=1)
                    A[row, next_seg_start:next_seg_start + self.N_COEFFS] = -self._poly_coeffs(0.0, deriv=1)
                    b[row] = 0.0
                    row += 1

                    # C2: Acceleration continuity
                    A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=2)
                    A[row, next_seg_start:next_seg_start + self.N_COEFFS] = -self._poly_coeffs(0.0, deriv=2)
                    b[row] = 0.0
                    row += 1

                    # C3: Jerk continuity
                    A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=3)
                    A[row, next_seg_start:next_seg_start + self.N_COEFFS] = -self._poly_coeffs(0.0, deriv=3)
                    b[row] = 0.0
                    row += 1

                    # C4: Snap continuity
                    A[row, seg_start:seg_start + self.N_COEFFS] = self._poly_coeffs(T, deriv=4)
                    A[row, next_seg_start:next_seg_start + self.N_COEFFS] = -self._poly_coeffs(0.0, deriv=4)
                    b[row] = 0.0
                    row += 1

            # End conditions (last segment)
            last_seg = n_segments - 1
            last_seg_start = last_seg * self.N_COEFFS
            T_last = segment_times[last_seg]

            # End velocity = 0
            A[row, last_seg_start:last_seg_start + self.N_COEFFS] = self._poly_coeffs(T_last, deriv=1)
            b[row] = 0.0
            row += 1

            # End acceleration = 0
            A[row, last_seg_start:last_seg_start + self.N_COEFFS] = self._poly_coeffs(T_last, deriv=2)
            b[row] = 0.0
            row += 1

            # End jerk = 0
            A[row, last_seg_start:last_seg_start + self.N_COEFFS] = self._poly_coeffs(T_last, deriv=3)
            b[row] = 0.0
            row += 1

            # Solve linear system
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fall back to least squares if singular
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Store coefficients
            for seg in range(n_segments):
                self.coefficients[seg, axis, :] = x[seg * self.N_COEFFS:(seg + 1) * self.N_COEFFS]

        # Compute cumulative times for sampling
        self.cumulative_times = np.zeros(n_segments + 1)
        self.cumulative_times[1:] = np.cumsum(segment_times)
        self.total_time = self.cumulative_times[-1]

        return self.coefficients


class TrajectorySampler:
    """
    Module C: Trajectory Sampler

    Provides real-time desired state (position, velocity, acceleration) from trajectory.
    """

    def __init__(self, trajectory: MinSnapTrajectory):
        """
        Initialize sampler with a solved trajectory.

        Args:
            trajectory: MinSnapTrajectory instance with solved coefficients
        """
        self.traj = trajectory

    def _find_segment(self, t: float) -> Tuple[int, float]:
        """
        Find active segment and local time for global time t.

        Args:
            t: Global time (seconds from start)

        Returns:
            segment_idx: Active segment index
            tau: Local time within segment
        """
        # Clamp time to valid range
        t = np.clip(t, 0.0, self.traj.total_time)

        # Find segment
        segment_idx = np.searchsorted(self.traj.cumulative_times[1:], t, side='right')
        segment_idx = min(segment_idx, len(self.traj.segment_times) - 1)

        # Compute local time
        tau = t - self.traj.cumulative_times[segment_idx]
        tau = np.clip(tau, 0.0, self.traj.segment_times[segment_idx])

        return segment_idx, tau

    def _eval_poly(self, coeffs: np.ndarray, t: float, deriv: int = 0) -> float:
        """
        Evaluate polynomial and its derivatives at time t.

        Args:
            coeffs: Array of 8 polynomial coefficients
            t: Time value
            deriv: Derivative order

        Returns:
            Polynomial value at t
        """
        result = 0.0
        for i in range(deriv, len(coeffs)):
            factor = 1.0
            for j in range(deriv):
                factor *= (i - j)
            result += factor * coeffs[i] * (t ** (i - deriv))
        return result

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample desired state at time t.

        Args:
            t: Time from trajectory start (seconds)

        Returns:
            pos_d: Desired position [x, y, z]
            vel_d: Desired velocity [vx, vy, vz]
            acc_d: Desired acceleration [ax, ay, az]
            jerk_d: Desired jerk [jx, jy, jz]
        """
        seg_idx, tau = self._find_segment(t)

        pos_d = np.zeros(3)
        vel_d = np.zeros(3)
        acc_d = np.zeros(3)
        jerk_d = np.zeros(3)

        for axis in range(3):
            coeffs = self.traj.coefficients[seg_idx, axis, :]
            pos_d[axis] = self._eval_poly(coeffs, tau, deriv=0)
            vel_d[axis] = self._eval_poly(coeffs, tau, deriv=1)
            acc_d[axis] = self._eval_poly(coeffs, tau, deriv=2)
            jerk_d[axis] = self._eval_poly(coeffs, tau, deriv=3)

        return pos_d, vel_d, acc_d, jerk_d

    def sample_trajectory(self, dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample entire trajectory at fixed time intervals.

        Args:
            dt: Time step (seconds)

        Returns:
            times: Time array
            positions: (N, 3) position array
            velocities: (N, 3) velocity array
            accelerations: (N, 3) acceleration array
            jerks: (N, 3) jerk array
        """
        times = np.arange(0, self.traj.total_time + dt, dt)
        n_samples = len(times)

        positions = np.zeros((n_samples, 3))
        velocities = np.zeros((n_samples, 3))
        accelerations = np.zeros((n_samples, 3))
        jerks = np.zeros((n_samples, 3))

        for i, t in enumerate(times):
            pos_d, vel_d, acc_d, jerk_d = self.sample(t)
            positions[i] = pos_d
            velocities[i] = vel_d
            accelerations[i] = acc_d
            jerks[i] = jerk_d

        return times, positions, velocities, accelerations, jerks

    def get_yaw_from_velocity(self, vel: np.ndarray, default_yaw: float = 0.0) -> float:
        """
        Compute desired yaw angle from velocity vector.

        Args:
            vel: Velocity vector [vx, vy, vz]
            default_yaw: Default yaw when velocity is near zero

        Returns:
            Yaw angle in radians
        """
        vel_xy = vel[:2]
        if np.linalg.norm(vel_xy) < 0.1:
            return default_yaw
        return np.arctan2(vel[1], vel[0])

    @property
    def total_time(self) -> float:
        """Get total trajectory duration."""
        return self.traj.total_time

    @property
    def is_complete(self) -> bool:
        """Check if trajectory exists and is valid."""
        return self.traj.coefficients is not None


def generate_trajectory(
    waypoints: np.ndarray,
    v_avg: float = 1.0,
    epsilon: float = 0.1
) -> TrajectorySampler:
    """
    High-level function to generate a minimum snap trajectory from waypoints.

    Args:
        waypoints: (N, 3) array of waypoints
        v_avg: Average velocity for time allocation
        epsilon: RDP simplification threshold

    Returns:
        TrajectorySampler instance for querying the trajectory
    """
    # Process path
    processor = PathProcessor(epsilon=epsilon, v_avg=v_avg)
    keyframes, segment_times = processor.process(waypoints)

    # Generate trajectory
    trajectory = MinSnapTrajectory()
    trajectory.solve(keyframes, segment_times)

    # Create sampler
    sampler = TrajectorySampler(trajectory)

    return sampler
