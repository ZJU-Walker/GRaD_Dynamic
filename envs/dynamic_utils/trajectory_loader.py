"""
Trajectory Loader for Dynamic Objects
Loads and interpolates trajectories from CSV files
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


class TrajectoryLoader:
    """Loads and manages trajectories from CSV files"""
    
    def __init__(self, csv_file: str, loop: bool = True, device: str = 'cuda'):
        """
        Initialize trajectory loader
        
        Args:
            csv_file: Path to CSV file with trajectory data (contains absolute positions)
            loop: Whether to loop trajectory when reaching the end
            device: Device for tensors
        """
        self.csv_file = csv_file
        self.loop = loop
        self.device = device
        
        # Load trajectory data
        self.load_trajectory()
        
        # Create interpolators
        self.create_interpolators()
        
        # Track current time for continuous playback
        self.current_time = 0.0
        
    def load_trajectory(self):
        """Load trajectory data from CSV file"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Trajectory file not found: {self.csv_file}")
        
        # Read CSV
        df = pd.read_csv(self.csv_file)
        
        # Check required columns
        required_cols = ['time', 'position_x', 'position_y', 'position_z']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract data
        self.times = df['time'].values
        self.positions = np.column_stack([
            df['position_x'].values,
            df['position_y'].values,
            df['position_z'].values
        ])
        
        # Extract quaternions if available (optional)
        if all(col in df.columns for col in ['quat_x', 'quat_y', 'quat_z', 'quat_w']):
            self.quaternions = np.column_stack([
                df['quat_x'].values,
                df['quat_y'].values,
                df['quat_z'].values,
                df['quat_w'].values
            ])
        else:
            # Default to identity quaternion if not provided
            self.quaternions = np.tile([0, 0, 0, 1], (len(self.times), 1))
        
        # Store duration
        self.duration = self.times[-1] - self.times[0]
        
    
    def create_interpolators(self):
        """Create interpolation functions for smooth trajectory"""
        # Position interpolators
        self.interp_x = interp1d(self.times, self.positions[:, 0], 
                                 kind='linear', fill_value='extrapolate')
        self.interp_y = interp1d(self.times, self.positions[:, 1], 
                                 kind='linear', fill_value='extrapolate')
        self.interp_z = interp1d(self.times, self.positions[:, 2], 
                                 kind='linear', fill_value='extrapolate')
        
        # Quaternion interpolation would use slerp, but for simplicity we'll use nearest
        self.quat_times = self.times
        self.quat_values = self.quaternions
    
    def get_position_at_time(self, time: float) -> torch.Tensor:
        """
        Get interpolated position at given time
        
        Args:
            time: Time in seconds (can be any value, not just at dt=0.01 intervals)
            
        Returns:
            Position tensor (3,) - absolute position interpolated from CSV data
        """
        # Handle looping
        if self.loop and self.duration > 0:
            time = time % self.duration
        else:
            time = np.clip(time, self.times[0], self.times[-1])
        
        # Interpolate position at exact time (not limited to CSV dt intervals)
        x = float(self.interp_x(time))
        y = float(self.interp_y(time))
        z = float(self.interp_z(time))
        
        position = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
        
        # Return absolute position directly (no center offset needed)
        return position
    
    def get_velocity_at_time(self, time: float, dt: float = 0.02) -> torch.Tensor:
        """
        Compute velocity at given time using finite differences
        
        Args:
            time: Time in seconds
            dt: Time step for finite difference
            
        Returns:
            Velocity tensor (3,)
        """
        # Get current and next positions
        pos_current = self.get_position_at_time(time)
        pos_next = self.get_position_at_time(time + dt)
        
        # Compute velocity
        velocity = (pos_next - pos_current) / dt
        
        return velocity
    
    def get_quaternion_at_time(self, time: float) -> torch.Tensor:
        """
        Get quaternion orientation at given time (nearest neighbor for now)
        
        Args:
            time: Time in seconds
            
        Returns:
            Quaternion tensor (4,) as [x, y, z, w]
        """
        # Handle looping
        if self.loop and self.duration > 0:
            time = time % self.duration
        else:
            time = np.clip(time, self.times[0], self.times[-1])
        
        # Find nearest time index
        idx = np.argmin(np.abs(self.quat_times - time))
        quat = self.quat_values[idx]
        
        return torch.tensor(quat, device=self.device, dtype=torch.float32)
    
    def update(self, dt: float):
        """Update internal time counter"""
        self.current_time += dt
        if self.loop and self.duration > 0:
            self.current_time = self.current_time % self.duration
    
    def get_current_position(self) -> torch.Tensor:
        """Get position at current time"""
        return self.get_position_at_time(self.current_time)
    
    def get_current_velocity(self, dt: float = 0.02) -> torch.Tensor:
        """Get velocity at current time"""
        return self.get_velocity_at_time(self.current_time, dt)
    
    def reset(self):
        """Reset trajectory to beginning"""
        self.current_time = 0.0


def create_example_trajectories(output_dir: str = "trajectories"):
    """Create example trajectory CSV files for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Simple back-and-forth movement
    t = np.linspace(0, 10, 21)  # 10 seconds, 21 waypoints
    x = 2.0 * np.sin(2 * np.pi * t / 10)  # Oscillate in X
    y = np.zeros_like(t)  # Stay at Y=0
    z = np.zeros_like(t)  # Stay at Z=0
    
    df1 = pd.DataFrame({
        'time': t,
        'position_x': x,
        'position_y': y,
        'position_z': z,
        'quat_x': np.zeros_like(t),
        'quat_y': np.zeros_like(t),
        'quat_z': np.zeros_like(t),
        'quat_w': np.ones_like(t)
    })
    df1.to_csv(os.path.join(output_dir, 'back_and_forth.csv'), index=False)
    
    # Example 2: Circular motion
    t = np.linspace(0, 10, 41)  # 10 seconds, 41 waypoints
    radius = 3.0
    x = radius * np.cos(2 * np.pi * t / 10)
    y = radius * np.sin(2 * np.pi * t / 10)
    z = np.zeros_like(t)
    
    df2 = pd.DataFrame({
        'time': t,
        'position_x': x,
        'position_y': y,
        'position_z': z,
        'quat_x': np.zeros_like(t),
        'quat_y': np.zeros_like(t),
        'quat_z': np.zeros_like(t),
        'quat_w': np.ones_like(t)
    })
    df2.to_csv(os.path.join(output_dir, 'circular.csv'), index=False)
    
    # Example 3: Figure-8 pattern
    t = np.linspace(0, 10, 41)
    x = 3.0 * np.sin(2 * np.pi * t / 10)
    y = 3.0 * np.sin(4 * np.pi * t / 10)
    z = 0.5 * np.sin(2 * np.pi * t / 10)  # Slight vertical movement
    
    df3 = pd.DataFrame({
        'time': t,
        'position_x': x,
        'position_y': y,
        'position_z': z,
        'quat_x': np.zeros_like(t),
        'quat_y': np.zeros_like(t),
        'quat_z': np.zeros_like(t),
        'quat_w': np.ones_like(t)
    })
    df3.to_csv(os.path.join(output_dir, 'figure_eight.csv'), index=False)
    
    # Example 4: Vertical movement
    t = np.linspace(0, 10, 21)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = 1.5 * np.sin(2 * np.pi * t / 10) + 1.5  # Oscillate between 0 and 3
    
    df4 = pd.DataFrame({
        'time': t,
        'position_x': x,
        'position_y': y,
        'position_z': z,
        'quat_x': np.zeros_like(t),
        'quat_y': np.zeros_like(t),
        'quat_z': np.zeros_like(t),
        'quat_w': np.ones_like(t)
    })
    df4.to_csv(os.path.join(output_dir, 'vertical.csv'), index=False)
    
    print(f"Created example trajectories in {output_dir}/:")
    print("  - back_and_forth.csv: Oscillating movement along X axis")
    print("  - circular.csv: Circular motion in XY plane")
    print("  - figure_eight.csv: Figure-8 pattern in 3D")
    print("  - vertical.csv: Vertical oscillation")


if __name__ == "__main__":
    # Create example trajectories for testing
    create_example_trajectories()