"""
Dynamic Objects Manager for GRaD-Nav
Manages dynamic obstacles (spheres, boxes, cylinders) across parallel environments
"""

import torch
import numpy as np
from typing import List, Optional
import random

from .trajectory_loader import TrajectoryLoader


class DynamicObject:
    """Single dynamic object with position, velocity, and properties"""

    def __init__(self,
                 position: torch.Tensor,
                 velocity: torch.Tensor,
                 radius: float = 0.3,
                 obj_type: str = 'sphere',
                 obj_id: int = 0,
                 device: str = 'cuda'):

        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.device = device
        self.time = 0.0

        self.movement_pattern = None
        self.color = torch.tensor([1.0, 0.0, 0.0])
        self.material = 'diffuse'

    def update(self, dt: float):
        """Update position based on velocity and dt"""
        if self.movement_pattern:
            self.velocity = self.movement_pattern.get_velocity(
                self.position, self.time
            )
        self.position = self.position + self.velocity * dt
        self.time += dt

    def set_movement_pattern(self, pattern):
        """Assign a movement pattern to this object"""
        self.movement_pattern = pattern


class MovementPattern:
    """Base class for all movement patterns"""
    def get_velocity(self, position, time):
        raise NotImplementedError


class LinearPattern(MovementPattern):
    """Constant velocity in a direction"""
    def __init__(self, velocity: torch.Tensor):
        self.velocity = velocity

    def get_velocity(self, position, time):
        return self.velocity


class CircularPattern(MovementPattern):
    """Circular/orbital movement around a center"""
    def __init__(self, center, radius, angular_speed, axis='z'):
        self.center = center
        self.radius = radius
        self.omega = angular_speed
        self.axis = axis

    def get_velocity(self, position, time):
        angle = self.omega * time
        if self.axis == 'z':
            vx = -self.radius * self.omega * torch.sin(angle)
            vy = self.radius * self.omega * torch.cos(angle)
            vz = 0
        return torch.tensor([vx, vy, vz])


class SinusoidalPattern(MovementPattern):
    """Oscillating movement"""
    def __init__(self, amplitude, frequency, direction):
        self.amplitude = amplitude
        self.frequency = frequency
        self.direction = direction / torch.norm(direction)

    def get_velocity(self, position, time):
        speed = self.amplitude * self.frequency * torch.cos(
            2 * torch.pi * self.frequency * time
        )
        return speed * self.direction


class RandomWalkPattern(MovementPattern):
    """Stochastic movement"""
    def __init__(self, speed_range, change_interval=1.0):
        self.speed_range = speed_range
        self.change_interval = change_interval
        self.current_velocity = None
        self.last_change_time = 0

    def get_velocity(self, position, time):
        if time - self.last_change_time > self.change_interval:
            speed = torch.rand(1) * (self.speed_range[1] - self.speed_range[0]) + self.speed_range[0]
            direction = torch.randn(3)
            direction = direction / torch.norm(direction)
            self.current_velocity = speed * direction
            self.last_change_time = time
        return self.current_velocity


class TrajectoryPattern(MovementPattern):
    """Movement pattern based on predefined trajectory from CSV file"""
    def __init__(self, trajectory_file: str, loop: bool = True, device: str = 'cuda',
                 randomize_start_time: bool = True, eval_time_offset: float = None):
        self.trajectory_loader = TrajectoryLoader(
            csv_file=trajectory_file,
            loop=loop,
            device=device
        )
        self.device = device
        self.last_time = 0.0
        self.randomize_start_time = randomize_start_time
        self.eval_time_offset = eval_time_offset

        # Set initial time offset
        self.time_offset = 0.0
        if self.eval_time_offset is not None:
            # Fixed offset for eval — deterministic, reproducible
            self.time_offset = self.eval_time_offset
        elif self.randomize_start_time and self.trajectory_loader.duration > 0:
            self.time_offset = random.uniform(-5, 10)
        self.trajectory_loader.current_time = self.time_offset

    def get_velocity(self, position, time):
        dt = time - self.last_time
        if dt > 0:
            self.trajectory_loader.update(dt)
            self.last_time = time
        velocity = self.trajectory_loader.get_current_velocity()
        return velocity

    def get_position(self, time):
        return self.trajectory_loader.get_position_at_time(time + self.time_offset)

    def reset(self):
        self.trajectory_loader.reset()
        self.last_time = 0.0
        # Re-set time offset on reset
        if self.eval_time_offset is not None:
            self.time_offset = self.eval_time_offset
        elif self.randomize_start_time and self.trajectory_loader.duration > 0:
            self.time_offset = random.uniform(-5, 10)
        self.trajectory_loader.current_time = self.time_offset


class DynamicObjectManager:
    """Manages dynamic objects across multiple parallel environments"""

    def __init__(self,
                 num_envs: int,
                 device: str = 'cuda',
                 max_objects_per_env: int = 10):

        self.num_envs = num_envs
        self.device = device
        self.max_objects = max_objects_per_env

        # Batched storage for efficiency
        self.positions = torch.zeros(self.num_envs, self.max_objects, 3, device=device)
        self.velocities = torch.zeros(self.num_envs, self.max_objects, 3, device=device)
        self.radii = torch.zeros(self.num_envs, self.max_objects, device=device)
        self.active = torch.zeros(self.num_envs, self.max_objects, dtype=torch.bool, device=device)

        # Object types and shape parameters
        self.object_types = [['sphere' for _ in range(self.max_objects)] for _ in range(self.num_envs)]

        # Box parameters
        self.box_sizes = torch.zeros(self.num_envs, self.max_objects, 3, device=device)
        self.box_rotations = torch.zeros(self.num_envs, self.max_objects, 4, device=device)
        self.box_rotations[:, :, 3] = 1.0  # Identity quaternion

        # Cylinder parameters: (radius, height, axis)
        self.cylinder_params = torch.zeros(self.num_envs, self.max_objects, 3, device=device)
        self.cylinder_rotations = torch.zeros(self.num_envs, self.max_objects, 4, device=device)
        self.cylinder_rotations[:, :, 3] = 1.0

        # Movement patterns per object
        self.patterns = [[None for _ in range(self.max_objects)] for _ in range(self.num_envs)]

        # Time tracking
        self.time = torch.zeros(self.num_envs, device=device)

    def spawn_object(self,
                     env_id: int,
                     position: torch.Tensor,
                     velocity: torch.Tensor,
                     radius: float = 0.3,
                     pattern: MovementPattern = None,
                     obj_type: str = 'sphere',
                     box_size: list = None,
                     box_rotation: list = None,
                     cylinder_radius: float = None,
                     cylinder_height: float = None,
                     cylinder_axis: int = 2,
                     cylinder_rotation: list = None):
        """Spawn a single object in specified environment"""

        position = position.to(self.device)
        velocity = velocity.to(self.device)

        # Find first inactive slot
        for obj_idx in range(self.max_objects):
            if not self.active[env_id, obj_idx]:
                self.positions[env_id, obj_idx] = position
                self.velocities[env_id, obj_idx] = velocity
                self.active[env_id, obj_idx] = True
                self.patterns[env_id][obj_idx] = pattern
                self.object_types[env_id][obj_idx] = obj_type

                if obj_type == 'sphere':
                    self.radii[env_id, obj_idx] = radius
                elif obj_type == 'box' and box_size is not None:
                    self.box_sizes[env_id, obj_idx] = torch.tensor(box_size, device=self.device)
                    self.radii[env_id, obj_idx] = max(box_size) / 2.0
                    if box_rotation is not None:
                        self.box_rotations[env_id, obj_idx] = torch.tensor(box_rotation, device=self.device)
                    else:
                        self.box_rotations[env_id, obj_idx] = torch.tensor([0, 0, 0, 1], device=self.device)
                elif obj_type == 'cylinder' and cylinder_radius is not None and cylinder_height is not None:
                    self.cylinder_params[env_id, obj_idx] = torch.tensor(
                        [cylinder_radius, cylinder_height, cylinder_axis],
                        device=self.device
                    )
                    self.radii[env_id, obj_idx] = max(cylinder_radius, cylinder_height / 2.0)
                    if cylinder_rotation is not None:
                        self.cylinder_rotations[env_id, obj_idx] = torch.tensor(cylinder_rotation, device=self.device)
                    else:
                        self.cylinder_rotations[env_id, obj_idx] = torch.tensor([0, 0, 0, 1], device=self.device)
                break

    def spawn_random_objects(self,
                            env_ids: torch.Tensor,
                            num_objects: int,
                            spawn_region: dict):
        """Spawn random objects in multiple environments"""

        for env_id in env_ids:
            for _ in range(num_objects):
                x = torch.rand(1) * (spawn_region['x_max'] - spawn_region['x_min']) + spawn_region['x_min']
                y = torch.rand(1) * (spawn_region['y_max'] - spawn_region['y_min']) + spawn_region['y_min']
                z = torch.rand(1) * (spawn_region['z_max'] - spawn_region['z_min']) + spawn_region['z_min']
                position = torch.tensor([x, y, z], device=self.device)

                speed = torch.rand(1) * (spawn_region['speed_max'] - spawn_region['speed_min']) + spawn_region['speed_min']
                direction = torch.randn(3, device=self.device)
                direction = direction / torch.norm(direction)
                velocity = speed * direction

                radius = torch.rand(1) * (spawn_region['radius_max'] - spawn_region['radius_min']) + spawn_region['radius_min']

                pattern_type = torch.randint(0, 3, (1,)).item()
                if pattern_type == 0:
                    pattern = LinearPattern(velocity)
                elif pattern_type == 1:
                    pattern = CircularPattern(position, 1.0, 0.5)
                else:
                    pattern = SinusoidalPattern(1.0, 0.5, torch.randn(3))

                self.spawn_object(env_id.item(), position, velocity, radius.item(), pattern)

    def update(self, dt: float):
        """Update all active objects"""

        self.time += dt
        mask = self.active.unsqueeze(-1)

        # Update velocities from patterns
        for env_id in range(self.num_envs):
            for obj_idx in range(self.max_objects):
                if self.active[env_id, obj_idx] and self.patterns[env_id][obj_idx]:
                    self.velocities[env_id, obj_idx] = self.patterns[env_id][obj_idx].get_velocity(
                        self.positions[env_id, obj_idx],
                        self.time[env_id].item()
                    )

        # Update positions (out-of-place to avoid breaking autograd)
        self.positions = self.positions + self.velocities * dt * mask

    def get_distances_to_point(self, points: torch.Tensor):
        """
        Compute minimum distance from points to any active object

        Args:
            points: (num_envs, 3) query points (e.g., drone positions)

        Returns:
            distances: (num_envs,) minimum distance to any object
        """
        points_expanded = points.unsqueeze(1)
        distances = torch.norm(self.positions - points_expanded, dim=2)
        distances = distances - self.radii
        distances[~self.active] = float('inf')

        if self.max_objects > 0:
            min_distances, _ = torch.min(distances, dim=1)
        else:
            min_distances = torch.full((self.num_envs,), float('inf'), device=self.device)

        return min_distances

    def reset_env(self, env_ids: torch.Tensor):
        """Reset objects in specified environments"""
        for env_id in env_ids:
            self.active[env_id] = False
            self.positions[env_id] = 0
            self.velocities[env_id] = 0
            self.radii[env_id] = 0
            self.box_sizes[env_id] = 0
            self.object_types[env_id] = ['sphere'] * self.max_objects
            self.patterns[env_id] = [None] * self.max_objects
            self.time[env_id] = 0

    def get_active_objects(self, env_id: int):
        """Get list of active objects for a specific environment"""
        active_mask = self.active[env_id]
        objects = []

        for obj_idx in range(self.max_objects):
            if active_mask[obj_idx]:
                obj = DynamicObject(
                    position=self.positions[env_id, obj_idx],
                    velocity=self.velocities[env_id, obj_idx],
                    radius=self.radii[env_id, obj_idx].item(),
                    obj_id=obj_idx,
                    device=self.device
                )
                objects.append(obj)

        return objects
