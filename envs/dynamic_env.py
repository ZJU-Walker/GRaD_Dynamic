"""Dynamic environment with moving obstacle support.

Extends ExpertDroneEnv with:
- DynamicObjectManager for sphere/box/cylinder obstacles
- DepthAugmentor for depth/RGB injection via ray-casting
- Random or trajectory-based spawning
"""

import torch
import numpy as np

from envs.expert_env import ExpertDroneEnv
from envs.dynamic_utils import (
    DynamicObjectManager,
    DepthAugmentor,
    TrajectoryLoader,
    LinearPattern,
    CircularPattern,
    SinusoidalPattern,
    TrajectoryPattern,
)


class DynamicDroneEnv(ExpertDroneEnv):
    """Environment with dynamic obstacle support (sphere, box, cylinder)."""

    def __init__(
        self,
        render: bool = False,
        device: str = 'cuda:0',
        num_envs: int = 4096,
        seed: int = 0,
        episode_length: int = 1000,
        no_grad: bool = True,
        stochastic_init: bool = False,
        MM_caching_frequency: int = 1,
        early_termination: bool = True,
        map_name: str = 'gate_mid',
        env_hyper: dict = None,
        vel_net_cfg: dict = None,
        # Dynamic obstacle config (separate section in YAML)
        dynamic_objects_cfg: dict = None,
    ):
        # Initialize parent
        super().__init__(
            render=render,
            device=device,
            num_envs=num_envs,
            seed=seed,
            episode_length=episode_length,
            no_grad=no_grad,
            stochastic_init=stochastic_init,
            MM_caching_frequency=MM_caching_frequency,
            early_termination=early_termination,
            map_name=map_name,
            env_hyper=env_hyper,
            vel_net_cfg=vel_net_cfg,
        )

        # Dynamic object settings from separate config section
        self.dynamic_objects_cfg = dynamic_objects_cfg or {}
        self.use_dynamic_objects = self.dynamic_objects_cfg.get('enabled', False)

        if self.use_dynamic_objects:
            self._init_dynamic_objects()

    def _init_dynamic_objects(self):
        """Initialize dynamic object manager and depth augmentor."""
        cfg = self.dynamic_objects_cfg

        # Max objects per environment
        max_objects = cfg.get('max_objects_per_env', 10)

        # Initialize DynamicObjectManager
        self.dynamic_manager = DynamicObjectManager(
            num_envs=self.num_envs,
            device=self.device,
            max_objects_per_env=max_objects,
        )

        # Camera parameters for depth augmentation
        # These should match the GS camera intrinsics
        camera_params = cfg.get('camera_params', {
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 180.0,
            'width': 640,
            'height': 360,
        })

        # Initialize DepthAugmentor
        self.depth_augmentor = DepthAugmentor(
            camera_params=camera_params,
            device=self.device,
        )

        # Trajectory config - list of objects to spawn
        self.trajectory_objects = cfg.get('objects', [])

        # Collision tracking (for future reward integration)
        self.collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.min_dist_to_dynamic = torch.full((self.num_envs,), float('inf'), device=self.device)

        print(f"[DynamicDroneEnv] Initialized with max_objects={max_objects}, "
              f"num_trajectory_objects={len(self.trajectory_objects)}")

    def _spawn_dynamic_objects(self, env_ids: torch.Tensor):
        """Spawn dynamic objects in specified environments based on config."""
        # Reset objects in these environments first
        self.dynamic_manager.reset_env(env_ids)

        # Spawn each configured object
        for obj_cfg in self.trajectory_objects:
            obj_type = obj_cfg.get('type', 'sphere')
            radius = obj_cfg.get('radius', 0.5)
            trajectory_file = obj_cfg.get('trajectory', None)

            for env_id in env_ids:
                env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id

                if trajectory_file:
                    try:
                        # Create trajectory pattern
                        pattern = TrajectoryPattern(
                            trajectory_file=trajectory_file,
                            loop=obj_cfg.get('loop', True),
                            device=self.device,
                        )

                        # Get initial position from trajectory
                        init_pos = pattern.get_position(0.0)

                        # Spawn object with trajectory
                        self.dynamic_manager.spawn_object(
                            env_id=env_id_int,
                            position=init_pos,
                            velocity=torch.zeros(3, device=self.device),
                            radius=radius,
                            pattern=pattern,
                            obj_type=obj_type,
                        )
                    except Exception as e:
                        print(f"[DynamicDroneEnv] Failed to load trajectory {trajectory_file}: {e}")
                else:
                    # Static position from config
                    position = torch.tensor(
                        obj_cfg.get('position', [0, 0, 1]),
                        device=self.device,
                        dtype=torch.float32
                    )
                    velocity = torch.tensor(
                        obj_cfg.get('velocity', [0, 0, 0]),
                        device=self.device,
                        dtype=torch.float32
                    )

                    self.dynamic_manager.spawn_object(
                        env_id=env_id_int,
                        position=position,
                        velocity=velocity,
                        radius=radius,
                        obj_type=obj_type,
                    )

    def _update_dynamic_objects(self):
        """Update dynamic object positions."""
        self.dynamic_manager.update(self.sim_dt)

    def _check_dynamic_collisions(self):
        """Check for collisions with dynamic objects."""
        # Get drone positions
        drone_pos = self.state_joint_q[:, 0:3]

        # Compute distances to dynamic objects
        self.min_dist_to_dynamic = self.dynamic_manager.get_distances_to_point(drone_pos)

        # Check for collisions (distance < 0 means overlap)
        collision_threshold = self.dynamic_objects_cfg.get('collision_threshold', 0.3)
        self.collision_buf = self.min_dist_to_dynamic < collision_threshold

    def reset(self, env_ids=None, force_reset=True):
        """Reset environments and spawn dynamic objects."""
        # Call parent reset
        obs = super().reset(env_ids, force_reset)

        # Spawn dynamic objects
        if self.use_dynamic_objects:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            self._spawn_dynamic_objects(env_ids)

        return obs

    def step(self, actions, vae_info):
        """Step with dynamic object update."""
        # Update dynamic objects before physics step
        if self.use_dynamic_objects:
            self._update_dynamic_objects()

        # Call parent step
        result = super().step(actions, vae_info)

        # Check collisions (for future reward integration)
        if self.use_dynamic_objects:
            self._check_dynamic_collisions()

        return result

    def process_GS_data(self, depth_list, rgb_img):
        """Process GS data and inject dynamic objects."""
        # Call parent to get visual features
        super().process_GS_data(depth_list, rgb_img)

        # Inject dynamic objects into depth and RGB
        if self.use_dynamic_objects:
            # Get camera poses for augmentation
            camera_poses = self._get_camera_poses()

            # Inject objects
            augmented_depth, augmented_rgb = self.depth_augmentor.inject_objects_with_rgb(
                depth_maps=depth_list,
                rgb_images=rgb_img,
                camera_poses=camera_poses,
                dynamic_manager=self.dynamic_manager,
                use_shading=True,
            )

            # Store augmented data (for visualization if needed)
            self.augmented_depth = augmented_depth
            self.augmented_rgb = augmented_rgb

            # Re-process visual features with augmented data
            self._update_visual_features(augmented_depth, augmented_rgb)

    def _get_camera_poses(self):
        """Get camera poses for all environments.

        Returns:
            camera_poses: (num_envs, 7) tensor with [x, y, z, qx, qy, qz, qw]
        """
        # Get drone state
        torso_pos = self.state_joint_q[:, 0:3]
        torso_quat = self.state_joint_q[:, 3:7]  # (x, y, z, w)

        # Apply GS coordinate transform (same as in calculateObservations)
        gs_pos = torso_pos + self.gs_origin_offset
        gs_pos[:, 1] = -gs_pos[:, 1]
        gs_pos[:, 2] = -gs_pos[:, 2]

        # Combine into camera pose
        camera_poses = torch.cat([gs_pos, torso_quat], dim=-1)

        return camera_poses

    def _update_visual_features(self, depth_list, rgb_img):
        """Update visual features with augmented images."""
        import torch.nn as nn

        _, H, _, _ = depth_list.shape
        depth_list_up = depth_list[:, 0:int(H/2), :, :]
        self.depth_list = torch.abs(torch.amin(depth_list_up, dim=(1, 2, 3))).unsqueeze(1).to(device=self.device)

        visual_tensor = rgb_img.permute(0, 3, 1, 2)
        resize = nn.AdaptiveAvgPool2d((224, 224))
        visual_tensor = resize(visual_tensor).to(self.device)
        self.visual_info = self.visual_net(visual_tensor).detach()

        # Update DualEncoder features for vel_net (if enabled)
        if self.vel_net_enabled and self.dual_encoder is not None:
            with torch.no_grad():
                depth_tensor = depth_list.permute(0, 3, 1, 2)
                depth_tensor = resize(depth_tensor).to(self.device)
                self.rgb_feat_vel, self.depth_feat_vel = self.dual_encoder(visual_tensor, depth_tensor)

    def get_dynamic_object_info(self):
        """Get information about dynamic objects for debugging/visualization.

        Returns:
            dict with positions, velocities, radii, types for active objects
        """
        if not self.use_dynamic_objects:
            return None

        return {
            'positions': self.dynamic_manager.positions.clone(),
            'velocities': self.dynamic_manager.velocities.clone(),
            'radii': self.dynamic_manager.radii.clone(),
            'active': self.dynamic_manager.active.clone(),
            'object_types': self.dynamic_manager.object_types,
            'min_distance': self.min_dist_to_dynamic.clone(),
            'collisions': self.collision_buf.clone(),
        }

    def spawn_box(self, env_id: int, position, size, velocity=None, rotation=None):
        """Convenience method to spawn a box obstacle.

        Args:
            env_id: Environment index
            position: (3,) position tensor or list
            size: (3,) box dimensions [width, height, depth]
            velocity: (3,) velocity tensor (optional)
            rotation: (4,) quaternion [x,y,z,w] (optional)
        """
        if not self.use_dynamic_objects:
            print("[DynamicDroneEnv] Dynamic objects not enabled")
            return

        position = torch.tensor(position, device=self.device, dtype=torch.float32)
        size = list(size) if not isinstance(size, list) else size
        velocity = torch.zeros(3, device=self.device) if velocity is None else torch.tensor(velocity, device=self.device)
        rotation = [0, 0, 0, 1] if rotation is None else list(rotation)

        self.dynamic_manager.spawn_object(
            env_id=env_id,
            position=position,
            velocity=velocity,
            obj_type='box',
            box_size=size,
            box_rotation=rotation,
        )

    def spawn_cylinder(self, env_id: int, position, radius, height, axis=2, velocity=None, rotation=None):
        """Convenience method to spawn a cylinder obstacle.

        Args:
            env_id: Environment index
            position: (3,) position tensor or list
            radius: Cylinder radius
            height: Cylinder height
            axis: Main axis (0=X, 1=Y, 2=Z)
            velocity: (3,) velocity tensor (optional)
            rotation: (4,) quaternion [x,y,z,w] (optional)
        """
        if not self.use_dynamic_objects:
            print("[DynamicDroneEnv] Dynamic objects not enabled")
            return

        position = torch.tensor(position, device=self.device, dtype=torch.float32)
        velocity = torch.zeros(3, device=self.device) if velocity is None else torch.tensor(velocity, device=self.device)
        rotation = [0, 0, 0, 1] if rotation is None else list(rotation)

        self.dynamic_manager.spawn_object(
            env_id=env_id,
            position=position,
            velocity=velocity,
            obj_type='cylinder',
            cylinder_radius=radius,
            cylinder_height=height,
            cylinder_axis=axis,
            cylinder_rotation=rotation,
        )

    def spawn_sphere(self, env_id: int, position, radius, velocity=None, pattern=None):
        """Convenience method to spawn a sphere obstacle.

        Args:
            env_id: Environment index
            position: (3,) position tensor or list
            radius: Sphere radius
            velocity: (3,) velocity tensor (optional)
            pattern: MovementPattern instance (optional)
        """
        if not self.use_dynamic_objects:
            print("[DynamicDroneEnv] Dynamic objects not enabled")
            return

        position = torch.tensor(position, device=self.device, dtype=torch.float32)
        velocity = torch.zeros(3, device=self.device) if velocity is None else torch.tensor(velocity, device=self.device)

        self.dynamic_manager.spawn_object(
            env_id=env_id,
            position=position,
            velocity=velocity,
            radius=radius,
            pattern=pattern,
            obj_type='sphere',
        )
