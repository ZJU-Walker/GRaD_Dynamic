"""Dynamic environment with moving obstacle support.

Extends ExpertDroneEnv with:
- DynamicObjectManager for sphere/box/cylinder obstacles
- DepthAugmentor for depth/RGB injection via ray-casting
- Random or trajectory-based spawning
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

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


def get_T_world_to_camera(pos, quat_xyzw, device):
    """
    Compute 4x4 transformation matrix from world to camera frame.
    Accounts for GS coordinate flip (Y and Z negated).

    From test_dynamic_obstacle.py - this is the correct transformation.
    """
    # Apply GS coordinate flip to position
    gs_pos = pos.detach().cpu().numpy().copy()
    gs_pos[1] = -gs_pos[1]
    gs_pos[2] = -gs_pos[2]

    # Convert quaternion to rotation matrix
    quat_np = quat_xyzw.detach().cpu().numpy()
    R_world_to_drone = R.from_quat(quat_np).as_matrix()

    # Apply coordinate flip to rotation
    flip_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    R_world_to_drone_gs = R_world_to_drone @ flip_matrix

    # Drone to camera rotation
    R_drone_to_camera = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    R_world_to_camera = R_world_to_drone_gs @ R_drone_to_camera

    # Build 4x4 transformation matrix
    rotation_part = R_world_to_camera.T
    translation_part = -rotation_part @ gs_pos

    T = np.eye(4)
    T[:3, :3] = rotation_part
    T[:3, 3] = translation_part

    return torch.tensor(T, device=device, dtype=torch.float32).unsqueeze(0)


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
        """Spawn dynamic objects in specified environments based on config.

        Supports sphere, box, and cylinder object types with full parameter support.
        """

        # Reset objects in these environments first
        self.dynamic_manager.reset_env(env_ids)

        # Spawn each configured object
        for obj_cfg in self.trajectory_objects:
            obj_type = obj_cfg.get('type', 'sphere')

            # Generic radius (for spheres)
            radius = obj_cfg.get('radius', 0.5)

            # Cylinder parameters
            cylinder_radius = obj_cfg.get('cylinder_radius', None)
            cylinder_height = obj_cfg.get('cylinder_height', None)
            cylinder_axis = obj_cfg.get('cylinder_axis', 2)
            cylinder_rotation = obj_cfg.get('cylinder_rotation', None)

            # Box parameters
            box_size = obj_cfg.get('box_size', None)
            box_rotation = obj_cfg.get('box_rotation', None)

            # Trajectory file
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
                    except Exception as e:
                        print(f"[DynamicDroneEnv] Failed to load trajectory {trajectory_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    # Static position from config
                    init_pos = torch.tensor(
                        obj_cfg.get('position', [0, 0, 1]),
                        device=self.device,
                        dtype=torch.float32
                    )
                    pattern = None

                # Get velocity from config (default zeros)
                velocity = torch.tensor(
                    obj_cfg.get('velocity', [0, 0, 0]),
                    device=self.device,
                    dtype=torch.float32
                )

                # Spawn object with all parameters
                self.dynamic_manager.spawn_object(
                    env_id=env_id_int,
                    position=init_pos,
                    velocity=velocity,
                    radius=radius,
                    pattern=pattern,
                    obj_type=obj_type,
                    cylinder_radius=cylinder_radius,
                    cylinder_height=cylinder_height,
                    cylinder_axis=cylinder_axis,
                    cylinder_rotation=cylinder_rotation,
                    box_size=box_size,
                    box_rotation=box_rotation,
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
        # Spawn dynamic objects BEFORE parent reset
        # (because parent reset calls calculateObservations -> process_GS_data)
        if self.use_dynamic_objects:
            if env_ids is None:
                spawn_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            else:
                spawn_env_ids = env_ids
            self._spawn_dynamic_objects(spawn_env_ids)

        # Call parent reset (this will call calculateObservations -> process_GS_data)
        obs = super().reset(env_ids, force_reset)

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
        # Using camera_poses approach like original drone_dynamic_expert.py
        if self.use_dynamic_objects:
            # Get drone state
            torso_pos = self.state_joint_q[:, 0:3]
            torso_quat = self.state_joint_q[:, 3:7]  # (x, y, z, w)

            # Convert drone pose to camera pose for each environment
            # camera_poses format: (B, 7) with [x, y, z, qx, qy, qz, qw]
            camera_poses_list = []
            for i in range(self.num_envs):
                cam_pos, cam_quat = self._get_camera_pose_from_torso(
                    torso_pos[i], torso_quat[i]
                )
                camera_pose = torch.cat([cam_pos, cam_quat])  # (7,)
                camera_poses_list.append(camera_pose)
            camera_poses = torch.stack(camera_poses_list, dim=0)  # (B, 7)

            # Inject objects using camera_poses
            # NOTE: Object positions stay in world frame - no flipping needed!
            # The world_to_camera_transform in DepthAugmentor handles the transformation
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

            # # Debug: collect frames for video
            # if not hasattr(self, '_debug_frames'):
            #     self._debug_frames = []
            #     self._debug_frame_count = 0

            # # Save every frame (keep as RGB for render_and_save)
            # rgb_np = augmented_rgb[0].detach().cpu().numpy()
            # frame = (rgb_np * 255).astype('uint8')
            # self._debug_frames.append(frame)

            # self._debug_frame_count += 1

            # # Save video after 30 frames (quick test)
            # if len(self._debug_frames) >= 30:
            #     print(f"[DEBUG] Saving video with {len(self._debug_frames)} frames...")
            #     video_path = '/home/irislab/ke/GRaD_Dynamic_onboard/debug_dynamic_cylinder.mp4'
            #     from controller.nav_helpers import render_and_save
            #     render_and_save(self._debug_frames, video_path, fps=25)
            #     print(f"[DEBUG] Video saved to {video_path}")
            #     assert False, f"Debug video saved to {video_path} - stopping to verify"
            # # debug code for veryfying dynamic support

            # Re-process visual features with augmented data
            self._update_visual_features(augmented_depth, augmented_rgb)

    def _get_camera_pose_from_torso(self, torso_pos, torso_quat):
        """
        Transform torso/drone pose to camera pose.
        From original drone_dynamic_expert.py.

        Drone/World frame: X forward, Y left, Z up
        Camera frame: X right, Y down, Z forward
        """
        # Camera position is the same as drone position (no offset)
        camera_pos = torso_pos.clone()

        # Apply drone-to-camera rotation transformation
        quat_np = torso_quat.detach().cpu().numpy()
        R_drone = R.from_quat(quat_np).as_matrix()
        R_drone = torch.from_numpy(R_drone).to(torso_quat.device).float()

        # Rotation matrices for drone-to-camera transformation
        sin90 = 1.0
        cos90 = 0.0
        R_90_y = torch.tensor([
            [cos90, 0, sin90],
            [0, 1, 0],
            [-sin90, 0, cos90]
        ], dtype=torch.float32, device=torso_quat.device)

        sin90 = -1.0
        cos90 = 0.0
        R_neg90_z = torch.tensor([
            [cos90, -sin90, 0],
            [sin90, cos90, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=torso_quat.device)

        # Apply transformation to get camera orientation in world frame
        # R_camera = R_drone.T @ R_90_y @ R_neg90_z TODO
        R_camera = R_drone @ R_90_y @ R_neg90_z

        # Convert rotation matrix back to quaternion
        R_camera_np = R_camera.detach().cpu().numpy()
        r_camera = R.from_matrix(R_camera_np)
        camera_quat_np = r_camera.as_quat()  # [x,y,z,w] format
        camera_quat = torch.from_numpy(camera_quat_np).to(torso_quat.device).float()

        return camera_pos, camera_quat

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
