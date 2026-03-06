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

        # Collision tracking for reward integration
        self.collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.min_dist_to_dynamic = torch.full((self.num_envs,), float('inf'), device=self.device)

        # Phase 2 curriculum tracking
        self.current_phase = 1  # 1 = static only, 2 = dynamic with danger-aware rewards
        self.dynamic_spawn_prob = 0.0  # Controlled by GradNavDynamic curriculum

        # Temporal danger tracking (dynamic object distance-based, NOT general depth)
        self.prev_dynamic_dist = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.danger_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # === Metrics storage for logging ===
        self._metrics = {
            # Distance metrics
            'min_dist_to_dynamic': torch.zeros(self.num_envs, device=self.device),
            'depth_dist': torch.zeros(self.num_envs, device=self.device),
            'delta_d': torch.zeros(self.num_envs, device=self.device),
            'd_dot': torch.zeros(self.num_envs, device=self.device),
            'ttc': torch.full((self.num_envs,), float('inf'), device=self.device),
            # Danger flags
            'danger_flag': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'collision_flag': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'approaching': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            # Velocity
            'forward_vel': torch.zeros(self.num_envs, device=self.device),
            # Reward components
            'dynamic_obst_reward': torch.zeros(self.num_envs, device=self.device),
            'r_retreat': torch.zeros(self.num_envs, device=self.device),
            'r_no_forward': torch.zeros(self.num_envs, device=self.device),
            'r_back_pen': torch.zeros(self.num_envs, device=self.device),
            'r_wait': torch.zeros(self.num_envs, device=self.device),
            'r_forward_progress': torch.zeros(self.num_envs, device=self.device),
            'phase2_reward': torch.zeros(self.num_envs, device=self.device),
            # Spawn info
            'num_active_objects': torch.zeros(self.num_envs, device=self.device),
        }

        print(f"[DynamicDroneEnv] Initialized with max_objects={max_objects}, "
              f"num_trajectory_objects={len(self.trajectory_objects)}, "
              f"dynamic_reward: threshold={self.cfg.dynamic_obst_threshold}, strength={self.cfg.dynamic_obst_strength}")

    def _spawn_dynamic_objects(self, env_ids: torch.Tensor):
        """Spawn dynamic objects in specified environments based on config.

        Supports sphere, box, and cylinder object types with full parameter support.
        Respects spawn probability for curriculum learning (Phase 2).
        """

        # Reset objects in these environments first
        self.dynamic_manager.reset_env(env_ids)

        # Phase 1: Don't spawn dynamic objects
        if self.current_phase == 1:
            return

        # Phase 2: Spawn based on probability (per environment)
        # Each environment independently rolls the dice
        spawn_mask = torch.rand(len(env_ids), device=self.device) < self.dynamic_spawn_prob

        # Filter env_ids that will get dynamic objects
        env_ids_to_spawn = env_ids[spawn_mask]
        if len(env_ids_to_spawn) == 0:
            return

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

            # Trajectory file(s) - support single file or list for random selection
            trajectory_file = obj_cfg.get('trajectory', None)
            trajectory_files = obj_cfg.get('trajectories', None)  # List of files for random selection

            for env_id in env_ids_to_spawn:
                env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id

                # Select trajectory file (random if list provided)
                selected_trajectory = None
                if trajectory_files is not None and len(trajectory_files) > 0:
                    # Randomly select from list
                    import random
                    selected_trajectory = random.choice(trajectory_files)
                elif trajectory_file is not None:
                    selected_trajectory = trajectory_file

                if selected_trajectory:
                    try:
                        # Create trajectory pattern
                        eval_time_offset = obj_cfg.get('eval_time_offset', None)
                        pattern = TrajectoryPattern(
                            trajectory_file=selected_trajectory,
                            loop=obj_cfg.get('loop', True),
                            device=self.device,
                            eval_time_offset=eval_time_offset,
                        )
                        # Get initial position from trajectory
                        init_pos = pattern.get_position(0.0)
                    except Exception as e:
                        print(f"[DynamicDroneEnv] Failed to load trajectory {selected_trajectory}: {e}")
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
        # Get drone positions (detached to avoid gradient flow through dynamic obstacle reward)
        drone_pos = self.state_joint_q[:, 0:3].detach()

        # Compute distances to dynamic objects (no gradients needed)
        with torch.no_grad():
            self.min_dist_to_dynamic = self.dynamic_manager.get_distances_to_point(drone_pos)

        # Check for collisions (distance < 0 means overlap)
        collision_threshold = self.dynamic_objects_cfg.get('collision_threshold', 0.3)
        self.collision_buf = self.min_dist_to_dynamic < collision_threshold

    def _compute_danger_info(self):
        """Compute temporal danger info for Phase 2 reward.

        Uses DYNAMIC OBJECT distance (self.min_dist_to_dynamic) to detect approaching dynamic obstacles.
        This is critical: we must NOT use general depth (self.depth_list) because that would trigger
        danger-aware rewards when approaching STATIC obstacles (gates, walls), causing the drone
        to deviate from the trajectory even without dynamic obstacles nearby.

        Wrapped in no_grad to avoid gradient flow (consistent with _check_dynamic_collisions).

        Returns:
            danger_t: (num_envs,) bool tensor - True if approaching dynamic object + risky
            delta_d: (num_envs,) float tensor - distance change to dynamic object (d_t - d_{t-1})
        """
        with torch.no_grad():
            # Get current distance to DYNAMIC objects (not general depth!)
            # self.min_dist_to_dynamic is updated in _check_dynamic_collisions()
            # When no dynamic objects are active, this is inf, so danger_t will be False
            d_t = self.min_dist_to_dynamic.detach()  # (num_envs,)

            # Check for valid previous distance (skip first step after reset where prev=inf)
            valid_prev = torch.isfinite(self.prev_dynamic_dist)

            # Also check current distance is finite (dynamic object exists)
            valid_curr = torch.isfinite(d_t)

            # Compute distance change (only valid where both prev and curr are finite)
            delta_d = torch.where(
                valid_prev & valid_curr,
                d_t - self.prev_dynamic_dist,
                torch.zeros_like(d_t)  # No change if no valid comparison
            )

            # Compute distance rate (d_dot = delta_d / dt)
            d_dot = delta_d / self.sim_dt

            # Compute TTC (only meaningful when approaching, i.e., d_dot < 0)
            eps = 1e-6
            ttc = torch.where(
                d_dot < -eps,
                d_t / (-d_dot + eps),
                torch.full_like(d_t, float('inf'))
            )

            # Danger gating condition:
            # - Must have valid previous distance
            # - Must have valid current distance (dynamic object exists)
            # - Must be approaching (d_dot < 0)
            # - Must be close enough (d_t < threshold)
            # - Must have low TTC
            approaching = (d_dot < -eps) & valid_prev & valid_curr
            close_enough = d_t < self.cfg.danger_dist_threshold
            low_ttc = ttc < self.cfg.danger_ttc_threshold

            danger_t = approaching & close_enough & low_ttc

            # Update previous distance for next step (no clone needed, already detached)
            self.prev_dynamic_dist = d_t
            self.danger_flag = danger_t

            # Store metrics for logging (all detached to avoid graph retention)
            self._metrics['depth_dist'] = d_t.detach()  # Now this is dynamic object dist
            self._metrics['delta_d'] = delta_d.detach()
            self._metrics['d_dot'] = d_dot.detach()
            self._metrics['ttc'] = ttc.detach()
            self._metrics['approaching'] = approaching.detach()
            self._metrics['danger_flag'] = danger_t.detach()

        return danger_t, delta_d

    def _compute_forward_velocity(self):
        """Compute forward velocity (velocity projected onto heading direction).

        Detached from gradient graph (consistent with _check_dynamic_collisions).

        When use_pred_vel_in_obs is enabled, uses blended velocity (same as observation)
        to ensure consistency between what policy sees and what reward uses.

        Returns:
            forward_vel: (num_envs,) float tensor - forward velocity (positive = forward, negative = backward)
        """
        from utils.rotation import quaternion_yaw_forward

        with torch.no_grad():
            # Get linear velocity (detached to avoid gradient flow)
            # state_joint_qd structure: [ang_vel (0:3), lin_vel (3:6)]
            gt_lin_vel = self.state_joint_qd[:, 3:6].detach()  # (num_envs, 3) world frame

            # Use blended velocity when use_pred_vel_in_obs is enabled (consistency with obs)
            if self.vel_net_enabled and self.use_pred_vel_in_obs:
                # Get blended velocity (same as used in observation)
                # Note: vel_net_pred_vel is in vel_frame (body or world)
                if self.vel_frame == 'world':
                    lin_vel = self.get_blended_velocity(gt_lin_vel)
                else:
                    # If vel_frame is body, we need world frame for heading projection
                    # Use GT world velocity blended with pred (transformed if needed)
                    lin_vel = self.get_blended_velocity(gt_lin_vel)
            else:
                lin_vel = gt_lin_vel

            # Get heading direction from quaternion (detached)
            torso_quat = self.state_joint_q[:, 3:7].detach()  # (x, y, z, w)
            heading_2d = quaternion_yaw_forward(torso_quat)  # (num_envs, 2)

            # Project velocity onto heading (2D)
            forward_vel = (lin_vel[:, :2] * heading_2d).sum(dim=1)

        return forward_vel

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

            # Reset temporal danger tracking for reset environments
            self.prev_dynamic_dist[spawn_env_ids] = float('inf')
            self.danger_flag[spawn_env_ids] = False

        # Call parent reset (this will call calculateObservations -> process_GS_data)
        obs = super().reset(env_ids, force_reset)

        # Replace viz_recorder's last frame with augmented images (for correct video output)
        if self.visualize and self.use_dynamic_objects and hasattr(self, 'augmented_rgb'):
            from torchvision.transforms import Resize
            img_transform = Resize((360, 640), antialias=True)

            # Replace RGB frame
            augmented_rgb_frame = torch.permute(self.augmented_rgb[0], (2, 0, 1))
            augmented_rgb_transformed = img_transform(augmented_rgb_frame)

            # Overlay velocity text on augmented image
            lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6]
            velo_data = lin_vel.clone().detach().cpu().numpy()
            vx, vy, vz = velo_data[0, 0], velo_data[0, 1], velo_data[0, 2]
            vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            text_lines = [
                f"Vel: [{vx:.2f}, {vy:.2f}, {vz:.2f}] m/s",
                f"|V|: {vel_mag:.2f} m/s",
            ]
            augmented_rgb_transformed = self.viz_recorder._overlay_text_on_image(
                augmented_rgb_transformed, text_lines
            )

            self.viz_recorder.img_record[-1] = augmented_rgb_transformed

            # Replace depth frame
            augmented_depth_np = self.augmented_depth.clone().detach().cpu().numpy()
            self.viz_recorder.depth_record[-1] = augmented_depth_np[0] / 2

        return obs

    def step(self, actions, vae_info):
        """Step with dynamic object update."""
        # Update dynamic objects before physics step
        if self.use_dynamic_objects:
            self._update_dynamic_objects()

        # Call parent step (which calls calculateReward -> _check_dynamic_collisions)
        result = super().step(actions, vae_info)

        # Replace viz_recorder's last frame with augmented images (for correct video output)
        if self.visualize and self.use_dynamic_objects and hasattr(self, 'augmented_rgb'):
            from torchvision.transforms import Resize
            img_transform = Resize((360, 640), antialias=True)

            # Replace RGB frame
            augmented_rgb_frame = torch.permute(self.augmented_rgb[0], (2, 0, 1))
            augmented_rgb_transformed = img_transform(augmented_rgb_frame)

            # Overlay velocity text on augmented image
            lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6]
            velo_data = lin_vel.clone().detach().cpu().numpy()
            vx, vy, vz = velo_data[0, 0], velo_data[0, 1], velo_data[0, 2]
            vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            text_lines = [
                f"Vel: [{vx:.2f}, {vy:.2f}, {vz:.2f}] m/s",
                f"|V|: {vel_mag:.2f} m/s",
            ]
            augmented_rgb_transformed = self.viz_recorder._overlay_text_on_image(
                augmented_rgb_transformed, text_lines
            )

            self.viz_recorder.img_record[-1] = augmented_rgb_transformed

            # Replace depth frame
            augmented_depth_np = self.augmented_depth.clone().detach().cpu().numpy()
            self.viz_recorder.depth_record[-1] = augmented_depth_np[0] / 2

        return result

    def calculateReward(self):
        """Calculate rewards including dynamic obstacle reward.

        Extends parent reward with dynamic obstacle avoidance reward.
        Uses same pattern as static obstacle reward in env_utils/reward.py.
        """
        # First check dynamic collisions to update min_dist_to_dynamic
        if self.use_dynamic_objects:
            self._check_dynamic_collisions()

        # Call parent reward calculation (static obstacles, waypoints, etc.)
        super().calculateReward()

        # Add dynamic obstacle penalty (penalize being close to dynamic obstacles)
        if self.use_dynamic_objects:
            # Penalty that increases as drone gets closer to dynamic obstacle
            # At threshold distance: penalty = 0
            # At distance 0: penalty = -threshold * strength
            # This encourages staying away from dynamic obstacles
            dynamic_obst_penalty = torch.where(
                self.min_dist_to_dynamic < self.cfg.dynamic_obst_threshold,
                -self.cfg.dynamic_obst_strength * (self.cfg.dynamic_obst_threshold - self.min_dist_to_dynamic),
                torch.zeros_like(self.min_dist_to_dynamic)
            )

            # Add to reward buffer
            self.rew_buf = self.rew_buf + dynamic_obst_penalty

            # Store metrics for logging (all detached to avoid graph retention)
            self._metrics['min_dist_to_dynamic'] = self.min_dist_to_dynamic.detach()
            self._metrics['collision_flag'] = self.collision_buf.detach()
            self._metrics['dynamic_obst_reward'] = dynamic_obst_penalty.detach()  # Now a penalty
            self._metrics['num_active_objects'] = self.dynamic_manager.active.sum(dim=1).float().detach()

            # Phase 2: Add danger-aware retreat rewards
            if self.current_phase == 2:
                # Compute danger info (uses depth-based distance)
                danger_t, delta_d = self._compute_danger_info()
                forward_vel = self._compute_forward_velocity()

                # (1) Retreat reward: encourage increasing distance when danger
                # delta_d > 0 means distance increased (good when in danger)
                # Clamp delta_d to reasonable range to prevent reward explosion
                delta_d_clamped = torch.clamp(delta_d, min=-1.0, max=1.0)
                r_retreat = self.cfg.k_retreat * danger_t.float() * delta_d_clamped

                # (2) No-forward penalty: discourage forward motion when danger
                # Clamp forward_vel to reasonable range
                forward_vel_clamped = torch.clamp(forward_vel, min=-5.0, max=5.0)
                r_no_forward = -self.cfg.k_no_forward * danger_t.float() * torch.clamp(forward_vel_clamped, min=0)

                # (3) Backward penalty: small regularization to prevent always reversing
                r_back_pen = -self.cfg.k_backward * torch.clamp(-forward_vel_clamped, min=0)

                # (4) Continuous velocity-danger penalty: penalize velocity proportional to danger
                # Replaces binary wait reward — gives gradient at ALL velocities
                # Slowing from 2.0->1.0 m/s is rewarded, not just crossing a threshold
                if self.vel_net_enabled and self.use_pred_vel_in_obs:
                    gt_vel = self.state_joint_qd[:, 3:6].detach()
                    blended_vel = self.get_blended_velocity(gt_vel)
                    vel_magnitude = torch.norm(blended_vel, dim=1)
                else:
                    vel_magnitude = torch.norm(self.state_joint_qd[:, 3:6].detach(), dim=1)
                r_wait = -self.cfg.k_wait * danger_t.float() * vel_magnitude

                # Combine Phase 2 rewards
                phase2_reward = r_retreat + r_no_forward + r_back_pen + r_wait

                # (4) Forward progress reward: encourage moving forward when SAFE
                # Only applies when no danger detected (obstacle far away)
                safe_t = ~danger_t  # Safe when NOT in danger
                # Also check if obstacle is beyond safe distance threshold
                safe_dist = self.min_dist_to_dynamic > self.cfg.safe_dist_threshold
                is_safe = safe_t | safe_dist  # Safe if no danger OR obstacle far away

                # Reward forward velocity when safe (penalize being slow when safe)
                # forward_vel > min_forward_vel gets positive reward
                # forward_vel < min_forward_vel gets negative reward (idle penalty)
                forward_progress = forward_vel_clamped - self.cfg.min_forward_vel
                r_forward_progress = self.cfg.forward_progress_weight * is_safe.float() * forward_progress

                # Add forward progress reward to phase2
                phase2_reward = phase2_reward + r_forward_progress

                # Replace any NaN/inf with 0 (safety check)
                phase2_reward = torch.nan_to_num(phase2_reward, nan=0.0, posinf=0.0, neginf=0.0)

                # Add Phase 2 rewards
                self.rew_buf = self.rew_buf + phase2_reward

                # Store Phase 2 metrics for logging
                self._metrics['forward_vel'] = forward_vel.detach()
                self._metrics['r_retreat'] = r_retreat.detach()
                self._metrics['r_no_forward'] = r_no_forward.detach()
                self._metrics['r_back_pen'] = r_back_pen.detach()
                self._metrics['r_wait'] = r_wait.detach()
                self._metrics['r_forward_progress'] = r_forward_progress.detach()
                self._metrics['phase2_reward'] = phase2_reward.detach()

    def process_GS_data(self, depth_list, rgb_img):
        """Process GS data and inject dynamic objects."""
        # Call parent to get visual features
        super().process_GS_data(depth_list, rgb_img)

        # Inject dynamic objects into depth and RGB
        # Using camera_poses approach like original drone_dynamic_expert.py
        # Wrapped in no_grad to avoid unnecessary gradient computation
        if self.use_dynamic_objects:
            with torch.no_grad():
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

            # # Debug: collect frames for video (RGB + Depth)
            # if not hasattr(self, '_debug_rgb_frames'):
            #     self._debug_rgb_frames = []
            #     self._debug_depth_frames = []
            #     self._debug_frame_count = 0

            # # Save RGB frame
            # rgb_np = augmented_rgb[0].detach().cpu().numpy()
            # rgb_frame = (rgb_np * 255).astype('uint8')
            # self._debug_rgb_frames.append(rgb_frame)

            # # Save Depth frame (normalize to 0-255 for visualization)
            # depth_np = augmented_depth[0, :, :, 0].detach().cpu().numpy()  # (H, W)
            # # Clip and normalize depth for visualization (0-10m range)
            # depth_clipped = np.clip(depth_np, 0, 10)
            # depth_normalized = (depth_clipped / 10 * 255).astype('uint8')
            # # Convert to RGB (grayscale -> 3 channel)
            # depth_frame = np.stack([depth_normalized] * 3, axis=-1)
            # self._debug_depth_frames.append(depth_frame)

            # self._debug_frame_count += 1

            # # Save videos after 100 frames
            # if len(self._debug_rgb_frames) >= 50:
            #     print(f"[DEBUG] Saving videos with {len(self._debug_rgb_frames)} frames...")
            #     from controller.nav_helpers import render_and_save

            #     # Save RGB video
            #     rgb_path = '/home/irislab/ke/GRaD_Dynamic_onboard/debug_dynamic_rgb.mp4'
            #     render_and_save(self._debug_rgb_frames, rgb_path, fps=25)
            #     print(f"[DEBUG] RGB video saved to {rgb_path}")

            #     # Save Depth video
            #     depth_path = '/home/irislab/ke/GRaD_Dynamic_onboard/debug_dynamic_depth.mp4'
            #     render_and_save(self._debug_depth_frames, depth_path, fps=25)
            #     print(f"[DEBUG] Depth video saved to {depth_path}")

            #     assert False, f"Debug videos saved - stopping to verify"
            # # debug code for verifying dynamic support

            # Re-process visual features with augmented data
            self._update_visual_features(augmented_depth, augmented_rgb)

    def _conjugate_quat(self, quat):
        """
        Conjugate quaternion tensor [x, y, z, w] -> [x, -y, -z, w]
        This fixes the handedness mismatch between simulation and rendering.
        Only negates Y and Z components.
        """
        result = quat.clone().detach()
        result[0] = quat[0]
        result[1] = -quat[1]
        result[2] = -quat[2]
        # result[3] = quat[3]  # w stays the same
        return result

    def _get_camera_pose_from_torso(self, torso_pos, torso_quat):
        """
        Transform torso/drone pose to camera pose.
        From original drone_dynamic_expert.py.

        Drone/World frame: X forward, Y left, Z up
        Camera frame: X right, Y down, Z forward
        """
        # Camera position is the same as drone position (no offset)
        camera_pos = torso_pos.clone().detach()

        # Conjugate quaternion to fix handedness mismatch
        torso_quat_conj = self._conjugate_quat(torso_quat)

        # Apply drone-to-camera rotation transformation
        quat_np = torso_quat_conj.detach().cpu().numpy()
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

        # _, H, _, _ = depth_list.shape
        # depth_list_up = depth_list[:, 0:int(H/2), :, :]
        self.depth_list = torch.abs(torch.amin(depth_list, dim=(1, 2, 3))).unsqueeze(1).to(device=self.device)

        # For policy (SqueezeNet): use 224x224 resized images
        visual_tensor = rgb_img.permute(0, 3, 1, 2)
        resize = nn.AdaptiveAvgPool2d((224, 224))
        visual_tensor_resized = resize(visual_tensor).to(self.device)
        self.visual_info = self.visual_net(visual_tensor_resized).detach()

        # For vel_net (DualEncoder): use RAW images (no resize) to match training
        if self.vel_net_enabled and self.dual_encoder is not None:
            with torch.no_grad():
                rgb_raw = rgb_img.permute(0, 3, 1, 2).to(self.device)
                depth_raw = depth_list.permute(0, 3, 1, 2).to(self.device)
                self.rgb_feat_vel, self.depth_feat_vel = self.dual_encoder(rgb_raw, depth_raw)

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

    def get_dynamic_metrics(self):
        """Get aggregated metrics for wandb logging.

        Returns dict with mean/min/max/count values across all environments.
        Only includes finite values in aggregations.
        """
        if not self.use_dynamic_objects:
            return {}

        metrics = {}

        # Helper to compute stats for a tensor, handling inf values
        def compute_stats(tensor, name, is_bool=False):
            if is_bool:
                metrics[f'{name}_count'] = tensor.sum().item()
                metrics[f'{name}_ratio'] = tensor.float().mean().item()
            else:
                finite_mask = torch.isfinite(tensor)
                if finite_mask.any():
                    finite_vals = tensor[finite_mask]
                    metrics[f'{name}_mean'] = finite_vals.mean().item()
                    metrics[f'{name}_min'] = finite_vals.min().item()
                    metrics[f'{name}_max'] = finite_vals.max().item()
                else:
                    metrics[f'{name}_mean'] = 0.0
                    metrics[f'{name}_min'] = 0.0
                    metrics[f'{name}_max'] = 0.0

        # Distance metrics
        compute_stats(self._metrics['min_dist_to_dynamic'], 'dyn/min_dist')
        compute_stats(self._metrics['depth_dist'], 'dyn/depth_dist')
        compute_stats(self._metrics['delta_d'], 'dyn/delta_d')
        compute_stats(self._metrics['d_dot'], 'dyn/d_dot')
        compute_stats(self._metrics['ttc'], 'dyn/ttc')

        # Boolean flags
        compute_stats(self._metrics['danger_flag'], 'dyn/danger', is_bool=True)
        compute_stats(self._metrics['collision_flag'], 'dyn/collision', is_bool=True)
        compute_stats(self._metrics['approaching'], 'dyn/approaching', is_bool=True)

        # Velocity
        compute_stats(self._metrics['forward_vel'], 'dyn/forward_vel')

        # Dynamic reward components
        compute_stats(self._metrics['dynamic_obst_reward'], 'dyn/r_obst')
        compute_stats(self._metrics['r_retreat'], 'dyn/r_retreat')
        compute_stats(self._metrics['r_no_forward'], 'dyn/r_no_forward')
        compute_stats(self._metrics['r_wait'], 'dyn/r_wait')
        compute_stats(self._metrics['r_forward_progress'], 'dyn/r_forward_progress')
        compute_stats(self._metrics['r_back_pen'], 'dyn/r_back_pen')
        compute_stats(self._metrics['phase2_reward'], 'dyn/r_phase2')

        # Base reward components (from reward.py)
        if hasattr(self, 'reward_components') and self.reward_components is not None:
            for name, value in self.reward_components.items():
                if isinstance(value, torch.Tensor):
                    compute_stats(value.detach(), f'rew/{name}')
                else:
                    metrics[f'rew/{name}'] = value

        # Total reward
        compute_stats(self.rew_buf.detach(), 'rew/total')

        # Spawn info
        metrics['dyn/num_active_objects'] = self._metrics['num_active_objects'].mean().item()

        # Curriculum info
        metrics['dyn/current_phase'] = self.current_phase
        metrics['dyn/spawn_prob'] = self.dynamic_spawn_prob

        return metrics

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
