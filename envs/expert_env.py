"""Expert environment for policy training.

Inherits from SimpleDroneEnv, adds:
- obs_buf (57 dims) and privilege_obs_buf (67 dims)
- Action delay processing
- Domain randomization
- Reward calculation
- VAE integration

Source: drone_long_traj.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.drone_env import SimpleDroneEnv
from envs.env_utils import EnvConfig, get_config, calculate_reward, check_termination, VisualizationRecorder
from envs.assets.quadrotor_dynamics import QuadrotorSimulator
from utils import torch_utils as tu
from utils.rotation import quaternion_to_euler, quaternion_yaw_forward
from utils.hist_obs_buffer import ObsHistBuffer
from utils.time_report import TimeReport
from models.policy.squeeze_net import VisualPerceptionNet


class ExpertDroneEnv(SimpleDroneEnv):
    """
    Expert environment for policy training.

    Preserves original structure from drone_long_traj.py.
    """

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
    ):
        # Initialize base SimpleDroneEnv (like original line 60)
        super().__init__(
            map_name=map_name,
            device=device,
            num_envs=num_envs,
            episode_length=episode_length,
        )

        # Agent name for logging (line 50)
        self.agent_name = 'expert_drone_env'

        # Time report for profiling (lines 64-67)
        self.time_report = TimeReport()
        self.time_report.add_timer("dynamic simulation")
        self.time_report.add_timer("3D GS inference")
        self.time_report.add_timer("point cloud collision check")

        # Hyperparameters (lines 51-56)
        if env_hyper is None:
            env_hyper = {}
        self.num_history = env_hyper.get('HISTORY_BUFFER_NUM', 5)
        self.num_latent = env_hyper.get('LATENT_VECT_NUM', 24)
        self.visual_feature_size = env_hyper.get('SINGLE_VISUAL_INPUT_SIZE', 16)
        self.num_privilege_obs = 27 + self.visual_feature_size + self.num_latent
        self.num_latent_obs = 17 + self.num_latent + self.visual_feature_size
        self.num_obs = 17 + self.num_latent + self.visual_feature_size
        self.num_actions = 4

        # Flags
        self.visualize = render
        self.no_grad = no_grad
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.domain_randomization = False

        # Config
        self.cfg = get_config(map_name)

        # Load reward parameters from config (lines 172-195 in original)
        self.heading_strength = self.cfg.heading_strength
        self.obstacle_strength = self.cfg.obstacle_strength
        self.obst_collision_limit = self.cfg.obst_collision_limit
        self.action_penalty = self.cfg.action_penalty
        self.smooth_penalty = self.cfg.smooth_penalty
        self.target_factor = self.cfg.target_factor
        self.pose_penalty = self.cfg.pose_penalty
        self.obst_threshold = self.cfg.obst_threshold
        self.survive_reward = self.cfg.survive_reward
        self.height_penalty = self.cfg.height_penalty
        self.waypoint_strength = self.cfg.waypoint_strength
        self.out_map_penalty = self.cfg.out_map_penalty
        self.action_change_penalty = self.cfg.action_change_penalty
        self.start_height = self.cfg.start_pos[2]
        self.target_height = self.cfg.target_height

        # Domain randomization params (lines 196-205 in original)
        self.min_mass = 0.8
        self.mass_range = 0.4
        self.min_thrust = 0.8
        self.thrust_range = 0.4

        # Action delay factors (lines 206-207 in original)
        self.br_delay_factor = self.cfg.br_delay_factor
        self.thrust_delay_factor = self.cfg.thrust_delay_factor

        # Initialize simulation
        self.init_sim()

        # Setup navigation goals (lines 116-131)
        self._setup_navigation()

        # Visual perception net (line 217)
        self.visual_net = VisualPerceptionNet(visual_feature_size=self.visual_feature_size).to(self.device)

        # Observation history buffer for VAE (lines 220-224)
        self.obs_hist_buf = ObsHistBuffer(
            batch_size=self.num_envs,
            vector_dim=self.num_obs,
            buffer_size=self.num_history,
            device=self.device,
        )

        # Pack hyperparameters for record in wandb (lines 227-251)
        self.hyper_parameter = {
            "heading_strength": self.heading_strength,
            "obstacle_strength": self.obstacle_strength,
            "obst_collision_limit": self.obst_collision_limit,
            "action_penalty": self.action_penalty,
            "smooth_penalty": self.smooth_penalty,
            "target_factor": self.target_factor,
            "pose_penalty": self.pose_penalty,
            "obst_threshold": self.obst_threshold,
            "survive_reward": self.survive_reward,
            "height_penalty": self.height_penalty,
            "sim_dt": self.dt,
            "waypoint_strength": self.waypoint_strength,
            "out_map_penalty": self.out_map_penalty,
            "action_change_penalty": self.action_change_penalty,
            "min_mass": self.min_mass,
            "mass_range": self.mass_range,
            "min_thrust": self.min_thrust,
            "thrust_range": self.thrust_range,
            "br_delay_factor": self.br_delay_factor,
            "thrust_delay_factor": self.thrust_delay_factor,
            "start_height": self.start_height,
            "target_height": self.target_height,
        }

        # Initialize observation buffers
        self.obs_buf = torch.zeros([self.num_envs, self.num_obs], device=self.device)
        self.privilege_obs_buf = torch.zeros([self.num_envs, self.num_privilege_obs], device=self.device)
        self.vae_obs_buf = torch.zeros([self.num_envs, self.num_obs], device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # GS rendering control
        self.gs_count = 0
        self.gs_freq = 1
        self.depth_list = torch.zeros([self.num_envs, 1], device=self.device)
        self.visual_info = torch.zeros([self.num_envs, self.visual_feature_size], device=self.device)

        # Visualization setup
        if self.visualize:
            from utils.common import get_time_stamp
            self.time_stamp = get_time_stamp()
            curr_path = os.getcwd()
            self.save_path = f'{curr_path}/output/expert_env/{self.map_name}/{self.time_stamp}' # TODO: Change path
            self.viz_recorder = VisualizationRecorder(self.save_path, episode_length, device)

    def init_sim(self):
        """Initialize simulation parameters. Source: lines 283-371"""
        self.dt = 0.05
        self.sim_dt = self.dt
        self.sim_time = 0.0
        self.num_frames = 0

        # Unit tensors (lines 291-293)
        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        # Start rotation (lines 295-301)
        self.start_rot = np.array(self.cfg.start_rotation)
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # Helper vectors (lines 304-306)
        self.up_vec = self.z_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # SysID parameters
        self.min_mass = 1.0
        self.mass_range = 0.2
        self.min_thrust = 24.0
        self.thrust_range = 4.0
        self.obs_noise_level = self.cfg.obs_noise_level
        self.br_delay_factor = self.cfg.br_delay_factor
        self.thrust_delay_factor = self.cfg.thrust_delay_factor
        self.br_action_strength = self.cfg.br_action_strength
        self.thrust_action_strength = self.cfg.thrust_action_strength
        self.start_height = self.cfg.start_pos[2]
        self.target_height = self.cfg.target_height
        self.init_inertia = [0.01, 0.012, 0.025]
        self.init_kp = [1.0, 1.2, 2.5]
        self.init_kd = [0.001, 0.001, 0.002]

        # Dynamics parameters (lines 309-316)
        self.mass = torch.full((self.num_envs,), self.min_mass + 0.5 * self.mass_range, device=self.device)
        self.max_thrust = torch.full((self.num_envs,), self.min_thrust + 0.5 * self.thrust_range, device=self.device)
        self.hover_thrust = self.mass * 9.81
        inertia = torch.tensor(self.init_inertia, device=self.device)
        self.inertia = torch.diag(inertia).unsqueeze(0).repeat(self.num_envs, 1, 1)
        Kp = torch.tensor(self.init_kp, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        Kd = torch.tensor(self.init_kd, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # QuadrotorSimulator (lines 319-330)
        self.quad_dynamics = QuadrotorSimulator(
            mass=self.mass,
            inertia=self.inertia,
            link_length=0.15,
            Kp=Kp,
            Kd=Kd,
            freq=200.0,
            max_thrust=self.max_thrust,
            total_time=self.sim_dt,
            rotor_noise_std=0.01,
            br_noise_std=0.01,
            device=self.device,
        )

        # Start state (lines 332-357)
        self.start_body_rate = [0., 0., 0.]
        self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
        self.control_base = self.start_norm_thrust[0]
        self.start_action = self.start_body_rate + self.start_norm_thrust

        self.start_pos = torch.tensor(self.cfg.start_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.start_pos_backup = self.start_pos.clone()
        self.start_joint_q = tu.to_torch(self.start_action, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_action, device=self.device)
        self.prev_thrust = torch.ones([self.num_envs, 1], device=self.device) * (self.hover_thrust / self.max_thrust).unsqueeze(-1)

        # Action buffers (lines 361-363)
        self.actions = self.start_joint_q.repeat(self.num_envs, 1).clone()
        self.prev_actions = self.actions.clone()
        self.prev_prev_actions = self.actions.clone()

        # State tensors (lines 365-371)
        self.state_joint_q = torch.zeros([self.num_envs, 7], device=self.device)  # pos + quat
        self.state_joint_qd = torch.zeros([self.num_envs, 6], device=self.device)  # ang_vel + lin_vel
        self.state_joint_qdd = torch.zeros([self.num_envs, 6], device=self.device)  # ang_acc + lin_acc

        # VAE variables (lines 369-370)
        self.latent_vect = torch.zeros([self.num_envs, self.num_latent], device=self.device, dtype=torch.float)
        self.prev_lin_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

    def _setup_navigation(self):
        """Setup navigation goals. Source: lines 116-131"""
        # Convert config waypoints to tensor
        self.reward_wp = torch.tensor(self.cfg.waypoints, device=self.device)
        self.target = torch.tensor(self.cfg.target_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.target_xy = self.target[:, 0:2]

        # Reference trajectory (plan using traj_planner if available)
        # For now, use waypoints as ref_traj
        traj_wp = self.reward_wp + self.point_cloud_offset[0].repeat((self.reward_wp.shape[0], 1))
        self.ref_traj = traj_wp  # Simplified - full implementation uses traj_planner

    def step(self, actions, vae_info):
        """
        Forward simulation step.

        Source: lines 374-461

        Args:
            actions: (N, 4) tensor - [body_rate_r, body_rate_p, body_rate_y, thrust]
            vae_info: (N, num_latent) tensor - VAE latent vector

        Returns:
            obs_buf, privilege_obs_buf, obs_hist, obs_vel, rew_buf, reset_buf, extras
        """
        self.latent_vect = vae_info.clone().detach()

        # Prepare VAE data (lines 378-380)
        self.obs_hist_buf.update(self.vae_obs_buf)
        obs_hist = self.obs_hist_buf.get_concatenated().clone().detach()
        obs_vel = self.privilege_obs_buf[:, 3:6].clone().detach()

        # Process actions (lines 382-396)
        actions = actions.view((self.num_envs, self.num_actions))
        prev_body_rate = self.prev_actions[:, 0:3].clone().detach()
        body_rate_cols = self.br_delay_factor * (torch.clip(actions[:, 0:3], -1., 1.) * self.br_action_strength) + (1 - self.br_delay_factor) * prev_body_rate
        body_rate_cols = torch.clip(body_rate_cols, -0.5, 0.5)

        prev_thrust = self.prev_actions[:, -1].unsqueeze(-1).clone().detach()
        thrust_col = self.thrust_delay_factor * ((torch.clip(actions[:, 3:], -1., 1.) + 1) * self.thrust_action_strength) + (1 - self.thrust_delay_factor) * prev_thrust

        actions = torch.cat([body_rate_cols, thrust_col], dim=1)
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        self.actions = actions
        control_input = (actions[:, 0:3], actions[:, 3])

        # Get current state (lines 399-404)
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7]
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]

        # Run dynamics simulation (lines 408-421)
        new_position, new_linear_velocity, new_angular_velocity, new_quaternion, new_linear_acceleration, new_angular_acceleration = \
            self.quad_dynamics.run_simulation(
                position=torso_pos,
                velocity=lin_vel,
                orientation=torso_quat[:, [3, 0, 1, 2]],  # (w, x, y, z)
                angular_velocity=ang_vel,
                control_input=control_input
            )

        if self.no_grad:
            new_position = new_position.detach()
            new_quaternion = new_quaternion.detach()
            new_linear_velocity = new_linear_velocity.detach()
            new_angular_velocity = new_angular_velocity.detach()

        # Update state (lines 424-430)
        self.state_joint_q.view(self.num_envs, -1)[:, 0:3] = new_position
        self.state_joint_q.view(self.num_envs, -1)[:, 3:7] = new_quaternion[:, [1, 2, 3, 0]].clone()  # (x,y,z,w)
        self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] = new_linear_velocity
        self.state_joint_qd.view(self.num_envs, -1)[:, 0:3] = new_angular_velocity
        self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6] = new_linear_acceleration.clone().detach()
        self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3] = new_angular_acceleration.clone().detach()

        self.sim_time += self.sim_dt
        self.reset_buf = torch.zeros_like(self.reset_buf)
        self.progress_buf += 1
        self.num_frames += 1

        # Update observations and rewards (lines 436-437)
        self.calculateObservations()
        self.calculateReward()

        # Handle resets (lines 439-451)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.privilege_obs_buf_before_reset = self.privilege_obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'privilege_obs_before_reset': self.privilege_obs_buf_before_reset,
                'episode_end': self.termination_buf,
            }

        if len(env_ids) > 0:
            self.reset(env_ids)

        # Sanity check (lines 454-459)
        if (torch.isnan(obs_hist).any() | torch.isinf(obs_hist).any()):
            print('obs hist nan')
            obs_hist = torch.nan_to_num(obs_hist, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(obs_vel).any() | torch.isinf(obs_vel).any()):
            print('obs vel nan')
            obs_vel = torch.nan_to_num(obs_vel, nan=0.0, posinf=1e3, neginf=-1e3)

        return self.obs_buf, self.privilege_obs_buf, obs_hist, obs_vel, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self, env_ids=None, force_reset=True):
        """
        Reset environments.

        Source: lines 464-535
        """
        if env_ids is None:
            if force_reset:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            # Domain randomization (lines 470-500)
            if not self.visualize and self.domain_randomization:
                num_reset_envs = len(env_ids)
                mass_with_noise = torch.rand(num_reset_envs, device=self.device) * self.mass_range + self.min_mass
                self.mass[env_ids] = mass_with_noise
                max_thrust_with_noise = torch.rand(num_reset_envs, device=self.device) * self.thrust_range + self.min_thrust
                self.max_thrust[env_ids] = max_thrust_with_noise
                inertia = torch.tensor(self.init_inertia, device=self.device)
                inertial_noise = (torch.rand(num_reset_envs, 3, device=self.device) - 0.5) * 2 * 0.2 * inertia
                randomized_inertia = inertia + inertial_noise
                self.inertia[env_ids] = torch.diag_embed(randomized_inertia)
                self.hover_thrust[env_ids] = self.mass[env_ids] * 9.81

                # Reinitialize QuadrotorSimulator
                self.quad_dynamics = QuadrotorSimulator(
                    mass=self.mass,
                    inertia=self.inertia,
                    link_length=0.15,
                    Kp=torch.tensor(self.init_kp, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    Kd=torch.tensor(self.init_kd, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    freq=200.0,
                    max_thrust=self.max_thrust,
                    total_time=self.sim_dt,
                    rotor_noise_std=0.01,
                    br_noise_std=0.01,
                    device=self.device,
                )

                self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
                self.control_base = self.start_norm_thrust[0]
                self.start_action = self.start_body_rate + self.start_norm_thrust
                self.start_joint_q = tu.to_torch(self.start_action, device=self.device)

            # Clone state (lines 504-506)
            self.state_joint_q = self.state_joint_q.clone()
            self.state_joint_qd = self.state_joint_qd.clone()
            self.state_joint_qdd = self.state_joint_qdd.clone()

            # Reset to start state (lines 509-515)
            self.state_joint_q[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.clone()
            self.state_joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.
            self.state_joint_qdd.view(self.num_envs, -1)[env_ids, :] = 0.
            self.actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
            self.prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
            self.prev_prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()

            # Stochastic initialization (lines 518-527)
            if not self.visualize and self.stochastic_init:
                self.state_joint_q.view(self.num_envs, -1)[env_ids, 0:3] += 0.5 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device=self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device=self.device) - 0.5)
                self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(
                    self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7],
                    tu.quat_from_angle_axis(angle, axis)
                )
                self.state_joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.state_joint_qdd.view(self.num_envs, -1)[env_ids, :] = 0.05 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone() + 0.05 * torch.rand(size=(len(env_ids), 4), device=self.device)
                self.prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
                self.prev_prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()

            # Clear VAE variables (line 530)
            self.latent_vect[env_ids] = torch.zeros([len(env_ids), self.num_latent], device=self.device, dtype=torch.float)

            self.progress_buf[env_ids] = 0
            self.calculateObservations()

        return self.obs_buf
    
    def process_GS_data(self, depth_list, rgb_img):
        """Process GS render output. Source: lines 580-589"""
        batch, H, W, ch = depth_list.shape
        depth_list_up = depth_list[:, 0:int(H/2), :, :]
        self.depth_list = torch.abs(torch.amin(depth_list_up, dim=(1, 2, 3))).unsqueeze(1).to(device=self.device)

        visual_tensor = rgb_img.permute(0, 3, 1, 2)
        resize = nn.AdaptiveAvgPool2d((224, 224))
        visual_tensor = resize(visual_tensor).to(self.device)
        self.visual_info = self.visual_net(visual_tensor).detach()

    def calculateObservations(self):
        """
        Calculate observation buffers.

        Source: lines 592-684
        """
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7]  # (x,y,z,w)
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6]
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]
        lin_acceleration = self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6]

        target_dirs = tu.normalize(lin_vel[:, 0:2].clone())
        up_vec = tu.quat_rotate(torso_quat.clone(), self.up_vec)
        rpy = quaternion_to_euler(torso_quat[:, [3, 0, 2, 1]])  # input is (w,x,z,y)
        heading_vec = torch.cat([torch.cos(rpy[:, 2].unsqueeze(-1)), torch.sin(rpy[:, 2].unsqueeze(-1))], dim=1)

        # GS rendering
        gs_pos = torso_pos + self.gs_origin_offset
        gs_pos[:, 1] = -gs_pos[:, 1]
        gs_pos[:, 2] = -gs_pos[:, 2]
        gs_pose = torch.cat([gs_pos, torch.zeros([self.num_envs, 3], device=self.device), torso_quat], dim=-1)
        rpy_data = rpy.clone().detach().cpu().numpy()

        if not self.visualize:
            if self.gs_count % self.gs_freq == 0:
                depth_list, rgb_img = self.gs.render(gs_pose)
                self.process_GS_data(depth_list, rgb_img)
            self.gs_count += 1

        # Visualization recording
        if self.visualize:
            depth_list, rgb_img = self.gs.render(gs_pose)
            self.process_GS_data(depth_list, rgb_img)
            self.viz_recorder.record_visualization_data(
                torso_pos, ang_vel, lin_vel, rpy,
                self.actions, rgb_img, depth_list
            )

        # Ablation variable
        latent_abalation = torch.zeros_like(self.latent_vect)

        # Adding noise (lines 628-632)
        torso_pos_noise = self.obs_noise_level * (torch.rand_like(torso_pos) - 0.5)
        lin_vel_noise = self.obs_noise_level * (torch.rand_like(lin_vel) - 0.5)
        visual_noise = self.obs_noise_level * (torch.rand_like(self.visual_info) - 0.5)
        latent_noise = self.obs_noise_level * (torch.rand_like(self.latent_vect) - 0.5)
        torso_quat_noise = self.obs_noise_level * (torch.rand_like(torso_quat) - 0.5)

        # privilege_obs_buf (lines 634-648)
        self.privilege_obs_buf = torch.cat([
            torso_pos[:, :],                                              # 0:3
            lin_vel,                                                      # 3:6 
            lin_acceleration,                                             # 6:9
            torso_quat,                                                   # 9:13
            ang_vel,                                                      # 13:16
            up_vec[:, 1:2],                                               # 16
            (heading_vec * target_dirs).sum(dim=-1).unsqueeze(-1),        # 17
            self.actions,                                                 # 18:22
            self.prev_actions,                                            # 22:26
            self.depth_list,                                              # 26
            self.visual_info,                                             # 27:43
            self.latent_vect,                                             # 43:67
        ], dim=-1)

        # obs_buf (lines 650-660)
        self.obs_buf = torch.cat([
            (torso_pos[:, 2] + torso_pos_noise[:, 2]).unsqueeze(-1),      # 0 z-pos
            (lin_vel[:, 2] + lin_vel_noise[:, 2]).unsqueeze(-1),          # 1 z-vel TODO Change to vel_net
            torso_quat + torso_quat_noise,                                # 2:6
            self.actions,                                                 # 6:10
            self.prev_actions,                                            # 10:14
            lin_vel,                                                      # 14:17 TODO Change to vel_net
            self.visual_info + visual_noise,                              # 17:33
            self.latent_vect + latent_noise,                              # 33:57
        ], dim=-1)

        # vae_obs_buf (lines 663-673)
        self.vae_obs_buf = torch.cat([
            (torso_pos[:, 2] + torso_pos_noise[:, 2]).unsqueeze(-1),      # 0 z-pos
            (lin_vel[:, 2] + lin_vel_noise[:, 2]).unsqueeze(-1),          # 1 z-vel
            torso_quat + torso_quat_noise,                                # 2:6
            self.actions,                                                 # 6:10
            self.prev_actions,                                            # 10:14
            lin_vel,                                                      # 14:17 TODO Change to vel_net
            self.visual_info + visual_noise,                              # 17:33
            latent_abalation,                                             # 33:57
        ], dim=-1)

        # Sanity check (lines 676-684)
        if (torch.isnan(self.obs_buf).any() | torch.isinf(self.obs_buf).any()):
            print("obs buf nan")
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(self.privilege_obs_buf).any() | torch.isinf(self.privilege_obs_buf).any()):
            print("privilege obs buf nan")
            self.privilege_obs_buf = torch.nan_to_num(self.privilege_obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(self.vae_obs_buf).any() | torch.isinf(self.vae_obs_buf).any()):
            print("vae obs buf nan")
            self.vae_obs_buf = torch.nan_to_num(self.vae_obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)

    def calculateReward(self):
        """
        Calculate rewards and check termination.

        Source: lines 687-795
        """
        self.rew_buf = calculate_reward(
            obs_buf=self.obs_buf,
            privilege_obs_buf=self.privilege_obs_buf,
            prev_prev_actions=self.prev_prev_actions,
            waypoints=self.reward_wp,
            target=self.target,
            ref_traj=self.ref_traj,
            point_cloud=self.point_cloud,
            cfg=self.cfg,
            start_rotation=self.start_rotation,
            point_cloud_offset=self.point_cloud_offset,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Check termination (lines 775-790)
        combined_condition = check_termination(
            obs_buf=self.obs_buf,
            privilege_obs_buf=self.privilege_obs_buf,
            progress_buf=self.progress_buf,
            cfg=self.cfg,
            early_termination=self.early_termination,
        )

        self.reset_buf = torch.where(combined_condition, torch.ones_like(self.reset_buf), self.reset_buf)

        # Visualization save (lines 792-794)
        if self.visualize:
            episode_count = self.progress_buf[0].item()
            if episode_count == (self.cfg.episode_length - 1) or combined_condition.any():
                self.viz_recorder.save_recordings(self.reward_wp, self.target)

    def clear_grad(self, checkpoint=None):
        """Clear gradients. Source: lines 538-560"""
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state_joint_q.clone()
                checkpoint['joint_qd'] = self.state_joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['prev_actions'] = self.prev_actions.clone()
                checkpoint['prev_prev_actions'] = self.prev_prev_actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()
                checkpoint['latent_vect'] = self.latent_vect.clone()
                checkpoint['prev_lin_vel'] = self.prev_lin_vel

            self.state_joint_q = checkpoint['joint_q'].clone()
            self.state_joint_qd = checkpoint['joint_qd'].clone()
            self.actions = checkpoint['actions'].clone()
            self.prev_actions = checkpoint['prev_actions'].clone()
            self.prev_prev_actions = checkpoint['prev_prev_actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()
            self.latent_vect = checkpoint['latent_vect'].clone()
            self.prev_lin_vel = checkpoint['prev_lin_vel'].clone()

    def initialize_trajectory(self):
        """Initialize trajectory. Source: lines 563-566"""
        self.clear_grad()
        self.calculateObservations()
        return self.obs_buf, self.privilege_obs_buf
    
    def get_checkpoint(self):
        """Get checkpoint. Source: lines 568-578"""
        checkpoint = {}
        checkpoint['joint_q'] = self.state_joint_q.clone()
        checkpoint['joint_qd'] = self.state_joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['prev_actions'] = self.prev_actions.clone()
        checkpoint['prev_prev_actions'] = self.prev_prev_actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()
        checkpoint['latent_vect'] = self.latent_vect.clone()
        checkpoint['prev_lin_vel'] = self.prev_lin_vel.clone()
        return checkpoint