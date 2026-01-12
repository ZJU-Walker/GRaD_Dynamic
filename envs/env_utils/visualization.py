"""Visualization utilities for policy training.

Preserves original pattern from drone_long_traj.py lines 797-904.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.transforms import Resize


class VisualizationRecorder:
    """Records and saves visualization data during episodes."""

    def __init__(self, save_path: str, episode_length: int, device: str = 'cuda:0'):
        self.save_path = save_path
        self.episode_length = episode_length
        self.device = device

        os.makedirs(save_path, exist_ok=True)

        # Initialize records (same as original lines 263-280)
        self.x_record = []
        self.y_record = []
        self.z_record = []
        self.roll_record = []
        self.pitch_record = []
        self.yaw_record = []
        self.velo_x_record = []
        self.velo_y_record = []
        self.velo_z_record = []
        self.ang_velo_x_record = []
        self.ang_velo_y_record = []
        self.ang_velo_z_record = []
        self.depth_record = []
        self.img_record = []
        self.action_record = np.zeros([episode_length, 4])
        self.episode_count = 0

        self.img_transform = Resize((360, 640), antialias=True)

    def record_visualization_data(self, torso_pos, ang_vel, lin_vel, rpy_data,
                                    actions, rgb_img, depth_list):
        """
        Record a single frame of data.

        Source: drone_long_traj.py lines 797-829

        Args:
            torso_pos: (1, 3) tensor - position
            ang_vel: (1, 3) tensor - angular velocity
            lin_vel: (1, 3) tensor - linear velocity
            rpy_data: (1, 3) tensor - roll, pitch, yaw
            actions: (1, 4) tensor - actions [r, p, y, thrust]
            rgb_img: (1, H, W, 3) tensor - RGB image
            depth_list: (1, H, W, 1) tensor - depth image
        """
        # Process image for video (lines 803-804)
        rgb_img = torch.permute(rgb_img[0], (2, 0, 1))
        img = self.img_transform(rgb_img)

        # Convert to numpy (lines 806-810)
        pos_data = torso_pos.clone().detach().cpu().numpy()
        depth_data = depth_list.clone().detach().cpu().numpy()
        action_data = actions.clone().detach().cpu().numpy()
        ang_vel_data = ang_vel.clone().detach().cpu().numpy()
        velo_data = lin_vel.clone().detach().cpu().numpy()
        rpy_np = rpy_data.clone().detach().cpu().numpy()

        # Record data (lines 812-825)
        self.x_record.append(pos_data[0, 0])
        self.y_record.append(pos_data[0, 1])
        self.z_record.append(pos_data[0, 2])
        self.roll_record.append(rpy_np[0, 0])
        self.pitch_record.append(rpy_np[0, 1])
        self.yaw_record.append(rpy_np[0, 2])
        self.velo_x_record.append(velo_data[0, 0])
        self.velo_y_record.append(velo_data[0, 1])
        self.velo_z_record.append(velo_data[0, 2])
        self.ang_velo_x_record.append(ang_vel_data[0, 0])
        self.ang_velo_y_record.append(ang_vel_data[0, 1])
        self.ang_velo_z_record.append(ang_vel_data[0, 2])
        self.depth_record.append(depth_data[0] / 2)
        self.img_record.append(img)

        # Record actions (lines 827-829)
        if self.episode_count < self.episode_length:
            self.action_record[self.episode_count, :] = action_data
        self.episode_count += 1

    def save_recordings(self, reward_wp, target):
        """
        Save all recorded data as plots and video.

        Source: drone_long_traj.py lines 832-904

        Args:
            reward_wp: (num_wp, 3) tensor - waypoints
            target: (1, 3) tensor - target position
        """
        save_path = self.save_path

        # figure z over time (lines 836-840)
        plt.figure()
        plt.plot(range(len(self.z_record)), self.z_record)
        plt.xlabel("Step")
        plt.ylabel("Z/m")
        plt.savefig(f'{save_path}/z_plot.png')

        # figure traj (lines 843-859)
        wp_np = reward_wp.clone().detach().cpu().numpy()
        target_np = target[0, :].clone().detach().cpu().numpy()
        plt.figure()
        plt.plot(self.x_record, self.y_record)
        # waypoints
        wp_num, _ = wp_np.shape
        if wp_num > 0:
            for i in range(wp_num - 1):
                circle = plt.Circle((wp_np[i][0], wp_np[i][1]), 0.3, color='b', fill=True)
                plt.gca().add_patch(circle)
            circle = plt.Circle((wp_np[-1][0], wp_np[-1][1]), 0.5, color='r', fill=True)
            plt.gca().add_patch(circle)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X/m")
        plt.ylabel("Y/m")
        plt.savefig(f'{save_path}/traj_plot.png')

        # figure pose over time (lines 862-869)
        plt.figure()
        plt.plot(range(len(self.roll_record)), self.roll_record)
        plt.plot(range(len(self.pitch_record)), self.pitch_record)
        plt.plot(range(len(self.yaw_record)), self.yaw_record)
        plt.xlabel("Step")
        plt.ylabel("rad")
        plt.legend(['r', 'p', 'y'])
        plt.savefig(f'{save_path}/pose_plot.png')

        # figure velo over time (lines 872-879)
        plt.figure()
        plt.plot(range(len(self.velo_x_record)), self.velo_x_record)
        plt.plot(range(len(self.velo_y_record)), self.velo_y_record)
        plt.plot(range(len(self.velo_z_record)), self.velo_z_record)
        plt.xlabel("Step")
        plt.ylabel("m/s")
        plt.legend(['vx', 'vy', 'vz', 'vae_vx', 'vae_vy', 'vae_vz'])
        plt.savefig(f'{save_path}/velo_plot.png')

        # figure action (lines 882-898)
        plt.figure()
        for action_id in range(3):
            plt.plot(range(np.shape(self.action_record)[0]), self.action_record[:, action_id], label=f'rotor {action_id}')
        plt.plot(range(len(self.ang_velo_x_record)), self.ang_velo_x_record, linestyle='dashed')
        plt.plot(range(len(self.ang_velo_y_record)), self.ang_velo_y_record, linestyle='dashed')
        plt.plot(range(len(self.ang_velo_z_record)), self.ang_velo_z_record, linestyle='dashed')
        plt.xlabel('Step')
        plt.ylabel('body_rate')
        plt.legend(['r_des', 'p_des', 'y_des', 'r', 'p', 'y'])
        plt.savefig(f'{save_path}/body_rate_plot.png')
        np.savetxt(f'{save_path}/action.txt', self.action_record, delimiter=',')

        plt.figure()
        plt.plot(range(np.shape(self.action_record)[0]), self.action_record[:, 3])
        plt.xlabel('Step')
        plt.ylabel('normalized_force')
        plt.savefig(f'{save_path}/action_force_plot.png')

        # save video (lines 901-904)
        video_tensor = torch.permute(torch.stack(self.img_record), (0, 2, 3, 1)) * 255
        video_tensor = video_tensor.to('cpu')
        video_tensor_uint8 = video_tensor.to(dtype=torch.uint8)
        write_video(f'{save_path}/ego_video.mp4', video_tensor_uint8, fps=20)
        print(f"Visualization saved to {save_path}")
