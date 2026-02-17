"""Visualization utilities for policy training.

Preserves original pattern from drone_long_traj.py lines 797-904.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import write_video
from torchvision.transforms import Resize
import cv2


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

    def _overlay_text_on_image(self, img_tensor, text_lines):
        """
        Overlay text on image tensor.

        Args:
            img_tensor: (C, H, W) tensor with values in [0, 1] or [0, 255]
            text_lines: list of strings to overlay

        Returns:
            img_tensor with text overlay (same range as input)
        """
        # Convert to numpy HWC for cv2
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Check if image is in [0, 1] or [0, 255] range
        is_normalized = img_np.max() <= 1.0

        if is_normalized:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        img_np = np.ascontiguousarray(img_np)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # White
        shadow_color = (0, 0, 0)  # Black shadow for readability

        y_offset = 30
        for i, text in enumerate(text_lines):
            y = y_offset + i * 28
            # Draw shadow
            cv2.putText(img_np, text, (12, y+2), font, font_scale, shadow_color, thickness+1)
            # Draw text
            cv2.putText(img_np, text, (10, y), font, font_scale, color, thickness)

        # Convert back to tensor (same range as input)
        if is_normalized:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_tensor

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

        # Get velocity data for overlay
        velo_data = lin_vel.clone().detach().cpu().numpy()
        vx, vy, vz = velo_data[0, 0], velo_data[0, 1], velo_data[0, 2]
        vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

        # Overlay velocity text on image
        text_lines = [
            f"Vel: [{vx:.2f}, {vy:.2f}, {vz:.2f}] m/s",
            f"|V|: {vel_mag:.2f} m/s",
        ]
        img = self._overlay_text_on_image(img, text_lines)

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
        # Move image to CPU to avoid device mismatch and save GPU memory
        self.img_record.append(img.cpu())

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
        plt.legend(['vx', 'vy', 'vz'])
        plt.savefig(f'{save_path}/velo_plot.png')

        # figure forward velocity (body-frame forward + velocity magnitude)
        plt.figure(figsize=(12, 8))

        # Compute forward velocity (velocity projected onto heading direction)
        vx = np.array(self.velo_x_record)
        vy = np.array(self.velo_y_record)
        vz = np.array(self.velo_z_record)
        yaw = np.array(self.yaw_record)

        # Forward velocity = vx * cos(yaw) + vy * sin(yaw)
        v_forward = vx * np.cos(yaw) + vy * np.sin(yaw)
        # Lateral velocity = -vx * sin(yaw) + vy * cos(yaw)
        v_lateral = -vx * np.sin(yaw) + vy * np.cos(yaw)
        # Velocity magnitude (horizontal)
        v_mag_horizontal = np.sqrt(vx**2 + vy**2)
        # Velocity magnitude (3D)
        v_mag_3d = np.sqrt(vx**2 + vy**2 + vz**2)

        plt.subplot(2, 1, 1)
        plt.plot(range(len(v_forward)), v_forward, 'b-', linewidth=2, label='Forward (body X)')
        plt.plot(range(len(v_lateral)), v_lateral, 'g-', linewidth=1.5, label='Lateral (body Y)')
        plt.plot(range(len(vz)), vz, 'r-', linewidth=1.5, label='Vertical (Z)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel("Step")
        plt.ylabel("m/s")
        plt.title("Body-Frame Velocity Components")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(v_mag_horizontal)), v_mag_horizontal, 'b-', linewidth=2, label='Horizontal |V|')
        plt.plot(range(len(v_mag_3d)), v_mag_3d, 'r-', linewidth=2, label='3D |V|')
        plt.plot(range(len(v_forward)), v_forward, 'g--', linewidth=1.5, alpha=0.7, label='Forward V')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel("Step")
        plt.ylabel("m/s")
        plt.title("Velocity Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/forward_velo_plot.png')
        print(f"  Forward velocity plot saved: {save_path}/forward_velo_plot.png")

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
        # Move all images to CPU before stacking to avoid device mismatch
        img_record_cpu = [img.cpu() if isinstance(img, torch.Tensor) else img for img in self.img_record]
        video_tensor = torch.permute(torch.stack(img_record_cpu), (0, 2, 3, 1)) * 255
        video_tensor = video_tensor.to('cpu')
        video_tensor_uint8 = video_tensor.to(dtype=torch.uint8)
        write_video(f'{save_path}/ego_video.mp4', video_tensor_uint8, fps=20)
        print(f"Visualization saved to {save_path}")
