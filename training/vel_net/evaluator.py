"""
Velocity Network Evaluator.

Flies the drone through a trajectory and compares vel_net predictions
with ground truth velocities. Saves video and velocity plots.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
from typing import Dict
from tqdm import tqdm

from envs.drone_env import SimpleDroneEnv
from controller.geometric_controller import GeometricController
from controller.nav_helpers import (
    get_path,
    generate_bspline_trajectory_from_path,
    render_and_save,
    save_trajectory_3d_plot,
    save_trajectory_topdown,
    save_trajectory_profile,
)
from models.vel_net import VELO_NET
from models.vel_net.visual_encoder import DualEncoder
from models.vel_net.vel_obs_utils import quaternion_to_rot6d
from training.vel_net.dataset import IMUAugmentation


# Map configurations (same as data_collector.py)
# Each config specifies gs_map (GS scene) and waypoints
MAP_CONFIGS = {
    "gate_mid": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 1.2],
        "waypoints": [
            [-0.2, -0.1, 1.2],
            [1.6, 0.7, 1.1],
            [3.7, 1.5, 0.7],
            [5.8, 0.0, 0.9],
        ],
        "destination": [7.5, -2.0, 1.2],
    },
    "gate_left": {
        "gs_map": "gate_left",
        "start": [-6.0, 0.0, 1.2],
        "waypoints": [
            [-0.2, 1.2, 1.4],
            [3.7, 1.2, 0.6],
            [5.8, 0.0, 1.2],
        ],
        "destination": [7.0, -2.0, 1.2],
    },
    "gate_right": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 1.3],
        "waypoints": [
            [-3.0, -1.0, 1.3],
            [0.0, -1.4, 1.5],
            [1.8, 0.6, 1.1],
            [3.7, 1.4, 0.7],
            [5.8, 0.0, 1.3],
        ],
        "destination": [7.0, -2.0, 1.3],
    },
    # Additional diverse trajectories (all use gate_mid scene)
    "gate_mid_high": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 1.6],
        "waypoints": [
            [-0.2, -0.1, 1.6],
            [1.6, 0.7, 1.5],
            [3.7, 1.5, 1.3],
            [5.8, 0.0, 1.5],
        ],
        "destination": [7.5, -2.0, 1.6],
    },
    "gate_mid_low": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 0.8],
        "waypoints": [
            [-0.2, -0.1, 0.8],
            [1.6, 0.7, 0.7],
            [3.7, 1.5, 0.5],
            [5.8, 0.0, 0.6],
        ],
        "destination": [7.5, -2.0, 0.8],
    },
    "zigzag": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 1.2],
        "waypoints": [
            [-4.0, 1.5, 1.0],
            [-2.0, -1.5, 1.4],
            [0.0, 1.5, 1.0],
            [2.0, -1.0, 1.3],
            [4.0, 1.0, 0.9],
            [6.0, -0.5, 1.1],
        ],
        "destination": [7.5, -2.0, 1.2],
    },
    "straight": {
        "gs_map": "gate_mid",
        "start": [-6.0, 0.0, 1.2],
        "waypoints": [
            [-3.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
            [3.0, 0.0, 1.2],
            [6.0, 0.0, 1.2],
        ],
        "destination": [7.5, 0.0, 1.2],
    },
    "reverse": {
        "gs_map": "gate_mid",
        "start": [7.5, -2.0, 1.2],
        "waypoints": [
            [5.8, 0.0, 0.9],
            [3.7, 1.5, 0.7],
            [1.6, 0.7, 1.1],
            [-0.2, -0.1, 1.2],
        ],
        "destination": [-6.0, 0.0, 1.2],
    },
}

PC_TO_SIM_OFFSET = np.array([6.0, 0.0, 0.0])


def fly_and_evaluate(
    model: VELO_NET,
    encoder: DualEncoder,
    vel_mean: torch.Tensor,
    vel_std: torch.Tensor,
    accel_mean: torch.Tensor = None,
    accel_std: torch.Tensor = None,
    delta_mean: torch.Tensor = None,
    delta_std: torch.Tensor = None,
    map_name: str = 'gate_mid',
    v_avg: float = 1.0,
    output_dir: str = 'output/vel_net_eval',
    device: str = 'cuda:0',
    max_steps: int = 3000,
    smoothing: float = 0.18,
    imu_noise: bool = False,
    action_noise: float = 0.0,
) -> Dict[str, float]:
    """
    Fly drone through trajectory and evaluate vel_net predictions.

    Uses DIRECT DELTA-V prediction mode: vel_pred = prev_vel + delta_v

    Args:
        model: Trained VELO_NET model
        encoder: Trained DualEncoder
        vel_mean: Velocity mean for normalizing INPUT prev_vel (3,)
        vel_std: Velocity std for normalizing INPUT prev_vel (3,)
        accel_mean: Accel mean for normalizing INPUT accel (3,)
        accel_std: Accel std for normalizing INPUT accel (3,)
        delta_mean: Delta mean for denormalizing delta_v OUTPUT (3,)
        delta_std: Delta std for denormalizing delta_v OUTPUT (3,)
        map_name: Map name
        v_avg: Average velocity (m/s)
        output_dir: Output directory for video and plots
        device: PyTorch device
        max_steps: Maximum simulation steps
        smoothing: B-spline corner smoothing
        imu_noise: Whether to add IMU noise augmentation (for realistic testing)
        action_noise: Action noise std (0.0-0.3), adds random noise to body rates

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    encoder.eval()

    vel_mean = vel_mean.to(device)
    vel_std = vel_std.to(device)
    accel_mean = accel_mean.to(device) if accel_mean is not None else torch.zeros(3, device=device)
    accel_std = accel_std.to(device) if accel_std is not None else torch.ones(3, device=device)
    delta_mean = delta_mean.to(device) if delta_mean is not None else torch.zeros(3, device=device)
    delta_std = delta_std.to(device) if delta_std is not None else torch.ones(3, device=device)

    # IMU noise augmentation (sample bias/scale once per flight, like per-sequence in training)
    imu_aug = None
    imu_bias = np.zeros(3)
    imu_scale = np.ones(3)
    if imu_noise:
        imu_aug = IMUAugmentation(enabled=True)
        # Sample constant bias and scale for this flight (per-sequence augmentation)
        imu_bias = np.random.uniform(-imu_aug.bias_range, imu_aug.bias_range, size=3)
        imu_scale = np.random.uniform(1.0 - imu_aug.scale_range, 1.0 + imu_aug.scale_range, size=3)
        print(f"  IMU noise enabled: bias={imu_bias}, scale={imu_scale}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Velocity Network Evaluation Flight")
    print(f"{'='*60}")
    print(f"  Map: {map_name}")
    print(f"  Velocity: {v_avg} m/s")
    print(f"  Output: {output_dir}")
    if action_noise > 0:
        print(f"  Action noise: std={action_noise}")
    print(f"{'='*60}\n")

    # 1. Setup trajectory (same as waypoint_nav_geometric.py)
    config = MAP_CONFIGS[map_name]
    gs_map = config.get("gs_map", map_name)  # GS scene to use
    start_pos_pc = config["start"]
    waypoints_pc = config["waypoints"]
    destination_pc = config["destination"]

    # Get A* path
    ply_file = Path(__file__).parent.parent.parent / "envs" / "assets" / "point_cloud" / "sv_1007_gate_mid.ply"
    path_pc, _ = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=0.5,
    )

    # Convert to simulation space
    path_sim = [np.array(p) + PC_TO_SIM_OFFSET for p in path_pc]
    start_pos_sim = np.array(start_pos_pc) + PC_TO_SIM_OFFSET

    # Generate B-spline trajectory
    sampler = generate_bspline_trajectory_from_path(
        path=path_sim,
        v_avg=v_avg,
        corner_smoothing=smoothing,
    )

    # Save planning plots BEFORE flying
    full_waypoints_pc = [start_pos_pc] + waypoints_pc + [destination_pc]
    try:
        save_trajectory_topdown(
            sampler=sampler,
            astar_path=path_pc,
            waypoints=full_waypoints_pc,
            output_path=str(output_dir / f'eval_{map_name}_astar_bspline.png'),
            pc_offset=PC_TO_SIM_OFFSET,
        )
        print(f"Saved: {output_dir}/eval_{map_name}_astar_bspline.png")
    except Exception as e:
        print(f"Warning: Could not save astar_bspline plot: {e}")

    try:
        save_trajectory_profile(
            sampler=sampler,
            output_path=str(output_dir / f'eval_{map_name}_trajectory_profile.png'),
        )
        print(f"Saved: {output_dir}/eval_{map_name}_trajectory_profile.png")
    except Exception as e:
        print(f"Warning: Could not save trajectory_profile plot: {e}")

    # 2. Initialize environment
    print(f"Initializing environment (GS scene: {gs_map})...")
    env = SimpleDroneEnv(
        map_name=gs_map,  # Use GS scene from config
        device=device,
        num_envs=1,
        episode_length=max_steps,
        render_resolution=0.4,  # Must match training data collection!
    )
    env.reset(start_position=start_pos_sim.tolist())

    # 3. Initialize controller
    controller = GeometricController(
        mass=env.mass,
        gravity=9.81,
        Kp=np.array([1.5, 1.5, 0.5]),
        Kv=np.array([2.0, 2.0, 1.5]),
        Kr=np.array([3.0, 3.0, 2.0]),
        Kw=np.array([0.8, 0.8, 0.5]),
        max_thrust=env.max_thrust,
        min_thrust=2.0,
        max_rate=1.0,
    )

    # 4. Fly and collect data
    frames = []
    vel_gt_list = []
    vel_pred_list = []
    timestamps = []
    trajectory_actual = []
    trajectory_desired = []

    dt = 1.0 / env.sim_freq  # Use env's sim frequency
    total_time = sampler.total_time + 2.0  # Extra hover time
    final_pos, _, _, _ = sampler.sample(sampler.total_time)
    start_pos_d, _, _, _ = sampler.sample(0.0)
    stabilize_steps = 50

    # Initialize prev_vel for auto-regressive inference
    # Keep both raw (for residual calc) and normalized (for model input)
    prev_vel_raw = torch.zeros(3, device=device)
    prev_vel_norm = (prev_vel_raw - vel_mean) / vel_std
    prev_action = torch.zeros(1, 4, device=device)

    # Track previous GT velocity for IMU acceleration computation
    prev_vel_gt = np.zeros(3)
    model_dt = getattr(model, 'dt', 1.0 / 30.0)  # Model's expected dt

    # Reset GRU hidden state for sequential processing
    model.reset_hidden_state(batch_size=1)

    print(f"Flying trajectory ({sampler.total_time:.1f}s)...")
    pbar = tqdm(total=max_steps, desc='Flight', unit='step')

    for step in range(max_steps):
        t = max(0, (step - stabilize_steps) * dt)

        if t > total_time:
            print(f"\nTrajectory completed at step {step}")
            break

        # Get current state
        state = env.get_state()
        pos = state['position'][0].cpu().numpy()
        vel_gt = state['velocity'][0].cpu().numpy()
        quat_xyzw = state['orientation'][0].cpu().numpy()
        omega = state['angular_velocity'][0].cpu().numpy()

        # Compute control action first
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        if step < stabilize_steps:
            pos_d = start_pos_d
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
            yaw_d = 0.0
        elif t <= sampler.total_time:
            pos_d, vel_d, acc_d, _ = sampler.sample(t)
            yaw_d = -sampler.get_yaw_from_velocity(vel_d, default_yaw=0.0)
        else:
            pos_d = final_pos
            vel_d = np.zeros(3)
            acc_d = np.zeros(3)
            yaw_d = 0.0

        # Record trajectory (after stabilization)
        if step >= stabilize_steps:
            trajectory_actual.append(pos.copy())
            trajectory_desired.append(np.array(pos_d).copy())

        thrust, omega_cmd, _ = controller.compute_from_quaternion(
            pos, vel_gt, quat_wxyz, omega,
            pos_d, vel_d, acc_d, yaw_d
        )

        rate_scale = 0.5
        action = np.array([
            np.clip(omega_cmd[0] / rate_scale, -1.0, 1.0),
            np.clip(-omega_cmd[1] / rate_scale, -1.0, 1.0),
            np.clip(omega_cmd[2] / rate_scale, -1.0, 1.0),
            np.clip(thrust / controller.max_thrust, 0.0, 1.0),
        ])

        # Add random noise to action (body rates only, not thrust)
        if action_noise > 0:
            noise = np.random.normal(0, action_noise, 3)
            action[0:3] = np.clip(action[0:3] + noise, -1.0, 1.0)

        # Step environment - returns rgb, depth
        _, rgb, depth, done, info = env.step(action)

        # rgb: (1, H, W, 3), depth: (1, H, W, 1)
        rgb_np = rgb[0].cpu().numpy()  # (H, W, 3)
        depth_np = depth[0].cpu().numpy().squeeze()  # (H, W)

        # Predict velocity with vel_net (after stabilization)
        if step >= stabilize_steps:
            with torch.no_grad():
                # Prepare images - normalize RGB
                if rgb_np.max() > 1.0:
                    rgb_normalized = rgb_np.astype(np.float32) / 255.0
                else:
                    rgb_normalized = rgb_np.astype(np.float32)

                rgb_tensor = torch.from_numpy(rgb_normalized)
                rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

                depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

                # Encode images
                rgb_feat, depth_feat = encoder(rgb_tensor, depth_tensor)

                # Build observation
                quat_tensor = torch.from_numpy(quat_xyzw.astype(np.float32)).unsqueeze(0).to(device)
                rot6d = quaternion_to_rot6d(quat_tensor)

                action_tensor = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)

                # Compute IMU acceleration from GT velocity derivative
                accel_gt = (vel_gt - prev_vel_gt) / model_dt

                # Apply IMU noise if enabled (matches training augmentation)
                if imu_noise:
                    accel_aug = accel_gt.copy()
                    accel_aug = accel_aug + imu_bias  # constant bias per flight
                    accel_aug = accel_aug * imu_scale  # constant scale per flight
                    accel_aug = accel_aug + np.random.normal(0, imu_aug.noise_std, size=3)  # per-frame noise
                    if np.random.random() < imu_aug.dropout_prob:  # sensor dropout
                        accel_aug = np.zeros(3)
                    accel_tensor = torch.from_numpy(accel_aug.astype(np.float32)).unsqueeze(0).to(device)
                else:
                    accel_tensor = torch.from_numpy(accel_gt.astype(np.float32)).unsqueeze(0).to(device)

                # Normalize acceleration for input
                accel_norm = (accel_tensor - accel_mean) / accel_std

                obs = torch.cat([
                    rot6d,           # 6
                    action_tensor,   # 4
                    prev_action,     # 4
                    prev_vel_norm.unsqueeze(0),   # 3 (normalized)
                    rgb_feat,        # 32
                    depth_feat,      # 32
                    accel_norm,      # 3 (normalized IMU acceleration)
                ], dim=1)

                # Model outputs delta_v (normalized) - use encode_step for GRU state
                delta_v_mu_norm, _ = model.encode_step(obs)

                # Direct delta-v: vel_pred = prev_vel + delta_v
                delta_v_raw = delta_v_mu_norm.squeeze(0) * delta_std + delta_mean
                vel_pred_tensor = prev_vel_raw + delta_v_raw
                vel_pred = vel_pred_tensor.cpu().numpy()

                # Update prev_vel for next step
                prev_vel_raw = vel_pred_tensor.detach()
                prev_vel_norm = (prev_vel_raw - vel_mean) / vel_std

            # Record data
            vel_gt_list.append(vel_gt.copy())
            vel_pred_list.append(vel_pred.copy())
            timestamps.append(t)

        # Always update prev_vel_gt for next acceleration computation (even during stabilization)
        prev_vel_gt = vel_gt.copy()

        # Update prev_action for next step
        prev_action = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)

        # Record frame for video
        if step % 2 == 0:
            if rgb_np.max() <= 1.0:
                frame = (rgb_np * 255).astype(np.uint8)
            else:
                frame = rgb_np.astype(np.uint8)
            frames.append(frame)

        # Check for collision/done
        if info.get('collision', False) or done:
            print(f"\nFlight ended: collision={info.get('collision', False)}, done={done}")
            break

        pbar.update(1)
        if len(vel_pred_list) > 0:
            pbar.set_postfix({
                't': f'{t:.1f}s',
                'vel_err': f'{np.linalg.norm(vel_pred_list[-1] - vel_gt_list[-1]):.3f}',
            })

    pbar.close()
    env.close()

    # Convert to arrays
    vel_gt_arr = np.array(vel_gt_list)
    vel_pred_arr = np.array(vel_pred_list)
    timestamps_arr = np.array(timestamps)

    # Compute metrics
    errors = np.abs(vel_pred_arr - vel_gt_arr)
    metrics = {
        'mae': np.mean(errors),
        'mae_x': np.mean(errors[:, 0]),
        'mae_y': np.mean(errors[:, 1]),
        'mae_z': np.mean(errors[:, 2]),
        'rmse': np.sqrt(np.mean(errors**2)),
        'max_error': np.max(errors),
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  MAE: {metrics['mae']:.4f} m/s")
    print(f"  MAE (x,y,z): [{metrics['mae_x']:.4f}, {metrics['mae_y']:.4f}, {metrics['mae_z']:.4f}]")
    print(f"  RMSE: {metrics['rmse']:.4f} m/s")
    print(f"  Max Error: {metrics['max_error']:.4f} m/s")
    print(f"{'='*60}\n")

    # Save video
    video_path = output_dir / f'eval_{map_name}_v{v_avg}.mp4'
    render_and_save(frames, str(video_path), fps=25)
    print(f"Video saved: {video_path}")

    # Save trajectory 3D plot (actual vs desired)
    if len(trajectory_actual) > 0 and len(trajectory_desired) > 0:
        waypoints_sim = [np.array(wp) + PC_TO_SIM_OFFSET for wp in full_waypoints_pc]
        try:
            traj_plot_path = output_dir / f'eval_{map_name}_trajectory.png'
            save_trajectory_3d_plot(
                trajectory_actual=trajectory_actual,
                trajectory_desired=trajectory_desired,
                waypoints=waypoints_sim,
                output_path=str(traj_plot_path),
            )
            print(f"Trajectory plot saved: {traj_plot_path}")
        except Exception as e:
            print(f"Warning: Could not save trajectory plot: {e}")

    # Save velocity plot
    plot_path = output_dir / f'eval_{map_name}_v{v_avg}_velocity.png'
    save_velocity_plot(timestamps_arr, vel_gt_arr, vel_pred_arr, metrics, str(plot_path))
    print(f"Plot saved: {plot_path}")

    # Save data
    data_path = output_dir / f'eval_{map_name}_v{v_avg}_data.npz'
    np.savez(
        data_path,
        timestamps=timestamps_arr,
        vel_gt=vel_gt_arr,
        vel_pred=vel_pred_arr,
        trajectory_actual=np.array(trajectory_actual),
        trajectory_desired=np.array(trajectory_desired),
        metrics=metrics,
    )
    print(f"Data saved: {data_path}")

    return metrics


def save_velocity_plot(
    timestamps: np.ndarray,
    vel_gt: np.ndarray,
    vel_pred: np.ndarray,
    metrics: dict,
    output_path: str,
):
    """Save velocity comparison plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Velocity X
    axes[0].plot(timestamps, vel_gt[:, 0], 'b-', label='GT', linewidth=1.5)
    axes[0].plot(timestamps, vel_pred[:, 0], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
    axes[0].fill_between(timestamps, vel_gt[:, 0], vel_pred[:, 0], alpha=0.2, color='red')
    axes[0].set_ylabel('Vel X (m/s)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Velocity Prediction (Auto-regressive) | MAE: {metrics["mae"]:.4f} m/s | RMSE: {metrics["rmse"]:.4f} m/s')

    # Velocity Y
    axes[1].plot(timestamps, vel_gt[:, 1], 'b-', label='GT', linewidth=1.5)
    axes[1].plot(timestamps, vel_pred[:, 1], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
    axes[1].fill_between(timestamps, vel_gt[:, 1], vel_pred[:, 1], alpha=0.2, color='red')
    axes[1].set_ylabel('Vel Y (m/s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Velocity Z
    axes[2].plot(timestamps, vel_gt[:, 2], 'b-', label='GT', linewidth=1.5)
    axes[2].plot(timestamps, vel_pred[:, 2], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
    axes[2].fill_between(timestamps, vel_gt[:, 2], vel_pred[:, 2], alpha=0.2, color='red')
    axes[2].set_ylabel('Vel Z (m/s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Velocity magnitude
    vel_mag_gt = np.linalg.norm(vel_gt, axis=1)
    vel_mag_pred = np.linalg.norm(vel_pred, axis=1)
    axes[3].plot(timestamps, vel_mag_gt, 'b-', label='GT', linewidth=1.5)
    axes[3].plot(timestamps, vel_mag_pred, 'r--', label='Pred', linewidth=1.5, alpha=0.8)
    axes[3].fill_between(timestamps, vel_mag_gt, vel_mag_pred, alpha=0.2, color='red')
    axes[3].set_ylabel('Vel Mag (m/s)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    print("Evaluator module loaded successfully")
