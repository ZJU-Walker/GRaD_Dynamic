"""
VelNetV2 Live Flight Evaluator.

Same live flight evaluation pattern as the legacy evaluator, adapted for v2:
  - CompactEncoder (RGB only), not DualEncoder
  - RAFT-Small for real-time optical flow
  - FlowEncoder for flow feature extraction
  - Full IMU (accel + gyro) in body frame
  - Additional metrics: angular velocity, direction accuracy, scale accuracy
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import cv2
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
from models.vel_net_v2 import VelNetV2
from models.vel_net.visual_encoder import CompactEncoder
from models.vel_net_body_legacy.vel_obs_utils_body import transform_worldvel_to_bodyvel
from training.vel_net_body.dataset import IMUAugmentation


# Reuse MAP_CONFIGS from legacy evaluator
from training.vel_net_body.evaluator import MAP_CONFIGS, PC_TO_SIM_OFFSET, save_velocity_plot


def fly_and_evaluate_v2(
    model: VelNetV2,
    encoder: CompactEncoder,
    vel_mean: torch.Tensor,
    vel_std: torch.Tensor,
    accel_mean: torch.Tensor = None,
    accel_std: torch.Tensor = None,
    gyro_mean: torch.Tensor = None,
    gyro_std: torch.Tensor = None,
    map_name: str = None,
    waypoints_name: str = 'gate_mid',
    v_avg: float = 1.0,
    output_dir: str = 'output/vel_net_v2_eval',
    device: str = 'cuda:0',
    max_steps: int = 3000,
    smoothing: float = 0.18,
    imu_noise: bool = False,
    action_noise: float = 0.0,
) -> Dict[str, float]:
    """
    Fly drone through trajectory and evaluate VelNetV2 predictions.

    Args:
        model: Trained VelNetV2 model
        encoder: Trained CompactEncoder (RGB only)
        vel_mean/std: For velocity normalization (unused in v2 direct mode, kept for API)
        accel_mean/std: For accel normalization (unused in v2 direct mode, kept for API)
        gyro_mean/std: For gyro normalization (unused in v2 direct mode, kept for API)
        map_name: GS map name (None = from waypoints config)
        waypoints_name: Trajectory config name
        v_avg: Average velocity (m/s)
        output_dir: Output directory
        device: PyTorch device
        max_steps: Maximum simulation steps
        smoothing: B-spline smoothing
        imu_noise: Add IMU noise augmentation
        action_noise: Action noise std

    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    encoder.eval()

    # Load RAFT-Small for real-time optical flow
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    raft_weights = Raft_Small_Weights.DEFAULT
    raft_model = raft_small(weights=raft_weights).to(device)
    raft_model.eval()
    raft_transforms = raft_weights.transforms()

    # IMU noise setup
    imu_aug = None
    imu_bias_accel = np.zeros(3)
    imu_scale_accel = np.ones(3)
    imu_bias_gyro = np.zeros(3)
    imu_scale_gyro = np.ones(3)
    if imu_noise:
        imu_aug = IMUAugmentation(enabled=True)
        imu_bias_accel = np.random.uniform(-imu_aug.bias_range, imu_aug.bias_range, size=3)
        imu_scale_accel = np.random.uniform(1.0 - imu_aug.scale_range, 1.0 + imu_aug.scale_range, size=3)
        imu_bias_gyro = np.random.uniform(-imu_aug.bias_range, imu_aug.bias_range, size=3)
        imu_scale_gyro = np.random.uniform(1.0 - imu_aug.scale_range, 1.0 + imu_aug.scale_range, size=3)
        print(f"  IMU noise enabled")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MAP_CONFIGS[waypoints_name]
    gs_map = map_name if map_name is not None else config.get("gs_map", waypoints_name)

    print(f"\n{'='*60}")
    print(f"VelNetV2 Evaluation Flight")
    print(f"{'='*60}")
    print(f"  GS Map: {gs_map}")
    print(f"  Waypoints: {waypoints_name}")
    print(f"  Velocity: {v_avg} m/s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Setup trajectory
    start_pos_pc = config["start"]
    waypoints_pc = config["waypoints"]
    destination_pc = config["destination"]

    ply_file = Path(__file__).parent.parent.parent / "envs" / "assets" / "point_cloud" / "sv_1007_gate_mid.ply"
    path_pc, _ = get_path(
        current_pos=start_pos_pc,
        waypoints=waypoints_pc,
        destination=destination_pc,
        ply_file=str(ply_file),
        wp_distance=0.5,
    )

    path_sim = [np.array(p) + PC_TO_SIM_OFFSET for p in path_pc]
    start_pos_sim = np.array(start_pos_pc) + PC_TO_SIM_OFFSET

    sampler = generate_bspline_trajectory_from_path(
        path=path_sim, v_avg=v_avg, corner_smoothing=smoothing,
    )

    # Planning plots
    full_waypoints_pc = [start_pos_pc] + waypoints_pc + [destination_pc]
    try:
        save_trajectory_topdown(
            sampler=sampler, astar_path=path_pc, waypoints=full_waypoints_pc,
            output_path=str(output_dir / f'eval_v2_{waypoints_name}_astar_bspline.png'),
            pc_offset=PC_TO_SIM_OFFSET,
        )
    except Exception as e:
        print(f"Warning: Could not save planning plot: {e}")

    # Initialize environment
    print(f"Initializing environment (GS scene: {gs_map})...")
    env = SimpleDroneEnv(
        map_name=gs_map, device=device, num_envs=1,
        episode_length=max_steps, render_resolution=0.4,
    )
    env.reset(start_position=start_pos_sim.tolist())

    controller = GeometricController(
        mass=env.mass, gravity=9.81,
        Kp=np.array([1.5, 1.5, 0.5]),
        Kv=np.array([2.0, 2.0, 1.5]),
        Kr=np.array([3.0, 3.0, 2.0]),
        Kw=np.array([0.8, 0.8, 0.5]),
        max_thrust=env.max_thrust,
        min_thrust=2.0, max_rate=1.0,
    )

    # Flight data
    frames = []
    vel_gt_list = []
    vel_pred_list = []
    omega_gt_list = []
    omega_pred_list = []
    timestamps = []
    trajectory_actual = []
    trajectory_desired = []

    dt = 1.0 / env.sim_freq
    total_time = sampler.total_time + 2.0
    final_pos, _, _, _ = sampler.sample(sampler.total_time)
    start_pos_d, _, _, _ = sampler.sample(0.0)
    stabilize_steps = 50

    prev_vel_raw = torch.zeros(3, device=device)
    prev_vel_gt = np.zeros(3)
    prev_rgb_tensor = None  # For optical flow computation
    model_dt = 1.0 / 30.0

    model.reset_hidden_state(batch_size=1)

    print(f"Flying trajectory ({sampler.total_time:.1f}s)...")
    pbar = tqdm(total=max_steps, desc='Flight', unit='step')

    for step in range(max_steps):
        t = max(0, (step - stabilize_steps) * dt)
        if t > total_time:
            print(f"\nTrajectory completed at step {step}")
            break

        state = env.get_state()
        pos = state['position'][0].cpu().numpy()
        vel_gt_world = state['velocity'][0].cpu().numpy()
        quat_xyzw = state['orientation'][0].cpu().numpy()
        omega_world = state['angular_velocity'][0].cpu().numpy()

        # Body frame velocity + angular velocity GT
        vel_gt_world_t = torch.from_numpy(vel_gt_world.astype(np.float32)).to(device)
        quat_t = torch.from_numpy(quat_xyzw.astype(np.float32)).to(device)
        vel_gt = transform_worldvel_to_bodyvel(vel_gt_world_t, quat_t).cpu().numpy()

        omega_world_t = torch.from_numpy(omega_world.astype(np.float32)).to(device)
        omega_gt = transform_worldvel_to_bodyvel(omega_world_t, quat_t).cpu().numpy()

        # Control
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        if step < stabilize_steps:
            pos_d, vel_d, acc_d, yaw_d = start_pos_d, np.zeros(3), np.zeros(3), 0.0
        elif t <= sampler.total_time:
            pos_d, vel_d, acc_d, _ = sampler.sample(t)
            yaw_d = -sampler.get_yaw_from_velocity(vel_d, default_yaw=0.0)
        else:
            pos_d, vel_d, acc_d, yaw_d = final_pos, np.zeros(3), np.zeros(3), 0.0

        if step >= stabilize_steps:
            trajectory_actual.append(pos.copy())
            trajectory_desired.append(np.array(pos_d).copy())

        thrust, omega_cmd, _ = controller.compute_from_quaternion(
            pos, vel_gt_world, quat_wxyz, omega_world,
            pos_d, vel_d, acc_d, yaw_d,
        )

        rate_scale = 0.5
        action = np.array([
            np.clip(omega_cmd[0] / rate_scale, -1.0, 1.0),
            np.clip(-omega_cmd[1] / rate_scale, -1.0, 1.0),
            np.clip(omega_cmd[2] / rate_scale, -1.0, 1.0),
            np.clip(thrust / controller.max_thrust, 0.0, 1.0),
        ])

        if action_noise > 0:
            noise = np.random.normal(0, action_noise, 3)
            action[0:3] = np.clip(action[0:3] + noise, -1.0, 1.0)

        _, rgb, depth, done, info = env.step(action)
        rgb_np = rgb[0].cpu().numpy()

        # Predict velocity (after stabilization)
        if step >= stabilize_steps:
            with torch.no_grad():
                # RGB feature extraction
                if rgb_np.max() > 1.0:
                    rgb_normalized = rgb_np.astype(np.float32) / 255.0
                else:
                    rgb_normalized = rgb_np.astype(np.float32)

                rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
                rgb_feat = encoder(rgb_tensor)  # (1, 32)

                # Optical flow
                if prev_rgb_tensor is not None:
                    img1_t, img2_t = raft_transforms(prev_rgb_tensor, rgb_tensor)
                    flow_predictions = raft_model(img1_t, img2_t)
                    flow = flow_predictions[-1]  # (1, 2, H, W)
                else:
                    # First frame: zero flow
                    flow = torch.zeros(1, 2, rgb_tensor.shape[2], rgb_tensor.shape[3], device=device)

                flow_feat = model.flow_encoder(flow)  # (1, 64)

                # IMU
                accel_gt = (vel_gt - prev_vel_gt) / model_dt
                omega_imu = omega_gt.copy()

                if imu_noise and imu_aug is not None:
                    accel_aug = accel_gt * imu_scale_accel + imu_bias_accel
                    accel_aug += np.random.normal(0, imu_aug.noise_std, size=3)
                    omega_aug = omega_imu * imu_scale_gyro + imu_bias_gyro
                    omega_aug += np.random.normal(0, imu_aug.noise_std, size=3)
                    accel_tensor = torch.from_numpy(accel_aug.astype(np.float32)).unsqueeze(0).to(device)
                    gyro_tensor = torch.from_numpy(omega_aug.astype(np.float32)).unsqueeze(0).to(device)
                else:
                    accel_tensor = torch.from_numpy(accel_gt.astype(np.float32)).unsqueeze(0).to(device)
                    gyro_tensor = torch.from_numpy(omega_imu.astype(np.float32)).unsqueeze(0).to(device)

                action_tensor = torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(device)

                out = model.encode_step(
                    rgb_feat=rgb_feat,
                    flow_feat=flow_feat,
                    imu_accel=accel_tensor,
                    imu_gyro=gyro_tensor,
                    prev_vel=prev_vel_raw.unsqueeze(0),
                    action=action_tensor,
                )

                vel_pred = out['velocity'].squeeze(0).cpu().numpy()
                omega_pred = out['angular_velocity'].squeeze(0).cpu().numpy()

                prev_vel_raw = out['velocity'].squeeze(0).detach()
                prev_rgb_tensor = rgb_tensor

            vel_gt_list.append(vel_gt.copy())
            vel_pred_list.append(vel_pred.copy())
            omega_gt_list.append(omega_gt.copy())
            omega_pred_list.append(omega_pred.copy())
            timestamps.append(t)

        prev_vel_gt = vel_gt.copy()

        # Video frame
        if step % 2 == 0:
            frame = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
            if step >= stabilize_steps and len(vel_pred_list) > 0:
                frame = frame.copy()
                vel_gt_curr = vel_gt_list[-1]
                vel_pred_curr = vel_pred_list[-1]
                err_norm = np.linalg.norm(vel_pred_curr - vel_gt_curr)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                    f"GT:   [{vel_gt_curr[0]:+.2f}, {vel_gt_curr[1]:+.2f}, {vel_gt_curr[2]:+.2f}]",
                    (5, 15), font, 0.4, (0, 255, 0), 1)
                cv2.putText(frame,
                    f"Pred: [{vel_pred_curr[0]:+.2f}, {vel_pred_curr[1]:+.2f}, {vel_pred_curr[2]:+.2f}]",
                    (5, 31), font, 0.4, (255, 255, 0), 1)
                color = (0, 0, 255) if err_norm > 0.3 else (0, 165, 255) if err_norm > 0.15 else (0, 255, 255)
                cv2.putText(frame,
                    f"Err:  |{err_norm:.3f}| m/s",
                    (5, 47), font, 0.4, color, 1)
            frames.append(frame)

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

    # Compute metrics
    vel_gt_arr = np.array(vel_gt_list)
    vel_pred_arr = np.array(vel_pred_list)
    omega_gt_arr = np.array(omega_gt_list)
    omega_pred_arr = np.array(omega_pred_list)
    timestamps_arr = np.array(timestamps)

    vel_errors = np.abs(vel_pred_arr - vel_gt_arr)
    omega_errors = np.abs(omega_pred_arr - omega_gt_arr)

    metrics = {
        'mae': np.mean(vel_errors),
        'mae_x': np.mean(vel_errors[:, 0]),
        'mae_y': np.mean(vel_errors[:, 1]),
        'mae_z': np.mean(vel_errors[:, 2]),
        'rmse': np.sqrt(np.mean(vel_errors**2)),
        'max_error': np.max(vel_errors),
        'omega_mae': np.mean(omega_errors),
    }

    print(f"\n{'='*60}")
    print(f"VelNetV2 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Vel MAE: {metrics['mae']:.4f} m/s")
    print(f"  Vel MAE (x,y,z): [{metrics['mae_x']:.4f}, {metrics['mae_y']:.4f}, {metrics['mae_z']:.4f}]")
    print(f"  Vel RMSE: {metrics['rmse']:.4f} m/s")
    print(f"  Omega MAE: {metrics['omega_mae']:.4f} rad/s")
    print(f"{'='*60}\n")

    # Save outputs
    video_path = output_dir / f'eval_v2_{waypoints_name}_v{v_avg}.mp4'
    render_and_save(frames, str(video_path), fps=25)
    print(f"Video saved: {video_path}")

    if len(trajectory_actual) > 0:
        waypoints_sim = [np.array(wp) + PC_TO_SIM_OFFSET for wp in full_waypoints_pc]
        try:
            save_trajectory_3d_plot(
                trajectory_actual=trajectory_actual,
                trajectory_desired=trajectory_desired,
                waypoints=waypoints_sim,
                output_path=str(output_dir / f'eval_v2_{waypoints_name}_trajectory.png'),
            )
        except Exception as e:
            print(f"Warning: Could not save trajectory plot: {e}")

    plot_path = output_dir / f'eval_v2_{waypoints_name}_v{v_avg}_velocity.png'
    save_velocity_plot(timestamps_arr, vel_gt_arr, vel_pred_arr, metrics, str(plot_path))
    print(f"Plot saved: {plot_path}")

    np.savez(
        output_dir / f'eval_v2_{waypoints_name}_v{v_avg}_data.npz',
        timestamps=timestamps_arr,
        vel_gt=vel_gt_arr, vel_pred=vel_pred_arr,
        omega_gt=omega_gt_arr, omega_pred=omega_pred_arr,
        trajectory_actual=np.array(trajectory_actual),
        trajectory_desired=np.array(trajectory_desired),
        metrics=metrics,
    )

    return metrics


if __name__ == '__main__':
    print("VelNetV2 Evaluator module loaded successfully")
