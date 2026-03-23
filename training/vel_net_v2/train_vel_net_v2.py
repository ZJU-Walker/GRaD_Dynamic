#!/usr/bin/env python3
"""
VelNetV2 Training Script — Main CLI Entry Point.

Multi-stage training pipeline for the v2 velocity estimation network.

Usage:
    # Pre-compute optical flow (run once, after backbone features exist)
    python training/vel_net_v2/precompute_optical_flow.py \
        --data_dir data/vel_net/sequences --device cuda:0

    # Stage 1: Geometry (angular velocity + translation direction)
    python training/vel_net_v2/train_vel_net_v2.py train --stage 1 \
        --data_dir data/vel_net/sequences --epochs 200

    # Stage 2: Dynamics (scale + correction), resume from Stage 1
    python training/vel_net_v2/train_vel_net_v2.py train --stage 2 \
        --data_dir data/vel_net/sequences \
        --resume checkpoints/vel_net_v2/stage1_best.pt --epochs 150

    # Stage 3: Joint fine-tuning, resume from Stage 2
    python training/vel_net_v2/train_vel_net_v2.py train --stage 3 \
        --data_dir data/vel_net/sequences \
        --resume checkpoints/vel_net_v2/stage2_best.pt --epochs 100

    # Auto 3-stage training (single command)
    python training/vel_net_v2/train_vel_net_v2.py train_all \
        --config training/vel_net_v2/configs/train_all.yaml

    # Resume auto training from stage 2
    python training/vel_net_v2/train_vel_net_v2.py train_all \
        --config training/vel_net_v2/configs/train_all.yaml --start_stage 2

    # Auto-regressive test on a sequence
    python training/vel_net_v2/train_vel_net_v2.py test \
        --checkpoint checkpoints/vel_net_v2/stage3_best.pt \
        --test_seq data/vel_net/sequences/seq_0000

    # Live evaluation flight
    python training/vel_net_v2/train_vel_net_v2.py eval \
        --checkpoint checkpoints/vel_net_v2/stage3_best.pt \
        --waypoints gate_mid --v_avg 1.0

    # Data collection (delegates to existing collector)
    python training/vel_net_v2/train_vel_net_v2.py collect \
        --waypoints gate_mid --n_sequences 30
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import yaml
import torch
import numpy as np

from models.vel_net_v2 import VelNetV2
from models.vel_net.visual_encoder import CompactEncoder


def collect_command(args):
    """Run data collection (delegates to existing data_collector.py)."""
    from training.vel_net_body.data_collector import collect_sequences
    collect_sequences(
        output_dir=args.output_dir,
        map_name=args.map,
        waypoints_name=args.waypoints,
        n_sequences=args.n_sequences,
        collection_freq=args.freq,
        v_min=args.v_min,
        v_max=args.v_max,
        smoothing=args.smoothing,
        action_noise=args.action_noise,
        waypoint_noise=args.waypoint_noise,
        device=args.device,
    )


def train_command(args):
    """Run multi-stage training."""
    from training.vel_net_v2.dataset_v2 import create_dataloaders_v2
    from training.vel_net_v2.trainer_v2 import VelNetV2Trainer

    device = args.device
    stage = args.stage

    seq_length = args.seq_length if args.seq_length > 0 else None
    stride = args.stride if args.seq_length > 0 else None

    print(f"\n{'='*60}")
    print(f"VelNetV2 Training — Stage {stage}")
    print(f"{'='*60}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Stage: {stage}")
    print(f"  Seq length: {seq_length or 'full'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Resume: {args.resume or 'None'}")
    print(f"{'='*60}\n")

    # Create encoder (RGB only)
    encoder = CompactEncoder(input_channels=3, output_dim=32).to(device)
    print(f"Encoder created: {encoder.num_trainable_params():,} trainable params")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders_v2(
        data_dir=args.data_dir,
        seq_length=seq_length,
        stride=stride,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    # Create model
    model = VelNetV2(
        rgb_feat_dim=32,
        flow_feat_dim=64,
        geo_hidden_dim=args.geo_hidden_dim,
        geo_gru_layers=args.geo_gru_layers,
        dyn_hidden_dim=args.dyn_hidden_dim,
        dyn_gru_layers=args.dyn_gru_layers,
    ).to(device)
    print(f"Model created: {model.num_total_params():,} total params")

    # Create trainer
    trainer = VelNetV2Trainer(
        model=model,
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        stage=stage,
        lr=args.lr,
        lr_geometry=args.lr_geometry,
        lr_dynamics=args.lr_dynamics,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        tf_start_epoch=args.tf_start_epoch,
        tf_end_epoch=args.tf_end_epoch,
        grad_accumulation_steps=args.grad_accum,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Resume from checkpoint
    if args.resume:
        # For stage transitions, load model weights but don't load optimizer
        load_optimizer = (stage == int(torch.load(args.resume, map_location='cpu', weights_only=False).get('stage', stage)))
        trainer.load_checkpoint(args.resume, load_optimizer=load_optimizer)

    # Train
    history = trainer.train(
        n_epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
    )

    print(f"\nStage {stage} training complete!")


def train_all_command(args):
    """Run all 3 training stages automatically with checkpoint chaining."""
    from training.vel_net_v2.dataset_v2 import create_dataloaders_v2
    from training.vel_net_v2.trainer_v2 import VelNetV2Trainer

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    device = args.device or config.get('device', 'cuda:0')
    data_dir = args.data_dir or config.get('data_dir', 'data/vel_net/sequences')
    checkpoint_dir = args.checkpoint_dir or config.get('checkpoint_dir', 'checkpoints/vel_net_v2')
    use_wandb = args.wandb or config.get('wandb', False)
    wandb_project = config.get('wandb_project', 'vel_net_v2')
    start_stage = args.start_stage

    model_cfg = config['model']
    data_cfg = config['data']
    defaults = config.get('training_defaults', {})

    print(f"\n{'='*60}")
    print(f"VelNetV2 — Auto 3-Stage Training")
    print(f"{'='*60}")
    print(f"  Data dir: {data_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Device: {device}")
    print(f"  Start stage: {start_stage}")
    print(f"{'='*60}\n")

    for stage in [1, 2, 3]:
        if stage < start_stage:
            print(f"Skipping stage {stage} (start_stage={start_stage})")
            continue

        stage_cfg = config['stages'][stage]
        # Merge defaults with stage overrides
        merged = {**defaults, **stage_cfg}

        print(f"\n{'='*60}")
        print(f"Stage {stage}/3")
        print(f"{'='*60}")
        print(f"  Epochs: {merged['epochs']}")
        print(f"  LR: {merged['lr']}")
        print(f"  Early stop patience: {merged['early_stop_patience']}")

        # Create encoder
        encoder = CompactEncoder(input_channels=3, output_dim=model_cfg['rgb_feat_dim']).to(device)
        print(f"  Encoder: {encoder.num_trainable_params():,} params")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders_v2(
            data_dir=data_dir,
            seq_length=data_cfg['seq_length'],
            stride=data_cfg['stride'],
            batch_size=data_cfg['batch_size'],
            val_ratio=data_cfg['val_ratio'],
        )

        # Create model
        model = VelNetV2(
            rgb_feat_dim=model_cfg['rgb_feat_dim'],
            flow_feat_dim=model_cfg['flow_feat_dim'],
            geo_hidden_dim=model_cfg['geo_hidden_dim'],
            geo_gru_layers=model_cfg['geo_gru_layers'],
            dyn_hidden_dim=model_cfg['dyn_hidden_dim'],
            dyn_gru_layers=model_cfg['dyn_gru_layers'],
        ).to(device)
        print(f"  Model: {model.num_total_params():,} params")

        # Create trainer
        trainer = VelNetV2Trainer(
            model=model,
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            stage=stage,
            lr=merged['lr'],
            lr_geometry=merged.get('lr_geometry', 3e-5),
            lr_dynamics=merged.get('lr_dynamics', 3e-5),
            weight_decay=merged.get('weight_decay', 1e-5),
            grad_clip=merged.get('grad_clip', 1.0),
            tf_start_epoch=merged.get('tf_start_epoch', 0),
            tf_end_epoch=merged.get('tf_end_epoch', 100),
            grad_accumulation_steps=merged.get('grad_accumulation_steps', 1),
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

        # Resume from previous stage checkpoint
        if stage > 1:
            prev_checkpoint = os.path.join(checkpoint_dir, f"stage{stage - 1}_best.pt")
            print(f"  Loading checkpoint: {prev_checkpoint}")
            trainer.load_checkpoint(prev_checkpoint, load_optimizer=False)

        # Train
        history = trainer.train(
            n_epochs=merged['epochs'],
            early_stop_patience=merged['early_stop_patience'],
        )

        best_val = min(history['val_loss']) if history.get('val_loss') else float('inf')
        print(f"\nStage {stage} complete! Best val loss: {best_val:.6f}")

    print(f"\n{'='*60}")
    print(f"All 3 stages complete!")
    print(f"Final checkpoint: {os.path.join(checkpoint_dir, 'stage3_best.pt')}")
    print(f"{'='*60}\n")


def test_command(args):
    """Run auto-regressive test on a sequence."""
    from models.vel_net_body_legacy.vel_obs_utils_body import transform_worldvel_to_bodyvel
    from training.vel_net_v2.dataset_v2 import compute_angular_velocity_body

    device = args.device

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create encoder
    encoder = CompactEncoder(input_channels=3, output_dim=32).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Create model
    config = checkpoint['model_config']
    model = VelNetV2(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Stage: {checkpoint.get('stage', '?')}, Epoch: {checkpoint.get('epoch', '?')}")

    model.eval()
    encoder.eval()

    # Load test sequence
    seq_path = Path(args.test_seq)
    telemetry = np.load(seq_path / "telemetry.npz")
    n_frames = min(len(telemetry['timestamps']), args.max_steps)
    dt = 1.0 / 30.0

    # Load precomputed data
    features = np.load(seq_path / "backbone_features.npz")
    rgb_backbone = torch.from_numpy(features['rgb_features']).to(device)

    flows_data = np.load(seq_path / "optical_flow.npz")
    optical_flows = torch.from_numpy(flows_data['flows']).to(device)

    # Prepare GT
    velocities_world = telemetry['velocities'].astype(np.float32)
    orientations = telemetry['orientations'].astype(np.float32)
    actions = telemetry['actions'].astype(np.float32)

    velocities_world_t = torch.from_numpy(velocities_world).to(device)
    orientations_t = torch.from_numpy(orientations).to(device)
    velocities_body = transform_worldvel_to_bodyvel(velocities_world_t, orientations_t).cpu().numpy()

    # Compute body-frame acceleration
    accel_world = np.zeros_like(velocities_world)
    accel_world[1:] = (velocities_world[1:] - velocities_world[:-1]) / dt
    accel_world[0] = accel_world[1]
    accel_body = transform_worldvel_to_bodyvel(
        torch.from_numpy(accel_world).to(device), orientations_t
    ).cpu().numpy()

    # Angular velocity GT for comparison (not model input — RotationNet predicts from flow)
    omega_body = compute_angular_velocity_body(orientations, dt)
    omega_body_padded = np.zeros((len(orientations), 3), dtype=np.float32)
    omega_body_padded[1:] = omega_body
    omega_body_padded[0] = omega_body[0] if len(omega_body) > 0 else np.zeros(3)

    # Auto-regressive test
    all_preds = []
    all_gts = []
    all_omega_preds = []
    all_omega_gts = []
    all_directions = []
    all_scales = []
    all_corrections = []

    prev_vel = torch.zeros(3, device=device)
    model.reset_hidden_state(batch_size=1)

    with torch.no_grad():
        for t in range(1, n_frames):
            rgb_feat = encoder.forward_from_backbone_features(rgb_backbone[t:t+1])

            flow_idx = max(0, t - 1)
            flow_raw = optical_flows[flow_idx:flow_idx+1]  # (1, 2, H, W)
            flow_feat = model.flow_encoder(flow_raw)

            accel_t = torch.from_numpy(accel_body[t].astype(np.float32)).unsqueeze(0).to(device)
            action_t = torch.from_numpy(actions[t].astype(np.float32)).unsqueeze(0).to(device)

            out = model.encode_step(
                rgb_feat=rgb_feat,
                flow_feat=flow_feat,
                flow_raw=flow_raw,
                imu_accel=accel_t,
                prev_vel=prev_vel.unsqueeze(0),
                action=action_t,
            )

            vel_pred = out['velocity'].squeeze(0)
            prev_vel = vel_pred.detach()

            all_preds.append(vel_pred.cpu().numpy())
            all_gts.append(velocities_body[t])
            all_omega_preds.append(out['angular_velocity'].squeeze(0).cpu().numpy())
            all_omega_gts.append(omega_body_padded[t])
            all_directions.append(out['translation_direction'].squeeze(0).cpu().numpy())
            all_scales.append(out['translation_scale'].squeeze(0).cpu().numpy())
            all_corrections.append(out['motion_correction'].squeeze(0).cpu().numpy())

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    all_omega_preds = np.array(all_omega_preds)  # (T, 3)
    all_omega_gts = np.array(all_omega_gts)      # (T, 3)
    all_directions = np.array(all_directions)   # (T, 3)
    all_scales = np.array(all_scales)           # (T, 1)
    all_corrections = np.array(all_corrections) # (T, 3)
    errors = np.abs(all_preds - all_gts)
    omega_errors = np.abs(all_omega_preds - all_omega_gts)

    metrics = {
        'mae': np.mean(errors),
        'mae_x': np.mean(errors[:, 0]),
        'mae_y': np.mean(errors[:, 1]),
        'mae_z': np.mean(errors[:, 2]),
        'rmse': np.sqrt(np.mean(errors**2)),
        'omega_mae': np.mean(omega_errors),
        'omega_mae_x': np.mean(omega_errors[:, 0]),
        'omega_mae_y': np.mean(omega_errors[:, 1]),
        'omega_mae_z': np.mean(omega_errors[:, 2]),
    }

    print(f"\nAuto-regressive Test (VelNetV2, {n_frames-1} steps):")
    print(f"  MAE: {metrics['mae']:.4f} m/s")
    print(f"  MAE (x,y,z): [{metrics['mae_x']:.4f}, {metrics['mae_y']:.4f}, {metrics['mae_z']:.4f}]")
    print(f"  RMSE: {metrics['rmse']:.4f} m/s")
    print(f"  Omega MAE: {metrics['omega_mae']:.4f} rad/s")
    print(f"  Omega MAE (x,y,z): [{metrics['omega_mae_x']:.4f}, {metrics['omega_mae_y']:.4f}, {metrics['omega_mae_z']:.4f}]")

    mean_correction = np.mean(all_corrections, axis=0)
    print(f"  Mean correction (Δv): [{mean_correction[0]:.4f}, {mean_correction[1]:.4f}, {mean_correction[2]:.4f}]")

    # Save plot
    if args.save_plot:
        import matplotlib.pyplot as plt

        timestamps = telemetry['timestamps'][1:n_frames]
        t_axis = timestamps - timestamps[0]

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        labels = ['Vel X', 'Vel Y', 'Vel Z']
        for i, label in enumerate(labels):
            axes[i].plot(t_axis, all_gts[:, i], 'b-', label='GT', linewidth=1.5)
            axes[i].plot(t_axis, all_preds[:, i], 'r--', label='Pred', linewidth=1.5)
            axes[i].set_ylabel(f'{label} (m/s)')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        vel_mag_gt = np.linalg.norm(all_gts, axis=1)
        vel_mag_pred = np.linalg.norm(all_preds, axis=1)
        axes[3].plot(t_axis, vel_mag_gt, 'b-', label='GT', linewidth=1.5)
        axes[3].plot(t_axis, vel_mag_pred, 'r--', label='Pred', linewidth=1.5)
        axes[3].set_ylabel('Vel Mag (m/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        axes[0].set_title(f'VelNetV2 Auto-Regressive Test | MAE: {metrics["mae"]:.4f} m/s')
        plt.tight_layout()

        plot_path = args.plot_path or str(seq_path / 'vel_v2_prediction.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")

        # 1) Direction (d) — x, y, z
        fig_dir, ax_dir = plt.subplots(figsize=(14, 3.5))
        for i, c in enumerate(['X', 'Y', 'Z']):
            ax_dir.plot(t_axis, all_directions[:, i], label=f'd_{c}', linewidth=1.2)
        ax_dir.set_ylabel('Direction (unit)')
        ax_dir.set_xlabel('Time (s)')
        ax_dir.set_title('Branch Decomposition | Direction (d)')
        ax_dir.legend(loc='upper right')
        ax_dir.grid(True, alpha=0.3)
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_direction.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")

        # 2) Scale (s) vs GT speed
        fig_sc, ax_sc = plt.subplots(figsize=(14, 3.5))
        ax_sc.plot(t_axis, all_scales[:, 0], 'k-', label='scale', linewidth=1.5)
        gt_speed = np.linalg.norm(all_gts, axis=1)
        ax_sc.plot(t_axis, gt_speed, 'b--', label='GT speed', linewidth=1.0, alpha=0.7)
        ax_sc.set_ylabel('Scale (m/s)')
        ax_sc.set_xlabel('Time (s)')
        ax_sc.set_title('Branch Decomposition | Scale (s) vs GT Speed')
        ax_sc.legend(loc='upper right')
        ax_sc.grid(True, alpha=0.3)
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_scale.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")

        # 3) d*s geometric velocity vs GT
        fig_geo, ax_geo = plt.subplots(figsize=(14, 3.5))
        geo_vel = all_directions * all_scales  # (T, 3)
        for i, c in enumerate(['X', 'Y', 'Z']):
            ax_geo.plot(t_axis, geo_vel[:, i], label=f'd*s {c}', linewidth=1.2)
            ax_geo.plot(t_axis, all_gts[:, i], '--', alpha=0.5, linewidth=1.0)
        ax_geo.set_ylabel('d * s (m/s)')
        ax_geo.set_xlabel('Time (s)')
        ax_geo.set_title('Branch Decomposition | d*s Geometric Velocity vs GT')
        ax_geo.legend(loc='upper right')
        ax_geo.grid(True, alpha=0.3)
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_geo_vel.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")

        # 4) Correction Δv (bias suspect)
        fig_cor, ax_cor = plt.subplots(figsize=(14, 3.5))
        for i, c in enumerate(['X', 'Y', 'Z']):
            ax_cor.plot(t_axis, all_corrections[:, i], label=f'Δv_{c}', linewidth=1.5)
        ax_cor.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax_cor.set_ylabel('Correction Δv (m/s)')
        ax_cor.set_xlabel('Time (s)')
        ax_cor.set_title('Branch Decomposition | Correction Δv')
        ax_cor.legend(loc='upper right')
        ax_cor.grid(True, alpha=0.3)
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_correction.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")

        # 5) Final velocity vs GT per-axis
        fig_fin, ax_fin = plt.subplots(figsize=(14, 3.5))
        colors = ['r', 'g', 'b']
        for i, c in enumerate(['X', 'Y', 'Z']):
            ax_fin.plot(t_axis, all_gts[:, i], color=colors[i], linestyle='-',
                        label=f'GT {c}', linewidth=1.2, alpha=0.7)
            ax_fin.plot(t_axis, all_preds[:, i], color=colors[i], linestyle='--',
                        label=f'Pred {c}', linewidth=1.2)
        ax_fin.set_ylabel('Velocity (m/s)')
        ax_fin.set_xlabel('Time (s)')
        ax_fin.set_title('Branch Decomposition | Final Velocity vs GT')
        ax_fin.legend(loc='upper right', ncol=2)
        ax_fin.grid(True, alpha=0.3)
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_final_vs_gt.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")

        # Combined decomposition plot (all 5 rows)
        fig2, axes2 = plt.subplots(5, 1, figsize=(14, 17.5), sharex=True)

        for i, c in enumerate(['X', 'Y', 'Z']):
            axes2[0].plot(t_axis, all_directions[:, i], label=f'd_{c}', linewidth=1.2)
        axes2[0].set_ylabel('Direction (unit)')
        axes2[0].set_title('Branch Decomposition | velocity = d * s + Δv')
        axes2[0].legend(loc='upper right')
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(t_axis, all_scales[:, 0], 'k-', label='scale', linewidth=1.5)
        axes2[1].plot(t_axis, gt_speed, 'b--', label='GT speed', linewidth=1.0, alpha=0.7)
        axes2[1].set_ylabel('Scale (m/s)')
        axes2[1].legend(loc='upper right')
        axes2[1].grid(True, alpha=0.3)

        for i, c in enumerate(['X', 'Y', 'Z']):
            axes2[2].plot(t_axis, geo_vel[:, i], label=f'd*s {c}', linewidth=1.2)
            axes2[2].plot(t_axis, all_gts[:, i], '--', alpha=0.5, linewidth=1.0)
        axes2[2].set_ylabel('d * s (m/s)')
        axes2[2].legend(loc='upper right')
        axes2[2].grid(True, alpha=0.3)

        for i, c in enumerate(['X', 'Y', 'Z']):
            axes2[3].plot(t_axis, all_corrections[:, i], label=f'Δv_{c}', linewidth=1.5)
        axes2[3].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        axes2[3].set_ylabel('Correction Δv (m/s)')
        axes2[3].legend(loc='upper right')
        axes2[3].grid(True, alpha=0.3)

        colors = ['r', 'g', 'b']
        for i, c in enumerate(['X', 'Y', 'Z']):
            axes2[4].plot(t_axis, all_gts[:, i], color=colors[i], linestyle='-',
                          label=f'GT {c}', linewidth=1.2, alpha=0.7)
            axes2[4].plot(t_axis, all_preds[:, i], color=colors[i], linestyle='--',
                          label=f'Pred {c}', linewidth=1.2)
        axes2[4].set_ylabel('Velocity (m/s)')
        axes2[4].set_xlabel('Time (s)')
        axes2[4].legend(loc='upper right', ncol=2)
        axes2[4].grid(True, alpha=0.3)

        plt.tight_layout()
        decomp_path = str(seq_path / 'vel_v2_decomposition.png')
        plt.savefig(decomp_path, dpi=150)
        plt.close()
        print(f"  Combined decomposition saved: {decomp_path}")

        # Angular velocity plot
        fig_omega, axes_omega = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        omega_labels = ['Omega X (roll)', 'Omega Y (pitch)', 'Omega Z (yaw)']
        for i, label in enumerate(omega_labels):
            axes_omega[i].plot(t_axis, all_omega_gts[:, i], 'b-', label='GT', linewidth=1.5)
            axes_omega[i].plot(t_axis, all_omega_preds[:, i], 'r--', label='Pred', linewidth=1.5)
            axes_omega[i].set_ylabel(f'{label} (rad/s)')
            axes_omega[i].legend(loc='upper right')
            axes_omega[i].grid(True, alpha=0.3)
        axes_omega[0].set_title(f'Angular Velocity | MAE: {metrics["omega_mae"]:.4f} rad/s')
        axes_omega[2].set_xlabel('Time (s)')
        plt.tight_layout()
        p = str(seq_path / 'vel_v2_angular_velocity.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p}")


def eval_command(args):
    """Run evaluation flight."""
    from training.vel_net_v2.evaluator_v2 import fly_and_evaluate_v2

    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    encoder = CompactEncoder(input_channels=3, output_dim=32).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    config = checkpoint['model_config']
    model = VelNetV2(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Stage: {checkpoint.get('stage', '?')}, Epoch: {checkpoint.get('epoch', '?')}")

    vel_mean = checkpoint.get('vel_mean', torch.zeros(3))
    vel_std = checkpoint.get('vel_std', torch.ones(3))
    accel_mean = checkpoint.get('accel_mean', torch.zeros(3))
    accel_std = checkpoint.get('accel_std', torch.ones(3))
    gyro_mean = checkpoint.get('gyro_mean', torch.zeros(3))
    gyro_std = checkpoint.get('gyro_std', torch.ones(3))

    fly_and_evaluate_v2(
        model=model,
        encoder=encoder,
        vel_mean=vel_mean,
        vel_std=vel_std,
        accel_mean=accel_mean,
        accel_std=accel_std,
        gyro_mean=gyro_mean,
        gyro_std=gyro_std,
        map_name=args.map,
        waypoints_name=args.waypoints,
        v_avg=args.v_avg,
        output_dir=args.output_dir,
        device=device,
        max_steps=args.max_steps,
        smoothing=args.smoothing,
        imu_noise=args.imu_noise,
        action_noise=args.action_noise,
    )


def main():
    parser = argparse.ArgumentParser(
        description='VelNetV2 Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # =========================================================================
    # Collect command
    # =========================================================================
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--output_dir', type=str, default='data/vel_net/sequences')
    collect_parser.add_argument('--map', type=str, default=None)
    collect_parser.add_argument('--waypoints', type=str, default='gate_mid')
    collect_parser.add_argument('--n_sequences', type=int, default=30)
    collect_parser.add_argument('--freq', type=float, default=30.0)
    collect_parser.add_argument('--v_min', type=float, default=0.5)
    collect_parser.add_argument('--v_max', type=float, default=2.0)
    collect_parser.add_argument('--smoothing', type=float, default=0.018)
    collect_parser.add_argument('--action_noise', type=float, default=0.0)
    collect_parser.add_argument('--waypoint_noise', type=float, default=0.0)
    collect_parser.add_argument('--device', type=str, default='cuda:0')

    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser('train', help='Train VelNetV2')
    train_parser.add_argument('--data_dir', type=str, default='data/vel_net/sequences')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vel_net_v2')
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                              help='Training stage (1=geometry, 2=dynamics, 3=joint)')

    # Model config
    train_parser.add_argument('--geo_hidden_dim', type=int, default=128)
    train_parser.add_argument('--geo_gru_layers', type=int, default=2)
    train_parser.add_argument('--dyn_hidden_dim', type=int, default=128)
    train_parser.add_argument('--dyn_gru_layers', type=int, default=2)

    # Sequence config
    train_parser.add_argument('--seq_length', type=int, default=64)
    train_parser.add_argument('--stride', type=int, default=32)

    # Training config
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--batch_size', type=int, default=8)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--lr_geometry', type=float, default=3e-5,
                              help='LR for geometry branch in Stage 3')
    train_parser.add_argument('--lr_dynamics', type=float, default=3e-5,
                              help='LR for dynamics branch in Stage 3')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5)
    train_parser.add_argument('--grad_clip', type=float, default=1.0)
    train_parser.add_argument('--grad_accum', type=int, default=1)
    train_parser.add_argument('--val_ratio', type=float, default=0.1)
    train_parser.add_argument('--early_stop_patience', type=int, default=30)

    # Scheduled sampling
    train_parser.add_argument('--tf_start_epoch', type=int, default=0)
    train_parser.add_argument('--tf_end_epoch', type=int, default=100)

    # Logging
    train_parser.add_argument('--wandb', action='store_true')
    train_parser.add_argument('--wandb_project', type=str, default='vel_net_v2')

    train_parser.add_argument('--device', type=str, default='cuda:0')

    # =========================================================================
    # Train All command (auto 3-stage)
    # =========================================================================
    train_all_parser = subparsers.add_parser('train_all',
        help='Run all 3 training stages automatically')
    train_all_parser.add_argument('--config', type=str, required=True,
                                  help='Path to unified train_all.yaml config')
    train_all_parser.add_argument('--device', type=str, default=None,
                                  help='Override device from config')
    train_all_parser.add_argument('--data_dir', type=str, default=None,
                                  help='Override data_dir from config')
    train_all_parser.add_argument('--checkpoint_dir', type=str, default=None,
                                  help='Override checkpoint_dir from config')
    train_all_parser.add_argument('--wandb', action='store_true',
                                  help='Enable wandb logging')
    train_all_parser.add_argument('--start_stage', type=int, default=1,
                                  choices=[1, 2, 3],
                                  help='Stage to start from (skip earlier stages)')

    # =========================================================================
    # Test command
    # =========================================================================
    test_parser = subparsers.add_parser('test', help='Auto-regressive test')
    test_parser.add_argument('--checkpoint', type=str, required=True)
    test_parser.add_argument('--test_seq', type=str, required=True)
    test_parser.add_argument('--max_steps', type=int, default=9999)
    test_parser.add_argument('--save_plot', action='store_true', default=True)
    test_parser.add_argument('--plot_path', type=str, default=None)
    test_parser.add_argument('--device', type=str, default='cuda:0')

    # =========================================================================
    # Eval command
    # =========================================================================
    eval_parser = subparsers.add_parser('eval', help='Evaluation flight')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--map', type=str, default=None)
    eval_parser.add_argument('--waypoints', type=str, default='gate_mid')
    eval_parser.add_argument('--v_avg', type=float, default=1.0)
    eval_parser.add_argument('--smoothing', type=float, default=0.018)
    eval_parser.add_argument('--max_steps', type=int, default=3000)
    eval_parser.add_argument('--output_dir', type=str, default='output/vel_net_v2_eval')
    eval_parser.add_argument('--imu_noise', action='store_true')
    eval_parser.add_argument('--action_noise', type=float, default=0.0)
    eval_parser.add_argument('--device', type=str, default='cuda:0')

    # =========================================================================
    # Parse and dispatch
    # =========================================================================
    args = parser.parse_args()

    if args.command == 'collect':
        collect_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'train_all':
        train_all_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'eval':
        eval_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
