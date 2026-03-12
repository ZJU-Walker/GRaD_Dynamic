"""
Velocity Network Trainer with Direct Delta-V Prediction.

Scheduled sampling gradually transitions from teacher forcing (using GT prev_vel)
to auto-regressive inference (using predicted prev_vel). This prevents the model
from learning the "copying shortcut" where it just outputs prev_vel.

DIRECT DELTA-V MODE:
The network directly predicts velocity change from observations:
    v_t = v_{t-1} + Network(obs)

Where:
- obs: Includes normalized IMU acceleration as an input feature (indicator)
- Network(obs): Directly predicts delta_v (velocity change)

IMU acceleration is kept in the observation as a feature but is NOT used for
physics integration. IMU noise augmentation during training helps the network
learn to be robust to noisy IMU readings:
- Bias drift: constant offset per sequence
- Scale error: multiplicative factor
- White noise: per-frame Gaussian
- Sensor dropout: random zeroing

Training stages:
- Start: 100% teacher forcing (use GT prev_vel)
- Decay: Linearly decrease teacher forcing ratio over N epochs
- End: 0% teacher forcing (use predicted prev_vel)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, List
import numpy as np
from tqdm import tqdm

from models.vel_net_body_legacy import VELO_NET_BODY
from models.vel_net.visual_encoder import DualEncoder  # Shared visual encoder

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class VelNetTrainer:
    """
    Trainer with scheduled sampling for velocity network.

    Key features:
    - Processes sequences step-by-step (not batched single frames)
    - Gradually replaces GT prev_vel with predicted prev_vel
    - teacher_forcing_ratio decays from 1.0 to target over epochs
    - Auto-regressive validation (0% teacher forcing)

    This breaks the "copying shortcut" where model just outputs prev_vel.
    """

    def __init__(
        self,
        model: VELO_NET_BODY,
        encoder: DualEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        tf_start_epoch: int = 0,
        tf_end_epoch: int = 100,
        grad_accumulation_steps: int = 1,
        device: str = 'cuda:0',
        checkpoint_dir: str = 'checkpoints/vel_net',
        use_wandb: bool = False,
        wandb_project: str = 'vel_net',
    ):
        """
        Initialize trainer.

        Args:
            model: VELO_NET model
            encoder: DualEncoder for visual features
            train_loader: Sequence-based training dataloader
            val_loader: Sequence-based validation dataloader
            lr: Learning rate
            weight_decay: Weight decay
            grad_clip: Gradient clipping
            tf_start_epoch: Epoch to start decaying (before = 100% GT)
            tf_end_epoch: Epoch to finish decaying (after = 0% GT, use predicted)
            device: PyTorch device
            checkpoint_dir: Checkpoint directory
            use_wandb: Enable wandb logging
            wandb_project: Wandb project name
        """
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.lr = lr
        self.grad_clip = grad_clip
        self.tf_start_epoch = tf_start_epoch
        self.tf_end_epoch = tf_end_epoch
        self.grad_accumulation_steps = grad_accumulation_steps

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(encoder.get_trainable_params()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6,
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_ar_mae = float('inf')
        self.epochs_without_improvement = 0

        # Compute normalization stats from training data
        # vel_mean/std: for normalizing prev_vel INPUT
        # accel_mean/std: for normalizing acceleration INPUT
        # delta_mean/std: for normalizing delta_v OUTPUT (target)
        stats = self._compute_velocity_stats()
        self.vel_mean, self.vel_std, self.accel_mean, self.accel_std, self.delta_mean, self.delta_std = stats
        print(f"  Velocity normalization (input): mean={self.vel_mean.cpu().numpy()}, std={self.vel_std.cpu().numpy()}")
        print(f"  Accel normalization (input):    mean={self.accel_mean.cpu().numpy()}, std={self.accel_std.cpu().numpy()}")
        print(f"  Delta normalization (output):   mean={self.delta_mean.cpu().numpy()}, std={self.delta_std.cpu().numpy()}")

        # Wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'tf_start_epoch': tf_start_epoch,
                    'tf_end_epoch': tf_end_epoch,
                    'model_params': sum(p.numel() for p in model.parameters()),
                },
            )

    def _compute_velocity_stats(self) -> Tuple[torch.Tensor, ...]:
        """Compute velocity, acceleration, and delta statistics from training data.

        Returns:
            vel_mean, vel_std: For normalizing prev_vel INPUT
            accel_mean, accel_std: For normalizing acceleration INPUT
            delta_mean, delta_std: For normalizing delta_v OUTPUT (target)
        """
        all_vels = []
        all_accels = []
        all_deltas = []

        for batch in self.train_loader:
            for b_idx in range(len(batch['velocities_gt'])):
                vel_gt = batch['velocities_gt'][b_idx].numpy()
                prev_vel = batch['initial_prev_vels'][b_idx].numpy()
                accel = batch['accel_gt'][b_idx].numpy()  # Use GT accel for stats

                all_vels.append(vel_gt)
                all_accels.append(accel)

                # Compute delta_v: vel_gt[t] - prev_vel[t] for each timestep
                # For first frame, delta = vel_gt[0] - initial_prev_vel
                # For subsequent frames, delta = vel_gt[t] - vel_gt[t-1]
                deltas = np.zeros_like(vel_gt)
                deltas[0] = vel_gt[0] - prev_vel  # First frame delta
                deltas[1:] = vel_gt[1:] - vel_gt[:-1]  # Subsequent frame deltas
                all_deltas.append(deltas)

        all_vels = np.concatenate(all_vels, axis=0)
        all_accels = np.concatenate(all_accels, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        # Velocity stats (for prev_vel input normalization)
        vel_mean = torch.from_numpy(all_vels.mean(axis=0).astype(np.float32)).to(self.device)
        vel_std = torch.from_numpy(all_vels.std(axis=0).astype(np.float32)).to(self.device)
        vel_std = torch.clamp(vel_std, min=1e-6)

        # Acceleration stats (for accel input normalization)
        accel_mean = torch.from_numpy(all_accels.mean(axis=0).astype(np.float32)).to(self.device)
        accel_std = torch.from_numpy(all_accels.std(axis=0).astype(np.float32)).to(self.device)
        accel_std = torch.clamp(accel_std, min=1e-6)

        # Delta stats (for delta_v output normalization)
        delta_mean = torch.from_numpy(all_deltas.mean(axis=0).astype(np.float32)).to(self.device)
        delta_std = torch.from_numpy(all_deltas.std(axis=0).astype(np.float32)).to(self.device)
        delta_std = torch.clamp(delta_std, min=1e-6)

        return vel_mean, vel_std, accel_mean, accel_std, delta_mean, delta_std

    def normalize_velocity(self, vel: torch.Tensor) -> torch.Tensor:
        """Normalize velocity to zero-mean, unit-variance (for input)."""
        return (vel - self.vel_mean) / self.vel_std

    def denormalize_velocity(self, vel_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize velocity back to original scale."""
        return vel_norm * self.vel_std + self.vel_mean

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Normalize delta (velocity change) to zero-mean, unit-variance."""
        return (delta - self.delta_mean) / self.delta_std

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize delta back to original scale (m/s)."""
        return delta_norm * self.delta_std + self.delta_mean

    def normalize_accel(self, accel: torch.Tensor) -> torch.Tensor:
        """Normalize acceleration to zero-mean, unit-variance (for input)."""
        return (accel - self.accel_mean) / self.accel_std

    def get_teacher_forcing_ratio(self) -> float:
        """
        Get current teacher forcing ratio based on epoch.

        - Before tf_start_epoch: 1.0 (100% GT)
        - Between tf_start_epoch and tf_end_epoch: linear decay
        - After tf_end_epoch: 0.0 (0% GT, use predicted)
        """
        if self.epoch < self.tf_start_epoch:
            return 1.0  # 100% GT
        if self.epoch >= self.tf_end_epoch:
            return 0.0  # 0% GT (use predicted)

        # Linear decay between start and end
        progress = (self.epoch - self.tf_start_epoch) / (self.tf_end_epoch - self.tf_start_epoch)
        return 1.0 - progress  # 1.0 -> 0.0

    def _load_images_for_frame(self, seq_path: str, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load RGB and depth images for a single frame."""
        from PIL import Image

        seq_path = Path(seq_path)
        rgb_path = seq_path / "rgb" / f"{frame_idx:06d}.png"
        depth_path = seq_path / "depth" / f"{frame_idx:06d}.npy"

        rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
        depth = np.load(depth_path).astype(np.float32).squeeze()

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return rgb_tensor, depth_tensor

    def _load_backbone_features(self, seq_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load precomputed backbone features for a sequence (cached on GPU)."""
        if not hasattr(self, '_feature_cache'):
            self._feature_cache = {}

        if seq_path not in self._feature_cache:
            feature_path = Path(seq_path) / "backbone_features.npz"
            features = np.load(feature_path)
            # Convert to tensor and move to GPU once (cached)
            rgb_tensor = torch.from_numpy(features['rgb_features']).to(self.device)
            depth_tensor = torch.from_numpy(features['depth_features']).to(self.device)
            self._feature_cache[seq_path] = (rgb_tensor, depth_tensor)

        return self._feature_cache[seq_path]

    def _check_precomputed_available(self) -> bool:
        """Check if precomputed features exist."""
        for batch in self.train_loader:
            for seq_path in batch['seq_paths']:
                if not (Path(seq_path) / "backbone_features.npz").exists():
                    return False
            break
        return True

    def train_epoch(self, pbar: tqdm = None) -> Dict[str, float]:
        """Train for one epoch with scheduled sampling and direct delta-v prediction.

        DIRECT DELTA-V MODE: Model outputs delta_v, velocity = prev_vel + delta_v
        vel_pred = prev_vel + delta_v

        Loss is computed on the delta_v (normalized).
        IMU acceleration is used as an input feature but NOT for physics integration.
        """
        from models.vel_net_body_legacy.vel_obs_utils_body import quaternion_to_rot6d, transform_worldvel_to_bodyvel

        self.model.train()
        self.encoder.train()

        tf_ratio = self.get_teacher_forcing_ratio()
        dt = self.model.dt  # Time step for physics integration

        total_loss = 0.0
        total_vel_error = 0.0  # Track reconstructed velocity error for monitoring
        total_tf_frames = 0
        total_pred_frames = 0
        n_sequences = 0

        # Use AMP for faster training
        scaler = torch.cuda.amp.GradScaler()

        # Gradient accumulation
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        batch_idx = 0

        for batch in self.train_loader:
            batch_loss = 0.0
            batch_vel_error = 0.0
            batch_frames = 0

            with torch.cuda.amp.autocast():
                for b_idx in range(len(batch['seq_paths'])):
                    seq_path = batch['seq_paths'][b_idx]
                    frame_indices = batch['frame_indices'][b_idx]
                    orientations = batch['orientations'][b_idx].to(self.device)
                    actions = batch['actions'][b_idx].to(self.device)
                    prev_actions = batch['prev_actions'][b_idx].to(self.device)
                    velocities_gt_raw = batch['velocities_gt'][b_idx].to(self.device)  # Raw GT (m/s)
                    prev_vel_raw = batch['initial_prev_vels'][b_idx].to(self.device)  # Raw (m/s)
                    accel_aug = batch['accel_aug'][b_idx].to(self.device)  # Corrupted IMU (m/s^2)

                    # For model INPUT: normalize prev_vel
                    prev_vel_norm = self.normalize_velocity(prev_vel_raw)
                    # Keep raw prev_vel for physics integration
                    prev_vel_raw_current = prev_vel_raw.clone()

                    # Load precomputed backbone features (already on GPU)
                    rgb_backbone, depth_backbone = self._load_backbone_features(seq_path)

                    seq_length = len(frame_indices)
                    self.model.reset_hidden_state(batch_size=1)

                    # Batch FC layer: process ALL frames at once (576 -> 32)
                    frame_idx_tensor = torch.tensor(frame_indices, device=self.device)
                    rgb_bb_all = rgb_backbone[frame_idx_tensor]  # (seq_len, 576)
                    depth_bb_all = depth_backbone[frame_idx_tensor]  # (seq_len, 576)
                    rgb_feat_all, depth_feat_all = self.encoder.forward_from_backbone_features(rgb_bb_all, depth_bb_all)

                    # Precompute rot6d for all frames
                    rot6d_all = quaternion_to_rot6d(orientations)  # (seq_len, 6)

                    # Precompute GT delta_v for the entire sequence (direct delta-v mode)
                    # delta_v_gt[t] = vel_gt[t] - prev_vel[t]
                    # For stable targets, use GT prev_vel
                    delta_v_gt_raw = torch.zeros_like(velocities_gt_raw)
                    # First frame: delta = vel_gt[0] - initial_prev_vel
                    delta_v_gt_raw[0] = velocities_gt_raw[0] - prev_vel_raw
                    # Rest of frames: delta = vel_gt[t] - vel_gt[t-1]
                    delta_v_gt_raw[1:] = velocities_gt_raw[1:] - velocities_gt_raw[:-1]

                    # Normalize acceleration for input
                    accel_aug_norm = self.normalize_accel(accel_aug)  # (seq_len, 3)

                    # Sequential GRU loop (needed for scheduled sampling)
                    for t in range(seq_length):
                        # Build observation with normalized IMU accel (76 dims, no action)
                        obs = torch.cat([
                            rot6d_all[t:t+1],
                            prev_vel_norm.unsqueeze(0),  # Normalized prev_vel as input
                            rgb_feat_all[t:t+1],
                            depth_feat_all[t:t+1],
                            accel_aug_norm[t:t+1],  # Normalized IMU acceleration
                        ], dim=1)

                        # Model outputs delta_v (normalized)
                        delta_v_mu_norm, _ = self.model.encode_step(obs)

                        # GT delta_v: vel_gt[t] - prev_vel[t] (direct delta-v)
                        # For stable targets, use GT-based delta
                        delta_v_gt_t = delta_v_gt_raw[t:t+1]
                        delta_v_gt_norm = self.normalize_delta(delta_v_gt_t)

                        # Loss on normalized delta_v
                        loss = F.mse_loss(delta_v_mu_norm, delta_v_gt_norm)
                        batch_loss += loss
                        batch_frames += 1

                        # Reconstruct velocity: vel_pred = prev_vel + delta_v (no physics integration)
                        delta_v_pred_raw = self.denormalize_delta(delta_v_mu_norm)
                        vel_pred_raw = prev_vel_raw_current.unsqueeze(0) + delta_v_pred_raw
                        vel_gt_raw = velocities_gt_raw[t:t+1]

                        # Track velocity reconstruction error (for monitoring)
                        batch_vel_error += F.l1_loss(vel_pred_raw, vel_gt_raw).item()

                        # Scheduled sampling: choose GT or predicted for next prev_vel
                        if np.random.random() < tf_ratio:
                            # Teacher forcing: use GT velocity
                            prev_vel_raw_current = vel_gt_raw.squeeze(0).detach()
                            prev_vel_norm = self.normalize_velocity(prev_vel_raw_current)
                            total_tf_frames += 1
                        else:
                            # Use predicted velocity
                            prev_vel_raw_current = vel_pred_raw.squeeze(0).detach()
                            prev_vel_norm = self.normalize_velocity(prev_vel_raw_current)
                            total_pred_frames += 1

                    n_sequences += 1

            if batch_frames > 0:
                avg_loss = batch_loss / batch_frames
                # Scale loss for gradient accumulation
                scaled_loss = avg_loss / self.grad_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += avg_loss.item()
                total_vel_error += batch_vel_error / batch_frames
                batch_idx += 1

                # Update weights every grad_accumulation_steps batches
                if batch_idx % self.grad_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.encoder.get_trainable_params()),
                        self.grad_clip
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    total_loss += accumulated_loss / self.grad_accumulation_steps
                    accumulated_loss = 0.0

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{total_loss / max(1, n_sequences // len(batch["seq_paths"])):.4f}',
                    'tf': f'{tf_ratio:.2f}',
                })

        # Handle remaining accumulated gradients
        if batch_idx % self.grad_accumulation_steps != 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.get_trainable_params()),
                self.grad_clip
            )
            scaler.step(self.optimizer)
            scaler.update()
            remaining = batch_idx % self.grad_accumulation_steps
            total_loss += accumulated_loss / remaining

        n_batches = len(self.train_loader)
        total_frames = total_tf_frames + total_pred_frames
        actual_tf_ratio = total_tf_frames / max(1, total_frames)

        return {
            'train/loss': total_loss / n_batches,
            'train/vel_mae': total_vel_error / n_batches,  # Reconstructed velocity MAE
            'train/tf_ratio_target': tf_ratio,
            'train/tf_ratio_actual': actual_tf_ratio,
            'train/tf_frames': total_tf_frames,
            'train/pred_frames': total_pred_frames,
        }

    @torch.no_grad()
    def validate_autoregressive(self) -> Dict[str, float]:
        """Validate using auto-regressive inference with direct delta-v prediction.

        DIRECT DELTA-V MODE: Model outputs delta_v, velocity = prev_vel + delta_v
        Note: Validation uses clean accel (no augmentation) as input feature.
        """
        from models.vel_net_body_legacy.vel_obs_utils_body import quaternion_to_rot6d, transform_worldvel_to_bodyvel

        self.model.eval()
        self.encoder.eval()

        all_errors = []
        total_mse = 0.0
        total_frames = 0

        for batch in self.val_loader:
            for b_idx in range(len(batch['seq_paths'])):
                seq_path = batch['seq_paths'][b_idx]
                frame_indices = batch['frame_indices'][b_idx]
                orientations = batch['orientations'][b_idx].to(self.device)
                actions = batch['actions'][b_idx].to(self.device)
                prev_actions = batch['prev_actions'][b_idx].to(self.device)
                velocities_gt_raw = batch['velocities_gt'][b_idx].to(self.device)  # Raw GT for metrics
                prev_vel_raw = batch['initial_prev_vels'][b_idx].to(self.device)  # Raw (m/s)
                # Validation uses clean accel (accel_aug = accel_gt for val dataset)
                accel = batch['accel_aug'][b_idx].to(self.device)  # Clean accel for validation

                # For model INPUT: normalize prev_vel
                prev_vel_norm = self.normalize_velocity(prev_vel_raw)
                # Keep raw prev_vel for velocity reconstruction
                prev_vel_raw_current = prev_vel_raw.clone()

                # Normalize acceleration for input
                accel_norm = self.normalize_accel(accel)  # (seq_len, 3)

                # Load precomputed backbone features (already on GPU)
                rgb_backbone, depth_backbone = self._load_backbone_features(seq_path)

                seq_length = len(frame_indices)
                self.model.reset_hidden_state(batch_size=1)

                # Batch FC layer: process ALL frames at once
                frame_idx_tensor = torch.tensor(frame_indices, device=self.device)
                rgb_bb_all = rgb_backbone[frame_idx_tensor]
                depth_bb_all = depth_backbone[frame_idx_tensor]
                rgb_feat_all, depth_feat_all = self.encoder.forward_from_backbone_features(rgb_bb_all, depth_bb_all)

                # Precompute rot6d for all frames
                rot6d_all = quaternion_to_rot6d(orientations)

                for t in range(seq_length):
                    # Build observation with normalized accel (76 dims, no action)
                    obs = torch.cat([
                        rot6d_all[t:t+1],
                        prev_vel_norm.unsqueeze(0),  # Normalized input
                        rgb_feat_all[t:t+1],
                        depth_feat_all[t:t+1],
                        accel_norm[t:t+1],  # Normalized acceleration as input feature
                    ], dim=1)

                    # Model outputs delta_v (normalized)
                    delta_v_mu_norm, _ = self.model.encode_step(obs)

                    # Reconstruct velocity: vel_pred = prev_vel + delta_v (direct delta-v)
                    delta_v_raw = self.denormalize_delta(delta_v_mu_norm)
                    vel_pred = prev_vel_raw_current.unsqueeze(0) + delta_v_raw
                    vel_gt = velocities_gt_raw[t:t+1]  # Raw GT in m/s

                    error = torch.abs(vel_pred - vel_gt).cpu().numpy()[0]
                    all_errors.append(error)
                    total_mse += F.mse_loss(vel_pred, vel_gt).item()
                    total_frames += 1

                    # Auto-regressive: use predicted velocity for next step
                    prev_vel_raw_current = vel_pred.squeeze(0)
                    prev_vel_norm = self.normalize_velocity(prev_vel_raw_current)

        all_errors = np.array(all_errors)
        mae = np.mean(all_errors)
        mae_xyz = np.mean(all_errors, axis=0)

        return {
            'val/ar_mse': total_mse / max(1, total_frames),
            'val/ar_mae': mae,
            'val/ar_mae_x': mae_xyz[0],
            'val/ar_mae_y': mae_xyz[1],
            'val/ar_mae_z': mae_xyz[2],
            'val/ar_rmse': np.sqrt(total_mse / max(1, total_frames)),
        }

    def train(self, n_epochs: int = 200, early_stop_patience: int = 30) -> Dict[str, List[float]]:
        """Main training loop with scheduled sampling."""
        history = {
            'train_loss': [],
            'val_ar_mae': [],
            'tf_ratio': [],
        }

        # Check for precomputed features
        if not self._check_precomputed_available():
            raise RuntimeError(
                "Precomputed backbone features not found!\n"
                "Run first: python training/vel_net/precompute_features.py "
                f"--data_dir <your_data_dir>"
            )

        print(f"\n{'='*60}")
        print(f"Velocity Network Training (Scheduled Sampling)")
        print(f"{'='*60}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Encoder trainable: {self.encoder.num_trainable_params():,}")
        print(f"  TF schedule: 100% GT (epoch 0-{self.tf_start_epoch}) -> decay -> 0% GT (epoch {self.tf_end_epoch}+)")
        print(f"  Using precomputed backbone features (fast mode)")
        print(f"{'='*60}\n")

        n_batches = len(self.train_loader)

        for epoch in range(n_epochs):
            self.epoch = epoch
            tf_ratio = self.get_teacher_forcing_ratio()

            batch_pbar = tqdm(
                total=n_batches,
                desc=f'Epoch {epoch:3d} [TF={tf_ratio:.2f}]',
                leave=False,
            )
            train_metrics = self.train_epoch(pbar=batch_pbar)
            batch_pbar.close()

            val_metrics = self.validate_autoregressive()
            self.scheduler.step(val_metrics['val/ar_mae'])

            if val_metrics['val/ar_mae'] < self.best_ar_mae:
                self.best_ar_mae = val_metrics['val/ar_mae']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best.pt')
            else:
                self.epochs_without_improvement += 1

            history['train_loss'].append(train_metrics['train/loss'])
            history['val_ar_mae'].append(val_metrics['val/ar_mae'])
            history['tf_ratio'].append(tf_ratio)

            if self.use_wandb:
                metrics = {**train_metrics, **val_metrics}
                metrics['epoch'] = epoch
                metrics['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(metrics)

            print(f"Epoch {epoch:3d} | TF={tf_ratio:.2f} | "
                  f"Loss={train_metrics['train/loss']:.4f} | "
                  f"AR_MAE={val_metrics['val/ar_mae']:.4f} | "
                  f"LR={self.optimizer.param_groups[0]['lr']:.1e}")

            if self.epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            if epoch % 50 == 0 and epoch > 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        self.save_checkpoint('final.pt')

        if self.use_wandb:
            wandb.finish()

        print(f"\nTraining complete! Best AR_MAE: {self.best_ar_mae:.4f}")
        return history

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_ar_mae': self.best_ar_mae,
            'tf_start_epoch': self.tf_start_epoch,
            'tf_end_epoch': self.tf_end_epoch,
            'model_config': {
                'num_obs': self.model.num_obs,
                'stack_size': self.model.stack_size,
                'hidden_dim': self.model.hidden_dim,
                'gru_layers': self.model.gru_layers,
                'dt': self.model.dt,
            },
            # Normalization stats (needed for inference)
            # vel_mean/std: for normalizing prev_vel INPUT
            'vel_mean': self.vel_mean.cpu(),
            'vel_std': self.vel_std.cpu(),
            # accel_mean/std: for normalizing acceleration INPUT
            'accel_mean': self.accel_mean.cpu(),
            'accel_std': self.accel_std.cpu(),
            # delta_mean/std: for denormalizing delta_v OUTPUT (DIRECT DELTA-V MODE)
            'delta_mean': self.delta_mean.cpu(),
            'delta_std': self.delta_std.cpu(),
            # Mode flags
            'direct_delta_mode': True,  # vel = prev_vel + delta_v (no physics integration)
            'residual_mode': True,      # For backward compatibility
        }
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_ar_mae = checkpoint.get('best_ar_mae', float('inf'))
        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")


def autoregressive_test(
    model: VELO_NET_BODY,
    encoder: DualEncoder,
    test_sequence_path: str,
    vel_mean: torch.Tensor,
    vel_std: torch.Tensor,
    accel_mean: torch.Tensor = None,
    accel_std: torch.Tensor = None,
    delta_mean: torch.Tensor = None,
    delta_std: torch.Tensor = None,
    device: str = 'cuda:0',
    max_steps: int = 300,
    save_plot: bool = True,
    plot_path: str = None,
) -> Dict[str, float]:
    """
    Test model with auto-regressive inference using direct delta-v prediction.

    DIRECT DELTA-V MODE: vel_pred = prev_vel + delta_v

    Args:
        vel_mean: Velocity mean for normalizing INPUT prev_vel (3,)
        vel_std: Velocity std for normalizing INPUT prev_vel (3,)
        accel_mean: Accel mean for normalizing INPUT accel (3,)
        accel_std: Accel std for normalizing INPUT accel (3,)
        delta_mean: Delta mean for denormalizing delta_v OUTPUT (3,)
        delta_std: Delta std for denormalizing delta_v OUTPUT (3,)
    """
    from PIL import Image
    from models.vel_net_body_legacy.vel_obs_utils_body import quaternion_to_rot6d, transform_worldvel_to_bodyvel

    model.eval()
    encoder.eval()

    vel_mean = vel_mean.to(device)
    vel_std = vel_std.to(device)
    accel_mean = accel_mean.to(device) if accel_mean is not None else torch.zeros(3, device=device)
    accel_std = accel_std.to(device) if accel_std is not None else torch.ones(3, device=device)
    delta_mean = delta_mean.to(device) if delta_mean is not None else torch.zeros(3, device=device)
    delta_std = delta_std.to(device) if delta_std is not None else torch.ones(3, device=device)
    dt = model.dt  # Time step for computing acceleration from velocities

    seq_path = Path(test_sequence_path)
    telemetry = np.load(seq_path / "telemetry.npz")

    n_frames = min(len(telemetry['timestamps']), max_steps)

    # Load world frame data
    velocities_world = telemetry['velocities'].astype(np.float32)
    orientations = telemetry['orientations'].astype(np.float32)

    # Compute acceleration in world frame first
    accel_world = np.zeros_like(velocities_world)
    accel_world[1:] = (velocities_world[1:] - velocities_world[:-1]) / dt
    accel_world[0] = accel_world[1]  # First frame uses next frame's accel

    # Transform velocities and acceleration to body frame
    velocities_world_t = torch.from_numpy(velocities_world).to(device)
    orientations_t = torch.from_numpy(orientations).to(device)
    accel_world_t = torch.from_numpy(accel_world).to(device)

    velocities_body = transform_worldvel_to_bodyvel(velocities_world_t, orientations_t).cpu().numpy()
    accel_body = transform_worldvel_to_bodyvel(accel_world_t, orientations_t).cpu().numpy()

    all_preds = []
    all_gts = []
    errors = []

    # Start with zero velocity (raw)
    prev_vel_raw = torch.zeros(3, device=device)
    prev_vel_norm = (prev_vel_raw - vel_mean) / vel_std

    # Reset GRU hidden state for sequential processing
    model.reset_hidden_state(batch_size=1)

    with torch.no_grad():
        for t in range(1, n_frames):
            rgb_path = seq_path / "rgb" / f"{t:06d}.png"
            depth_path = seq_path / "depth" / f"{t:06d}.npy"

            rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
            depth = np.load(depth_path).astype(np.float32).squeeze()

            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

            rgb_feat, depth_feat = encoder(rgb_tensor, depth_tensor)

            quat = torch.from_numpy(telemetry['orientations'][t].astype(np.float32)).unsqueeze(0).to(device)
            rot6d = quaternion_to_rot6d(quat)

            action = torch.from_numpy(telemetry['actions'][t].astype(np.float32)).unsqueeze(0).to(device)
            prev_action = torch.from_numpy(telemetry['actions'][t-1].astype(np.float32)).unsqueeze(0).to(device)

            # Get body frame acceleration for this frame and normalize
            accel = torch.from_numpy(accel_body[t].astype(np.float32)).unsqueeze(0).to(device)
            accel_norm = (accel - accel_mean) / accel_std

            # Build observation with normalized accel (76 dims, no action)
            obs = torch.cat([rot6d, prev_vel_norm.unsqueeze(0), rgb_feat, depth_feat, accel_norm], dim=1)

            # Model outputs delta_v (normalized)
            delta_v_mu_norm, _ = model.encode_step(obs)

            # Direct delta-v: vel_pred = prev_vel + delta_v
            delta_v_raw = delta_v_mu_norm.squeeze(0) * delta_std + delta_mean
            vel_pred = prev_vel_raw + delta_v_raw

            # Update for next step
            prev_vel_raw = vel_pred.detach()
            prev_vel_norm = (prev_vel_raw - vel_mean) / vel_std

            vel_gt = velocities_body[t]  # Body frame GT
            all_preds.append(vel_pred.cpu().numpy())
            all_gts.append(vel_gt)
            errors.append(np.abs(vel_pred.cpu().numpy() - vel_gt))

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    errors = np.array(errors)

    metrics = {
        'ar/mae': np.mean(errors),
        'ar/mae_x': np.mean(errors[:, 0]),
        'ar/mae_y': np.mean(errors[:, 1]),
        'ar/mae_z': np.mean(errors[:, 2]),
        'ar/max_error': np.max(errors),
        'ar/rmse': np.sqrt(np.mean(errors**2)),
    }

    print(f"\nAuto-regressive Test (Body Frame, Direct Delta-V, {n_frames-1} steps):")
    print(f"  MAE: {metrics['ar/mae']:.4f} m/s")
    print(f"  MAE (x,y,z): [{metrics['ar/mae_x']:.4f}, {metrics['ar/mae_y']:.4f}, {metrics['ar/mae_z']:.4f}]")
    print(f"  RMSE: {metrics['ar/rmse']:.4f} m/s")

    if save_plot:
        import matplotlib.pyplot as plt

        timestamps = telemetry['timestamps'][1:n_frames]
        t_axis = timestamps - timestamps[0]

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(t_axis, all_gts[:, 0], 'b-', label='GT', linewidth=1.5)
        axes[0].plot(t_axis, all_preds[:, 0], 'r--', label='Pred', linewidth=1.5)
        axes[0].set_ylabel('Vel X (m/s)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'Body Frame Velocity Prediction (Direct Delta-V) | MAE: {metrics["ar/mae"]:.4f} m/s')

        axes[1].plot(t_axis, all_gts[:, 1], 'b-', label='GT', linewidth=1.5)
        axes[1].plot(t_axis, all_preds[:, 1], 'r--', label='Pred', linewidth=1.5)
        axes[1].set_ylabel('Vel Y (m/s)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_axis, all_gts[:, 2], 'b-', label='GT', linewidth=1.5)
        axes[2].plot(t_axis, all_preds[:, 2], 'r--', label='Pred', linewidth=1.5)
        axes[2].set_ylabel('Vel Z (m/s)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        vel_mag_gt = np.linalg.norm(all_gts, axis=1)
        vel_mag_pred = np.linalg.norm(all_preds, axis=1)
        axes[3].plot(t_axis, vel_mag_gt, 'b-', label='GT', linewidth=1.5)
        axes[3].plot(t_axis, vel_mag_pred, 'r--', label='Pred', linewidth=1.5)
        axes[3].set_ylabel('Vel Mag (m/s)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        if plot_path is None:
            plot_path = seq_path / 'vel_prediction.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")

    return metrics


if __name__ == '__main__':
    print("Trainer module loaded successfully")
    print(f"Wandb available: {WANDB_AVAILABLE}")
