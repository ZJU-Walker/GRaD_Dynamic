"""
Velocity Network Trainer with Scheduled Sampling and Residual Prediction.

Scheduled sampling gradually transitions from teacher forcing (using GT prev_vel)
to auto-regressive inference (using predicted prev_vel). This prevents the model
from learning the "copying shortcut" where it just outputs prev_vel.

RESIDUAL PREDICTION MODE:
Instead of predicting absolute velocity, the model predicts velocity CHANGE (delta):
    vel_pred = prev_vel + model(obs)
This fixes the systematic vx underestimation bias by:
- Centering the output around 0 (delta mean ≈ 0)
- Smaller output range (delta std ≈ 0.003 vs velocity std ≈ 0.19)
- Model defaults to "maintain current velocity" instead of "regress to training mean"

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

from models.vel_net import VELO_NET
from models.vel_net.visual_encoder import DualEncoder

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
        model: VELO_NET,
        encoder: DualEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        tf_start_epoch: int = 0,
        tf_end_epoch: int = 100,
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
        # vel_mean/std: for normalizing prev_vel INPUT to the model
        # delta_mean/std: for normalizing the delta OUTPUT (target)
        self.vel_mean, self.vel_std, self.delta_mean, self.delta_std = self._compute_velocity_stats()
        print(f"  Velocity normalization (input): mean={self.vel_mean.cpu().numpy()}, std={self.vel_std.cpu().numpy()}")
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

    def _compute_velocity_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute velocity and delta statistics from training data for normalization.

        Returns:
            vel_mean, vel_std: For normalizing prev_vel INPUT
            delta_mean, delta_std: For normalizing delta OUTPUT (target)
        """
        all_vels = []
        all_deltas = []

        for batch in self.train_loader:
            for b_idx in range(len(batch['velocities_gt'])):
                vel_gt = batch['velocities_gt'][b_idx].numpy()
                prev_vel = batch['initial_prev_vels'][b_idx].numpy()

                all_vels.append(vel_gt)

                # Compute delta: vel_gt[t] - prev_vel[t] for each timestep
                # For first frame, delta = vel_gt[0] - initial_prev_vel
                # For subsequent frames, delta = vel_gt[t] - vel_gt[t-1]
                deltas = np.zeros_like(vel_gt)
                deltas[0] = vel_gt[0] - prev_vel  # First frame delta
                deltas[1:] = vel_gt[1:] - vel_gt[:-1]  # Subsequent frame deltas
                all_deltas.append(deltas)

        all_vels = np.concatenate(all_vels, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        # Velocity stats (for input normalization)
        vel_mean = torch.from_numpy(all_vels.mean(axis=0).astype(np.float32)).to(self.device)
        vel_std = torch.from_numpy(all_vels.std(axis=0).astype(np.float32)).to(self.device)
        vel_std = torch.clamp(vel_std, min=1e-6)

        # Delta stats (for output normalization)
        delta_mean = torch.from_numpy(all_deltas.mean(axis=0).astype(np.float32)).to(self.device)
        delta_std = torch.from_numpy(all_deltas.std(axis=0).astype(np.float32)).to(self.device)
        delta_std = torch.clamp(delta_std, min=1e-6)

        return vel_mean, vel_std, delta_mean, delta_std

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
        """Train for one epoch with scheduled sampling and RESIDUAL prediction.

        RESIDUAL MODE: Model outputs delta (velocity change), not absolute velocity.
        vel_pred = prev_vel + delta_pred
        Loss is computed on normalized delta.
        """
        from models.vel_net.vel_obs_utils import quaternion_to_rot6d

        self.model.train()
        self.encoder.train()

        tf_ratio = self.get_teacher_forcing_ratio()

        total_loss = 0.0
        total_vel_error = 0.0  # Track reconstructed velocity error for monitoring
        total_tf_frames = 0
        total_pred_frames = 0
        n_sequences = 0

        # Use AMP for faster training
        scaler = torch.cuda.amp.GradScaler()

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

                    # For model INPUT: normalize prev_vel
                    prev_vel_norm = self.normalize_velocity(prev_vel_raw)
                    # Keep raw prev_vel for delta calculation
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

                    # Precompute GT deltas for the entire sequence (always GT-based)
                    # delta_gt[t] = vel_gt[t] - vel_gt[t-1]
                    # This ensures stable targets regardless of scheduled sampling
                    deltas_gt_raw = torch.zeros_like(velocities_gt_raw)
                    deltas_gt_raw[0] = velocities_gt_raw[0] - prev_vel_raw  # First frame
                    deltas_gt_raw[1:] = velocities_gt_raw[1:] - velocities_gt_raw[:-1]  # Rest

                    # Sequential GRU loop (needed for scheduled sampling)
                    for t in range(seq_length):
                        obs = torch.cat([
                            rot6d_all[t:t+1],
                            actions[t:t+1],
                            prev_actions[t:t+1],
                            prev_vel_norm.unsqueeze(0),  # Normalized prev_vel as input
                            rgb_feat_all[t:t+1],
                            depth_feat_all[t:t+1],
                        ], dim=1)

                        # Model outputs DELTA (normalized)
                        delta_mu_norm, _ = self.model.encode_step(obs)

                        # GT delta: ALWAYS from GT velocities (not from potentially wrong prev_vel)
                        delta_gt_raw = deltas_gt_raw[t:t+1]
                        delta_gt_norm = self.normalize_delta(delta_gt_raw)

                        # Loss on normalized delta
                        loss = F.mse_loss(delta_mu_norm, delta_gt_norm)
                        batch_loss += loss
                        batch_frames += 1

                        # Reconstruct velocity for next step
                        delta_pred_raw = self.denormalize_delta(delta_mu_norm)
                        vel_pred_raw = prev_vel_raw_current.unsqueeze(0) + delta_pred_raw
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
                self.optimizer.zero_grad()
                scaler.scale(avg_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.encoder.get_trainable_params()),
                    self.grad_clip
                )
                scaler.step(self.optimizer)
                scaler.update()
                total_loss += avg_loss.item()
                total_vel_error += batch_vel_error / batch_frames

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{total_loss / max(1, n_sequences // len(batch["seq_paths"])):.4f}',
                    'tf': f'{tf_ratio:.2f}',
                })

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
        """Validate using auto-regressive inference with RESIDUAL prediction.

        RESIDUAL MODE: Model outputs delta, velocity = prev_vel + delta
        """
        from models.vel_net.vel_obs_utils import quaternion_to_rot6d

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

                # For model INPUT: normalize prev_vel
                prev_vel_norm = self.normalize_velocity(prev_vel_raw)
                # Keep raw prev_vel for residual calculation
                prev_vel_raw_current = prev_vel_raw.clone()

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
                    obs = torch.cat([
                        rot6d_all[t:t+1],
                        actions[t:t+1],
                        prev_actions[t:t+1],
                        prev_vel_norm.unsqueeze(0),  # Normalized input
                        rgb_feat_all[t:t+1],
                        depth_feat_all[t:t+1],
                    ], dim=1)

                    # Model outputs DELTA (normalized)
                    delta_mu_norm, _ = self.model.encode_step(obs)

                    # Reconstruct velocity: vel_pred = prev_vel + delta
                    delta_raw = self.denormalize_delta(delta_mu_norm)
                    vel_pred = prev_vel_raw_current.unsqueeze(0) + delta_raw
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
            },
            # Normalization stats (needed for inference)
            # vel_mean/std: for normalizing prev_vel INPUT
            'vel_mean': self.vel_mean.cpu(),
            'vel_std': self.vel_std.cpu(),
            # delta_mean/std: for denormalizing delta OUTPUT (RESIDUAL MODE)
            'delta_mean': self.delta_mean.cpu(),
            'delta_std': self.delta_std.cpu(),
            # Flag to indicate this is a residual prediction model
            'residual_mode': True,
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
    model: VELO_NET,
    encoder: DualEncoder,
    test_sequence_path: str,
    vel_mean: torch.Tensor,
    vel_std: torch.Tensor,
    delta_mean: torch.Tensor = None,
    delta_std: torch.Tensor = None,
    device: str = 'cuda:0',
    max_steps: int = 300,
    save_plot: bool = True,
    plot_path: str = None,
) -> Dict[str, float]:
    """
    Test model with auto-regressive inference using RESIDUAL prediction.

    RESIDUAL MODE: Model outputs delta (velocity change).
    vel_pred = prev_vel + delta

    Args:
        vel_mean: Velocity mean for normalizing INPUT prev_vel (3,)
        vel_std: Velocity std for normalizing INPUT prev_vel (3,)
        delta_mean: Delta mean for denormalizing OUTPUT (3,)
        delta_std: Delta std for denormalizing OUTPUT (3,)
    """
    from PIL import Image
    from models.vel_net.vel_obs_utils import quaternion_to_rot6d

    model.eval()
    encoder.eval()

    vel_mean = vel_mean.to(device)
    vel_std = vel_std.to(device)
    delta_mean = delta_mean.to(device) if delta_mean is not None else torch.zeros(3, device=device)
    delta_std = delta_std.to(device) if delta_std is not None else torch.ones(3, device=device)

    seq_path = Path(test_sequence_path)
    telemetry = np.load(seq_path / "telemetry.npz")

    n_frames = min(len(telemetry['timestamps']), max_steps)

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

            obs = torch.cat([rot6d, action, prev_action, prev_vel_norm.unsqueeze(0), rgb_feat, depth_feat], dim=1)

            # Model outputs DELTA (normalized)
            delta_mu_norm, _ = model.encode_step(obs)

            # Denormalize delta and reconstruct velocity
            delta_raw = delta_mu_norm.squeeze(0) * delta_std + delta_mean
            vel_pred = prev_vel_raw + delta_raw

            # Update for next step
            prev_vel_raw = vel_pred.detach()
            prev_vel_norm = (prev_vel_raw - vel_mean) / vel_std

            vel_gt = telemetry['velocities'][t]
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

    print(f"\nAuto-regressive Test ({n_frames-1} steps):")
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
        axes[0].set_title(f'Velocity Prediction (Auto-regressive) | MAE: {metrics["ar/mae"]:.4f} m/s')

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
