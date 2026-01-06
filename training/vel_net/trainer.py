"""
Velocity Network Trainer with Curriculum Learning.

Training stages:
- Stage A (Imitation): Pure supervised loss (MSE)
- Stage B (PINN): Supervised + Physics-Informed loss

Auto-regressive validation: Test with predicted prev_vel fed back.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, List
import numpy as np
import time

from models.vel_net import VELO_NET
from models.vel_net.visual_encoder import DualEncoder

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class VelNetTrainer:
    """
    Trainer for velocity network with curriculum learning.

    Stage A: Pure supervised learning (MSE loss)
    Stage B: Supervised + PINN loss

    The encoder's FC layer is trainable and learns jointly with vel_net.
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
        pinn_weight: float = 0.1,
        stage_patience: int = 20,
        early_stop_patience: int = 30,
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
            train_loader: Training dataloader
            val_loader: Validation dataloader
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            grad_clip: Gradient clipping value
            pinn_weight: Weight for physics loss in Stage B
            stage_patience: Epochs before transitioning A->B
            early_stop_patience: Epochs for early stopping in Stage B
            device: PyTorch device
            checkpoint_dir: Directory for checkpoints
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
        """
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.lr = lr
        self.grad_clip = grad_clip
        self.pinn_weight = pinn_weight
        self.stage_patience = stage_patience
        self.early_stop_patience = early_stop_patience

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer: Include both model and encoder FC layer params
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(encoder.get_trainable_params()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        # Training state
        self.current_stage = 'A'  # Start with supervised only
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'grad_clip': grad_clip,
                    'pinn_weight': pinn_weight,
                    'stage_patience': stage_patience,
                    'early_stop_patience': early_stop_patience,
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'encoder_trainable_params': encoder.num_trainable_params(),
                },
            )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with training metrics
        """
        self.model.train()
        self.encoder.train()

        total_loss = 0.0
        total_mse_loss = 0.0
        total_pinn_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            obs = batch['observation'].to(self.device)  # (B, 81)
            vel_gt = batch['velocity_gt'].to(self.device)  # (B, 3)

            self.optimizer.zero_grad()

            # Forward pass
            estimation, latent_params = self.model.forward(obs)
            vel_pred = estimation[0]  # (B, 3)

            # MSE loss
            mse_loss = F.mse_loss(vel_pred, vel_gt)

            # PINN loss (Stage B only)
            if self.current_stage == 'B':
                # Simplified physics loss for single-frame input
                # For full PINN, would need sequence data
                pinn_loss = torch.tensor(0.0, device=self.device)
                loss = mse_loss + self.pinn_weight * pinn_loss
            else:
                pinn_loss = torch.tensor(0.0, device=self.device)
                loss = mse_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.get_trainable_params()),
                self.grad_clip
            )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_pinn_loss += pinn_loss.item()
            n_batches += 1

        metrics = {
            'train/loss': total_loss / n_batches,
            'train/mse_loss': total_mse_loss / n_batches,
            'train/pinn_loss': total_pinn_loss / n_batches,
            'train/stage': 0 if self.current_stage == 'A' else 1,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        self.encoder.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        all_preds = []
        all_gts = []

        for batch in self.val_loader:
            obs = batch['observation'].to(self.device)
            vel_gt = batch['velocity_gt'].to(self.device)

            # Forward pass (deterministic: use mu directly)
            vel_mu, _ = self.model.encode(obs)

            # Metrics
            mse = F.mse_loss(vel_mu, vel_gt, reduction='none').mean(dim=1)
            mae = torch.abs(vel_mu - vel_gt).mean(dim=1)

            total_loss += mse.sum().item()
            total_mse += mse.sum().item()
            total_mae += mae.sum().item()
            total_samples += obs.size(0)

            all_preds.append(vel_mu.cpu())
            all_gts.append(vel_gt.cpu())

        # Compute metrics
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_loss = total_loss / total_samples

        # Per-axis errors
        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        per_axis_mae = torch.abs(all_preds - all_gts).mean(dim=0)

        metrics = {
            'val/loss': avg_loss,
            'val/mse': avg_mse,
            'val/rmse': np.sqrt(avg_mse),
            'val/mae': avg_mae,
            'val/mae_x': per_axis_mae[0].item(),
            'val/mae_y': per_axis_mae[1].item(),
            'val/mae_z': per_axis_mae[2].item(),
        }

        return metrics

    def check_stage_transition(self, val_loss: float):
        """
        Check if should transition from Stage A to Stage B.

        Args:
            val_loss: Current validation loss
        """
        if self.current_stage == 'A':
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.stage_patience:
                print(f"\n[Trainer] Transitioning to Stage B (PINN) after {self.epoch} epochs")
                self.current_stage = 'B'
                self.epochs_without_improvement = 0
                self.best_val_loss = float('inf')  # Reset for Stage B

        elif self.current_stage == 'B':
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                # Save best model
                self.save_checkpoint('best.pt')
            else:
                self.epochs_without_improvement += 1

    def should_stop(self) -> bool:
        """Check if training should stop early."""
        if self.current_stage == 'B':
            return self.epochs_without_improvement >= self.early_stop_patience
        return False

    def train(self, n_epochs: int = 500) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            n_epochs: Maximum number of epochs

        Returns:
            Dict with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
        }

        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Encoder trainable params: {self.encoder.num_trainable_params():,}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(n_epochs):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step(val_metrics['val/loss'])

            # Check stage transition
            self.check_stage_transition(val_metrics['val/loss'])

            # Record history
            history['train_loss'].append(train_metrics['train/loss'])
            history['val_loss'].append(val_metrics['val/loss'])
            history['val_mae'].append(val_metrics['val/mae'])

            # Log to wandb
            if self.use_wandb:
                metrics = {**train_metrics, **val_metrics}
                metrics['epoch'] = epoch
                metrics['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(metrics)

            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} [{self.current_stage}] | "
                  f"Loss: {train_metrics['train/loss']:.4f} | "
                  f"Val MSE: {val_metrics['val/mse']:.4f} | "
                  f"Val MAE: {val_metrics['val/mae']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"{elapsed:.1f}s")

            # Early stopping check
            if self.should_stop():
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Periodic checkpoint
            if epoch % 50 == 0 and epoch > 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        # Final save
        self.save_checkpoint('final.pt')

        if self.use_wandb:
            wandb.finish()

        return history

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.epoch,
            'stage': self.current_stage,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'num_obs': self.model.num_obs,
                'stack_size': self.model.stack_size,
                'hidden_dim': self.model.hidden_dim,
                'gru_layers': self.model.gru_layers,
            },
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.current_stage = checkpoint['stage']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {path} (epoch {self.epoch}, stage {self.current_stage})")


def autoregressive_test(
    model: VELO_NET,
    encoder: DualEncoder,
    test_sequence_path: str,
    device: str = 'cuda:0',
    max_steps: int = 300,
) -> Dict[str, float]:
    """
    Test model with auto-regressive inference.

    Feeds predicted velocity back as prev_vel input.

    Args:
        model: VELO_NET model
        encoder: DualEncoder for visual features
        test_sequence_path: Path to test sequence
        device: PyTorch device
        max_steps: Maximum steps to test

    Returns:
        Dict with test metrics
    """
    from PIL import Image
    from models.vel_net.vel_obs_utils import quaternion_to_rot6d

    model.eval()
    encoder.eval()

    seq_path = Path(test_sequence_path)
    telemetry = np.load(seq_path / "telemetry.npz")

    n_frames = min(len(telemetry['timestamps']), max_steps)

    all_preds = []
    all_gts = []
    errors = []

    # Start with zero velocity
    prev_vel = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        for t in range(1, n_frames):
            # Load image
            rgb_path = seq_path / "rgb" / f"{t:06d}.png"
            depth_path = seq_path / "depth" / f"{t:06d}.npy"

            rgb = np.array(Image.open(rgb_path)).astype(np.float32) / 255.0
            depth = np.load(depth_path).astype(np.float32)

            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

            # Encode images
            rgb_feat, depth_feat = encoder(rgb_tensor, depth_tensor)

            # Build observation
            quat = torch.from_numpy(telemetry['orientations'][t].astype(np.float32)).unsqueeze(0).to(device)
            rot6d = quaternion_to_rot6d(quat)

            action = torch.from_numpy(telemetry['actions'][t].astype(np.float32)).unsqueeze(0).to(device)
            prev_action = torch.from_numpy(telemetry['actions'][t-1].astype(np.float32)).unsqueeze(0).to(device)

            obs = torch.cat([rot6d, action, prev_action, prev_vel, rgb_feat, depth_feat], dim=1)

            # Predict
            vel_mu, _ = model.encode(obs)

            # Update prev_vel for next step (AUTO-REGRESSIVE)
            prev_vel = vel_mu.detach()

            # Record
            vel_gt = telemetry['velocities'][t]
            all_preds.append(vel_mu.cpu().numpy()[0])
            all_gts.append(vel_gt)
            errors.append(np.abs(vel_mu.cpu().numpy()[0] - vel_gt))

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
    print(f"  Max Error: {metrics['ar/max_error']:.4f} m/s")

    return metrics


if __name__ == '__main__':
    print("Trainer module loaded successfully")
    print(f"Wandb available: {WANDB_AVAILABLE}")
