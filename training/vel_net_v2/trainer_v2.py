"""
Multi-stage trainer for VelNetV2.

Stage 1 (Rotation): Train RotationNet only
  - Freeze TranslationBranch + Dynamics + FlowEncoder + Encoder
  - Optimize RotationNet
  - Loss: OmegaLoss

Stage 2 (Translation): Train TranslationBranch + Dynamics + FlowEncoder
  - Load from Stage 1, freeze RotationNet
  - Optimize TranslationBranch + Dynamics + FlowEncoder + encoder
  - Loss: TranslationLoss (direction + confidence + scale + velocity)

Stage 3 (Joint): Fine-tune all branches
  - Load from Stage 2, unfreeze all
  - Lower LR for pretrained branches
  - Loss: JointLoss (OmegaLoss + SmoothL1(vel) + regularization)

Follows same patterns as VelNetTrainer:
  - Precomputed backbone features (576-dim RGB only)
  - Precomputed optical flow (from RAFT-Small)
  - AMP + gradient accumulation
  - Scheduled sampling on prev_vel
  - W&B logging
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

from models.vel_net_v2 import VelNetV2, OmegaLoss, TranslationLoss, JointLoss
from models.vel_net.visual_encoder import CompactEncoder

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class VelNetV2Trainer:
    """
    Multi-stage trainer for VelNetV2.

    Args:
        model: VelNetV2 model
        encoder: CompactEncoder (RGB only, not DualEncoder)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        stage: Training stage (1, 2, or 3)
        lr: Learning rate
        lr_geometry: LR for geometry branch (Stage 3 only)
        lr_dynamics: LR for dynamics branch (Stage 3 only)
        weight_decay: Weight decay
        grad_clip: Gradient clipping norm
        tf_start_epoch: Epoch to start scheduled sampling decay
        tf_end_epoch: Epoch to finish scheduled sampling decay
        grad_accumulation_steps: Gradient accumulation
        device: PyTorch device
        checkpoint_dir: Checkpoint directory
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
    """

    def __init__(
        self,
        model: VelNetV2,
        encoder: CompactEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stage: int = 1,
        lr: float = 3e-4,
        lr_geometry: float = 3e-5,
        lr_dynamics: float = 3e-5,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        tf_start_epoch: int = 0,
        tf_end_epoch: int = 100,
        grad_accumulation_steps: int = 1,
        device: str = 'cuda:0',
        checkpoint_dir: str = 'checkpoints/vel_net_v2',
        use_wandb: bool = False,
        wandb_project: str = 'vel_net_v2',
    ):
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.stage = stage
        self.device = device
        self.lr = lr
        self.grad_clip = grad_clip
        self.tf_start_epoch = tf_start_epoch
        self.tf_end_epoch = tf_end_epoch
        self.grad_accumulation_steps = grad_accumulation_steps

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Configure model for stage
        model.stage = stage
        self._configure_stage(stage, lr, lr_geometry, lr_dynamics, weight_decay)

        # Training state
        self.epoch = 0
        self.best_val_metric = float('inf')
        self.epochs_without_improvement = 0

        # Compute normalization stats
        stats = self._compute_stats()
        self.vel_mean, self.vel_std = stats['vel_mean'], stats['vel_std']
        self.accel_mean, self.accel_std = stats['accel_mean'], stats['accel_std']
        self.gyro_mean, self.gyro_std = stats['gyro_mean'], stats['gyro_std']
        print(f"  Velocity norm: mean={self.vel_mean.cpu().numpy()}, std={self.vel_std.cpu().numpy()}")
        print(f"  Accel norm:    mean={self.accel_mean.cpu().numpy()}, std={self.accel_std.cpu().numpy()}")
        print(f"  Gyro norm:     mean={self.gyro_mean.cpu().numpy()}, std={self.gyro_std.cpu().numpy()}")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6,
        )

        # Loss function
        if stage == 1:
            self.loss_fn = OmegaLoss()
        elif stage == 2:
            self.loss_fn = TranslationLoss()
        elif stage == 3:
            self.loss_fn = JointLoss()

        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'stage': stage,
                    'lr': lr,
                    'model_params': model.num_total_params(),
                    'trainable_params': model.num_trainable_params(),
                },
            )

    def _configure_stage(self, stage, lr, lr_geometry, lr_dynamics, weight_decay):
        """Configure model freezing and optimizer for the given stage."""
        if stage == 1:
            # Stage A (Rotation): Train RotationNet only
            self.model.freeze_geometry()  # TranslationBranch + FlowEncoder
            self.model.freeze_dynamics()
            self.model.freeze_fusion()
            # Freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Optimize: RotationNet only
            params = list(self.model.get_rotation_params())
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        elif stage == 2:
            # Stage B (Translation): Train TranslationBranch + Dynamics + FlowEncoder + encoder
            self.model.freeze_rotation()
            # Unfreeze everything else
            for param in self.model.geometry.parameters():
                param.requires_grad = True
            for param in self.model.flow_encoder.parameters():
                param.requires_grad = True
            for param in self.model.dynamics.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = True
            # Optimize
            params = (
                list(self.model.get_geometry_params()) +
                list(self.model.get_dynamics_params()) +
                list(self.encoder.get_trainable_params())
            )
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        elif stage == 3:
            # Stage C (Joint): Train all with differential LRs
            self.model.unfreeze_all()
            param_groups = [
                {'params': self.model.get_rotation_params(), 'lr': lr_geometry},
                {'params': self.model.get_geometry_params(), 'lr': lr_geometry},
                {'params': self.model.get_dynamics_params(), 'lr': lr_dynamics},
                {'params': list(self.encoder.get_trainable_params()), 'lr': lr},
            ]
            self.optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        print(f"  Stage {stage} configured: {self.model.num_trainable_params():,} trainable params")

    def _compute_stats(self) -> Dict[str, torch.Tensor]:
        """Compute normalization statistics from training data."""
        all_vels, all_accels, all_gyros = [], [], []

        for batch in self.train_loader:
            for b_idx in range(len(batch['velocities_gt'])):
                all_vels.append(batch['velocities_gt'][b_idx].numpy())
                all_accels.append(batch['accel_gt'][b_idx].numpy())
                all_gyros.append(batch['gyro_gt'][b_idx].numpy())

        all_vels = np.concatenate(all_vels, axis=0)
        all_accels = np.concatenate(all_accels, axis=0)
        all_gyros = np.concatenate(all_gyros, axis=0)

        def to_stat(arr):
            mean = torch.from_numpy(arr.mean(axis=0).astype(np.float32)).to(self.device)
            std = torch.clamp(
                torch.from_numpy(arr.std(axis=0).astype(np.float32)).to(self.device),
                min=1e-6,
            )
            return mean, std

        vel_mean, vel_std = to_stat(all_vels)
        accel_mean, accel_std = to_stat(all_accels)
        gyro_mean, gyro_std = to_stat(all_gyros)

        return {
            'vel_mean': vel_mean, 'vel_std': vel_std,
            'accel_mean': accel_mean, 'accel_std': accel_std,
            'gyro_mean': gyro_mean, 'gyro_std': gyro_std,
        }

    def get_teacher_forcing_ratio(self) -> float:
        """Get current teacher forcing ratio."""
        if self.epoch < self.tf_start_epoch:
            return 1.0
        if self.epoch >= self.tf_end_epoch:
            return 0.0
        progress = (self.epoch - self.tf_start_epoch) / (self.tf_end_epoch - self.tf_start_epoch)
        return 1.0 - progress

    def _load_backbone_features(self, seq_path: str) -> torch.Tensor:
        """Load precomputed RGB backbone features for a sequence (cached)."""
        if not hasattr(self, '_feature_cache'):
            self._feature_cache = {}
        if seq_path not in self._feature_cache:
            feature_path = Path(seq_path) / "backbone_features.npz"
            features = np.load(feature_path)
            rgb_tensor = torch.from_numpy(features['rgb_features']).to(self.device)
            self._feature_cache[seq_path] = rgb_tensor
        return self._feature_cache[seq_path]

    def _load_optical_flows(self, seq_path: str) -> torch.Tensor:
        """Load precomputed optical flows for a sequence (cached)."""
        if not hasattr(self, '_flow_cache'):
            self._flow_cache = {}
        if seq_path not in self._flow_cache:
            flow_path = Path(seq_path) / "optical_flow.npz"
            flows = np.load(flow_path)
            flow_tensor = torch.from_numpy(flows['flows']).to(self.device)
            self._flow_cache[seq_path] = flow_tensor
        return self._flow_cache[seq_path]

    def _check_precomputed_available(self) -> Tuple[bool, bool]:
        """Check if precomputed features and flows exist."""
        has_features = True
        has_flows = True
        for batch in self.train_loader:
            for seq_path in batch['seq_paths']:
                if not (Path(seq_path) / "backbone_features.npz").exists():
                    has_features = False
                if not (Path(seq_path) / "optical_flow.npz").exists():
                    has_flows = False
            break
        return has_features, has_flows

    def train_epoch(self, pbar: tqdm = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.encoder.train()

        # But keep frozen parts in eval mode
        if self.stage == 1:
            self.model.geometry.eval()
            self.model.dynamics.eval()
            self.model.flow_encoder.eval()
            self.encoder.eval()
        elif self.stage == 2:
            self.model.rotation_net.eval()

        tf_ratio = self.get_teacher_forcing_ratio()
        scaler = torch.cuda.amp.GradScaler()

        total_loss = 0.0
        total_vel_error = 0.0
        n_sequences = 0
        total_tf_frames = 0
        total_pred_frames = 0

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
                    actions = batch['actions'][b_idx].to(self.device)
                    velocities_gt = batch['velocities_gt'][b_idx].to(self.device)
                    prev_vel_raw = batch['initial_prev_vels'][b_idx].to(self.device)
                    accel_aug = batch['accel_aug'][b_idx].to(self.device)
                    angular_velocity_gt = batch['angular_velocity_gt'][b_idx].to(self.device)
                    direction_gt = batch['translation_direction_gt'][b_idx].to(self.device)
                    scale_gt = batch['translation_scale_gt'][b_idx].to(self.device)
                    confidence_gt = batch['confidence_gt'][b_idx].to(self.device)

                    # Load precomputed features
                    rgb_backbone = self._load_backbone_features(seq_path)
                    optical_flows = self._load_optical_flows(seq_path)

                    seq_length = len(frame_indices)
                    self.model.reset_hidden_state(batch_size=1)

                    # Batch RGB FC: all frames at once (576→32)
                    frame_idx_tensor = torch.tensor(frame_indices, device=self.device)
                    rgb_bb_all = rgb_backbone[frame_idx_tensor]  # (L, 576)
                    rgb_feat_all = self.encoder.forward_from_backbone_features(rgb_bb_all)  # (L, 32)

                    # Batch flow encoding: all frames at once
                    flow_indices = [max(0, fi - 1) for fi in frame_indices]
                    flow_idx_tensor = torch.tensor(flow_indices, device=self.device)
                    flows_all = optical_flows[flow_idx_tensor]  # (L, 2, H, W) — raw flows
                    flow_feat_all = self.model.flow_encoder(flows_all)  # (L, 64) — pooled

                    prev_scale = None  # For translation loss smoothness
                    prev_omega_pred = None  # For omega temporal diff loss
                    prev_omega_gt = None

                    for t in range(seq_length):
                        # Get features for this timestep
                        rgb_feat = rgb_feat_all[t:t+1]    # (1, 32)
                        flow_feat = flow_feat_all[t:t+1]  # (1, 64)
                        flow_raw = flows_all[t:t+1]        # (1, 2, H, W)
                        imu_accel = accel_aug[t:t+1]       # (1, 3)
                        action = actions[t:t+1]             # (1, 4)

                        # Forward through model
                        out = self.model.encode_step(
                            rgb_feat=rgb_feat,
                            flow_feat=flow_feat,
                            flow_raw=flow_raw,
                            imu_accel=imu_accel,
                            prev_vel=prev_vel_raw.unsqueeze(0),
                            action=action,
                        )

                        # Compute loss based on stage
                        if self.stage == 1:
                            # Stage A: OmegaLoss only
                            loss_dict = self.loss_fn(
                                out['angular_velocity'],
                                angular_velocity_gt[t:t+1],
                                prev_omega_pred=prev_omega_pred,
                                prev_omega_gt=prev_omega_gt,
                            )
                            prev_omega_pred = out['angular_velocity'].detach()
                            prev_omega_gt = angular_velocity_gt[t:t+1].detach()

                        elif self.stage == 2:
                            # Stage B: TranslationLoss
                            gt = {
                                'translation_direction_gt': direction_gt[t:t+1],
                                'translation_scale_gt': scale_gt[t:t+1],
                                'confidence_gt': confidence_gt[t:t+1],
                                'velocity_gt': velocities_gt[t:t+1],
                            }
                            loss_dict = self.loss_fn(out, gt, prev_scale=prev_scale)
                            prev_scale = out['translation_scale'].detach()

                        else:  # stage 3
                            # Stage C: JointLoss
                            gt = {
                                'angular_velocity_gt': angular_velocity_gt[t:t+1],
                                'translation_direction_gt': direction_gt[t:t+1],
                                'translation_scale_gt': scale_gt[t:t+1],
                                'confidence_gt': confidence_gt[t:t+1],
                                'velocity_gt': velocities_gt[t:t+1],
                            }
                            loss_dict = self.loss_fn(
                                out, gt,
                                prev_omega_pred=prev_omega_pred,
                                prev_omega_gt=prev_omega_gt,
                            )
                            prev_omega_pred = out['angular_velocity'].detach()
                            prev_omega_gt = angular_velocity_gt[t:t+1].detach()

                        batch_loss += loss_dict['loss']
                        batch_frames += 1

                        # Track velocity error
                        vel_error = F.l1_loss(out['velocity'], velocities_gt[t:t+1]).item()
                        batch_vel_error += vel_error

                        # Scheduled sampling
                        if np.random.random() < tf_ratio:
                            prev_vel_raw = velocities_gt[t].detach()
                            total_tf_frames += 1
                        else:
                            prev_vel_raw = out['velocity'].squeeze(0).detach()
                            total_pred_frames += 1

                    n_sequences += 1

            if batch_frames > 0:
                avg_loss = batch_loss / batch_frames
                scaled_loss = avg_loss / self.grad_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += avg_loss.item()
                total_vel_error += batch_vel_error / batch_frames
                batch_idx += 1

                if batch_idx % self.grad_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    # Clip only trainable parameters
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    trainable_params += list(self.encoder.get_trainable_params())
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    total_loss += accumulated_loss / self.grad_accumulation_steps
                    accumulated_loss = 0.0

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{total_loss / max(1, batch_idx // self.grad_accumulation_steps):.4f}',
                    'tf': f'{tf_ratio:.2f}',
                })

        # Handle remaining accumulated gradients
        if batch_idx % self.grad_accumulation_steps != 0:
            scaler.unscale_(self.optimizer)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            trainable_params += list(self.encoder.get_trainable_params())
            torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
            remaining = batch_idx % self.grad_accumulation_steps
            total_loss += accumulated_loss / remaining

        n_batches = len(self.train_loader)
        total_frames = total_tf_frames + total_pred_frames
        actual_tf_ratio = total_tf_frames / max(1, total_frames)

        return {
            'train/loss': total_loss / max(1, n_batches),
            'train/vel_mae': total_vel_error / max(1, n_batches),
            'train/tf_ratio_target': tf_ratio,
            'train/tf_ratio_actual': actual_tf_ratio,
        }

    @torch.no_grad()
    def validate_autoregressive(self) -> Dict[str, float]:
        """Validate with auto-regressive inference (0% teacher forcing)."""
        self.model.eval()
        self.encoder.eval()

        all_vel_errors = []
        all_omega_errors = []
        all_dir_cos = []
        total_frames = 0

        for batch in self.val_loader:
            for b_idx in range(len(batch['seq_paths'])):
                seq_path = batch['seq_paths'][b_idx]
                frame_indices = batch['frame_indices'][b_idx]
                actions = batch['actions'][b_idx].to(self.device)
                velocities_gt = batch['velocities_gt'][b_idx].to(self.device)
                prev_vel_raw = batch['initial_prev_vels'][b_idx].to(self.device)
                accel = batch['accel_aug'][b_idx].to(self.device)
                angular_velocity_gt = batch['angular_velocity_gt'][b_idx].to(self.device)
                direction_gt = batch['translation_direction_gt'][b_idx].to(self.device)

                rgb_backbone = self._load_backbone_features(seq_path)
                optical_flows = self._load_optical_flows(seq_path)

                seq_length = len(frame_indices)
                self.model.reset_hidden_state(batch_size=1)

                frame_idx_tensor = torch.tensor(frame_indices, device=self.device)
                rgb_bb_all = rgb_backbone[frame_idx_tensor]
                rgb_feat_all = self.encoder.forward_from_backbone_features(rgb_bb_all)

                flow_indices = [max(0, fi - 1) for fi in frame_indices]
                flow_idx_tensor = torch.tensor(flow_indices, device=self.device)
                flows_all = optical_flows[flow_idx_tensor]  # (L, 2, H, W) raw
                flow_feat_all = self.model.flow_encoder(flows_all)  # (L, 64) pooled

                for t in range(seq_length):
                    out = self.model.encode_step(
                        rgb_feat=rgb_feat_all[t:t+1],
                        flow_feat=flow_feat_all[t:t+1],
                        flow_raw=flows_all[t:t+1],
                        imu_accel=accel[t:t+1],
                        prev_vel=prev_vel_raw.unsqueeze(0),
                        action=actions[t:t+1],
                    )

                    vel_err = torch.abs(out['velocity'] - velocities_gt[t:t+1]).cpu().numpy()[0]
                    all_vel_errors.append(vel_err)

                    omega_err = torch.abs(out['angular_velocity'] - angular_velocity_gt[t:t+1]).cpu().numpy()[0]
                    all_omega_errors.append(omega_err)

                    cos_sim = F.cosine_similarity(
                        out['translation_direction'], direction_gt[t:t+1], dim=-1
                    ).cpu().item()
                    all_dir_cos.append(cos_sim)

                    total_frames += 1

                    # Auto-regressive: use predicted velocity
                    prev_vel_raw = out['velocity'].squeeze(0)

        all_vel_errors = np.array(all_vel_errors)
        all_omega_errors = np.array(all_omega_errors)

        vel_mae = np.mean(all_vel_errors)
        vel_mae_xyz = np.mean(all_vel_errors, axis=0)

        return {
            'val/vel_mae': vel_mae,
            'val/vel_mae_x': vel_mae_xyz[0],
            'val/vel_mae_y': vel_mae_xyz[1],
            'val/vel_mae_z': vel_mae_xyz[2],
            'val/vel_rmse': np.sqrt(np.mean(all_vel_errors**2)),
            'val/omega_mae': np.mean(all_omega_errors),
            'val/direction_cos': np.mean(all_dir_cos),
        }

    def train(self, n_epochs: int = 200, early_stop_patience: int = 30) -> Dict[str, List[float]]:
        """Main training loop."""
        history = {'train_loss': [], 'val_vel_mae': [], 'val_loss': [], 'tf_ratio': []}

        # Check precomputed data
        has_features, has_flows = self._check_precomputed_available()
        if not has_features:
            raise RuntimeError(
                "Precomputed backbone features not found!\n"
                "Run: python training/vel_net_body/precompute_features.py --data_dir <data_dir>"
            )
        if not has_flows:
            raise RuntimeError(
                "Precomputed optical flow not found!\n"
                "Run: python training/vel_net_v2/precompute_optical_flow.py --data_dir <data_dir>"
            )

        stage_name = {1: 'Rotation', 2: 'Translation', 3: 'Joint'}[self.stage]
        print(f"\n{'='*60}")
        print(f"VelNetV2 Training — Stage {self.stage} ({stage_name})")
        print(f"{'='*60}")
        print(f"  Total params: {self.model.num_total_params():,}")
        print(f"  Trainable params: {self.model.num_trainable_params():,}")
        print(f"  Encoder trainable: {self.encoder.num_trainable_params():,}")
        print(f"  TF schedule: 100% (epoch 0-{self.tf_start_epoch}) → 0% (epoch {self.tf_end_epoch}+)")
        print(f"{'='*60}\n")

        n_batches = len(self.train_loader)

        for epoch in range(n_epochs):
            self.epoch = epoch
            tf_ratio = self.get_teacher_forcing_ratio()

            batch_pbar = tqdm(
                total=n_batches,
                desc=f'Epoch {epoch:3d} [S{self.stage} TF={tf_ratio:.2f}]',
                leave=False,
            )
            train_metrics = self.train_epoch(pbar=batch_pbar)
            batch_pbar.close()

            val_metrics = self.validate_autoregressive()
            self.scheduler.step(val_metrics['val/vel_mae'])

            # Track best model
            metric_key = 'val/vel_mae'
            if val_metrics[metric_key] < self.best_val_metric:
                self.best_val_metric = val_metrics[metric_key]
                self.epochs_without_improvement = 0
                self.save_checkpoint(f'stage{self.stage}_best.pt')
            else:
                self.epochs_without_improvement += 1

            history['train_loss'].append(train_metrics['train/loss'])
            history['val_vel_mae'].append(val_metrics['val/vel_mae'])
            history['val_loss'].append(val_metrics['val/vel_mae'])
            history['tf_ratio'].append(tf_ratio)

            if self.use_wandb:
                metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
                metrics['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(metrics)

            print(f"Epoch {epoch:3d} | S{self.stage} TF={tf_ratio:.2f} | "
                  f"Loss={train_metrics['train/loss']:.4f} | "
                  f"VelMAE={val_metrics['val/vel_mae']:.4f} | "
                  f"OmegaMAE={val_metrics['val/omega_mae']:.4f} | "
                  f"DirCos={val_metrics['val/direction_cos']:.4f} | "
                  f"LR={self.optimizer.param_groups[0]['lr']:.1e}")

            if self.epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            if epoch % 50 == 0 and epoch > 0:
                self.save_checkpoint(f'stage{self.stage}_epoch_{epoch}.pt')

        self.save_checkpoint(f'stage{self.stage}_final.pt')

        if self.use_wandb:
            wandb.finish()

        print(f"\nStage {self.stage} complete! Best VelMAE: {self.best_val_metric:.4f}")
        return history

    def save_checkpoint(self, filename: str):
        """Save checkpoint with all state."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.epoch,
            'stage': self.stage,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model._config,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'tf_start_epoch': self.tf_start_epoch,
            'tf_end_epoch': self.tf_end_epoch,
            # Normalization stats
            'vel_mean': self.vel_mean.cpu(),
            'vel_std': self.vel_std.cpu(),
            'accel_mean': self.accel_mean.cpu(),
            'accel_std': self.accel_std.cpu(),
            'gyro_mean': self.gyro_mean.cpu(),
            'gyro_std': self.gyro_std.cpu(),
        }
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except (ValueError, KeyError):
                print("  Warning: Could not load optimizer state (stage change?), using fresh optimizer")
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        print(f"Loaded checkpoint from {path} (epoch {self.epoch}, stage {checkpoint.get('stage', '?')})")
