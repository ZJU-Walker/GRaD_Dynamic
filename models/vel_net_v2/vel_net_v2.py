"""
VelNetV2: Composite model wrapping Geometry, Dynamics, and Fusion branches.

Multi-branch architecture for velocity estimation:
  - Geometry Branch: angular velocity + translation direction (RGB + flow + IMU)
  - Dynamics Branch: metric scale + correction (geometry outputs + IMU + actions)
  - Fusion Head: direction * scale + correction → final velocity

No Rot6D input — orientation-agnostic for real deployment.
RGB only — no depth features.
Dense optical flow via RAFT-Small (precomputed, encoded by trainable FlowEncoder).
Full IMU (accel + gyro) in body frame.
"""

import contextlib

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path

from models.vel_net_v2.flow_encoder import FlowEncoder
from models.vel_net_v2.geometry_branch import GeometryBranch
from models.vel_net_v2.dynamics_branch import DynamicsBranch
from models.vel_net_v2.fusion_head import FusionHead


class VelNetV2(nn.Module):
    """
    Composite velocity estimation network.

    Manages three branches + flow encoder with support for staged training.

    Args:
        rgb_feat_dim: RGB feature dimension from CompactEncoder (default: 32)
        flow_feat_dim: Optical flow feature dimension (default: 64)
        geo_hidden_dim: Geometry branch GRU hidden dim (default: 128)
        geo_gru_layers: Geometry branch GRU layers (default: 2)
        dyn_hidden_dim: Dynamics branch GRU hidden dim (default: 128)
        dyn_gru_layers: Dynamics branch GRU layers (default: 2)
    """

    def __init__(
        self,
        rgb_feat_dim: int = 32,
        flow_feat_dim: int = 64,
        geo_hidden_dim: int = 128,
        geo_gru_layers: int = 2,
        dyn_hidden_dim: int = 128,
        dyn_gru_layers: int = 2,
    ):
        super(VelNetV2, self).__init__()

        self.rgb_feat_dim = rgb_feat_dim
        self.flow_feat_dim = flow_feat_dim

        # Flow encoder: (B, 2, H, W) → (B, flow_feat_dim)
        self.flow_encoder = FlowEncoder(flow_dim=flow_feat_dim)

        # Geometry branch: angular velocity + translation direction
        self.geometry = GeometryBranch(
            rgb_feat_dim=rgb_feat_dim,
            flow_feat_dim=flow_feat_dim,
            hidden_dim=geo_hidden_dim,
            gru_layers=geo_gru_layers,
        )

        # Dynamics branch: scale + correction
        self.dynamics = DynamicsBranch(
            hidden_dim=dyn_hidden_dim,
            gru_layers=dyn_gru_layers,
        )

        # Fusion head: combine → final velocity
        self.fusion = FusionHead()

        # Training stage tracking
        self.stage = 3  # Default: joint training

        # Store config for checkpoint
        self._config = {
            'rgb_feat_dim': rgb_feat_dim,
            'flow_feat_dim': flow_feat_dim,
            'geo_hidden_dim': geo_hidden_dim,
            'geo_gru_layers': geo_gru_layers,
            'dyn_hidden_dim': dyn_hidden_dim,
            'dyn_gru_layers': dyn_gru_layers,
        }

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset GRU hidden states for both branches."""
        self.geometry.reset_hidden_state(batch_size)
        self.dynamics.reset_hidden_state(batch_size)

    def encode_step(
        self,
        rgb_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        imu_accel: torch.Tensor,
        imu_gyro: torch.Tensor,
        prev_vel: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process single timestep through all branches.

        Args:
            rgb_feat: (B, 32) RGB features from CompactEncoder FC
            flow_feat: (B, 64) flow features from FlowEncoder
            imu_accel: (B, 3) body-frame accelerometer
            imu_gyro: (B, 3) body-frame gyroscope
            prev_vel: (B, 3) previous velocity estimate
            action: (B, 4) [roll_rate, pitch_rate, yaw_rate, thrust]

        Returns:
            Dict with velocity (B,3), angular_velocity (B,3), and intermediate outputs
        """
        # Geometry branch
        # Stage 2: geometry is frozen — no_grad prevents hidden-state graph accumulation
        geo_ctx = torch.no_grad() if self.stage == 2 else contextlib.nullcontext()
        with geo_ctx:
            geo_out = self.geometry.forward_step(rgb_feat, flow_feat, imu_accel, imu_gyro, prev_vel)

        # Detach geometry outputs for dynamics input in staged training
        # Stage 1: dynamics is frozen, no gradient flow needed through it
        # Stage 2: geometry is frozen, prevent backward through it from dynamics
        if self.stage in (1, 2):
            geo_for_dyn = {k: v.detach() for k, v in geo_out.items()}
        else:
            geo_for_dyn = geo_out

        # Dynamics branch
        # Stage 1: dynamics is frozen — no_grad prevents hidden-state graph accumulation
        dyn_ctx = torch.no_grad() if self.stage == 1 else contextlib.nullcontext()
        with dyn_ctx:
            dyn_out = self.dynamics.forward_step(
                translation_direction=geo_for_dyn['translation_direction'],
                angular_velocity=geo_for_dyn['angular_velocity'],
                confidence=geo_for_dyn['confidence'],
                imu_accel=imu_accel,
                imu_gyro=imu_gyro,
                prev_vel=prev_vel,
                action=action,
            )

        # Fusion
        fused = self.fusion(geo_out, dyn_out)

        return fused

    def forward(
        self,
        rgb_feats_seq: torch.Tensor,
        flow_feats_seq: torch.Tensor,
        imu_accel_seq: torch.Tensor,
        imu_gyro_seq: torch.Tensor,
        prev_vel_init: torch.Tensor,
        actions_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process full sequence (batch processing for training).

        Args:
            rgb_feats_seq: (B, T, 32) RGB features per timestep
            flow_feats_seq: (B, T, 64) flow features per timestep
            imu_accel_seq: (B, T, 3) body-frame accelerometer
            imu_gyro_seq: (B, T, 3) body-frame gyroscope
            prev_vel_init: (B, 3) initial velocity estimate
            actions_seq: (B, T, 4) actions per timestep

        Returns:
            Tuple of:
              - vel_sequence: (B, T, 3) predicted velocities
              - omega_sequence: (B, T, 3) predicted angular velocities
              - intermediates: Dict of intermediate outputs for loss computation
        """
        B, T, _ = rgb_feats_seq.shape
        self.reset_hidden_state(B)

        vel_preds = []
        omega_preds = []
        directions = []
        scales = []
        corrections = []
        confidences = []

        prev_vel = prev_vel_init

        for t in range(T):
            out = self.encode_step(
                rgb_feat=rgb_feats_seq[:, t],
                flow_feat=flow_feats_seq[:, t],
                imu_accel=imu_accel_seq[:, t],
                imu_gyro=imu_gyro_seq[:, t],
                prev_vel=prev_vel,
                action=actions_seq[:, t],
            )

            vel_preds.append(out['velocity'])
            omega_preds.append(out['angular_velocity'])
            directions.append(out['translation_direction'])
            scales.append(out['translation_scale'])
            corrections.append(out['motion_correction'])
            confidences.append(out['confidence'])

            # Auto-regressive: use predicted velocity for next step
            prev_vel = out['velocity'].detach()

        vel_sequence = torch.stack(vel_preds, dim=1)      # (B, T, 3)
        omega_sequence = torch.stack(omega_preds, dim=1)   # (B, T, 3)

        intermediates = {
            'translation_direction': torch.stack(directions, dim=1),   # (B, T, 3)
            'translation_scale': torch.stack(scales, dim=1),           # (B, T, 1)
            'motion_correction': torch.stack(corrections, dim=1),      # (B, T, 3)
            'confidence': torch.stack(confidences, dim=1),             # (B, T, 1)
        }

        return vel_sequence, omega_sequence, intermediates

    # =========================================================================
    # Staged training utilities
    # =========================================================================

    def freeze_geometry(self):
        """Freeze geometry branch + flow encoder for Stage 2 training."""
        for param in self.geometry.parameters():
            param.requires_grad = False
        for param in self.flow_encoder.parameters():
            param.requires_grad = False
        self.geometry.eval()
        self.flow_encoder.eval()

    def freeze_dynamics(self):
        """Freeze dynamics branch for Stage 1 training."""
        for param in self.dynamics.parameters():
            param.requires_grad = False
        self.dynamics.eval()

    def freeze_fusion(self):
        """Freeze fusion head (no-op since FusionHead has no params by default)."""
        for param in self.fusion.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all branches for Stage 3 joint training."""
        for param in self.parameters():
            param.requires_grad = True

    def get_geometry_params(self):
        """Get geometry branch + flow encoder trainable parameters."""
        params = list(self.geometry.parameters()) + list(self.flow_encoder.parameters())
        return [p for p in params if p.requires_grad]

    def get_dynamics_params(self):
        """Get dynamics branch trainable parameters."""
        return [p for p in self.dynamics.parameters() if p.requires_grad]

    def get_all_params(self):
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    # =========================================================================
    # Checkpoint utilities
    # =========================================================================

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self._config,
            'stage': self.stage,
        }
        if extra is not None:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cpu') -> Tuple['VelNetV2', dict]:
        """Load model from checkpoint.

        Returns:
            Tuple of (model, full_checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['model_config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.stage = checkpoint.get('stage', 3)
        model.to(device)
        return model, checkpoint

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
