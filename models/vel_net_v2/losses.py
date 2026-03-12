"""
Stage-specific loss functions for vel_net_v2 multi-stage training.

Stage 1 (Geometry): angular velocity + translation direction + confidence
Stage 2 (Dynamics): scale + velocity reconstruction + temporal smoothness
Stage 3 (Joint): full velocity + angular velocity + regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GeometryLoss(nn.Module):
    """
    Stage 1 loss for geometry branch training.

    Components:
      - angular_velocity_loss: MSE(pred_omega, gt_omega)
      - direction_loss: 1 - cosine_similarity(pred_dir, gt_dir)
      - confidence_loss: BCE(pred_conf, gt_conf)

    Args:
        omega_weight: Weight for angular velocity loss (default: 1.0)
        direction_weight: Weight for direction loss (default: 1.0)
        confidence_weight: Weight for confidence loss (default: 0.5)
    """

    def __init__(
        self,
        omega_weight: float = 1.0,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
    ):
        super(GeometryLoss, self).__init__()
        self.omega_weight = omega_weight
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometry stage loss.

        Args:
            pred: Dict with angular_velocity (B,3), translation_direction (B,3), confidence (B,1)
            gt: Dict with angular_velocity_gt (B,3), translation_direction_gt (B,3), confidence_gt (B,1)

        Returns:
            Dict with total loss and individual components
        """
        # Angular velocity MSE
        omega_loss = F.mse_loss(pred['angular_velocity'], gt['angular_velocity_gt'])

        # Direction cosine loss: 1 - cos_sim
        cos_sim = F.cosine_similarity(pred['translation_direction'], gt['translation_direction_gt'], dim=-1)
        direction_loss = (1.0 - cos_sim).mean()

        # Confidence BCE (disable autocast — BCE is registered as unsafe)
        with torch.cuda.amp.autocast(enabled=False):
            confidence_loss = F.binary_cross_entropy(
                pred['confidence'].float(), gt['confidence_gt'].float()
            )

        total = (
            self.omega_weight * omega_loss +
            self.direction_weight * direction_loss +
            self.confidence_weight * confidence_loss
        )

        return {
            'loss': total,
            'omega_loss': omega_loss,
            'direction_loss': direction_loss,
            'confidence_loss': confidence_loss,
        }


class DynamicsLoss(nn.Module):
    """
    Stage 2 loss for dynamics branch training.

    Components:
      - scale_loss: MSE(pred_scale, gt_speed)
      - velocity_loss: MSE(dir*scale + correction, gt_vel)
      - temporal_smoothness: penalize scale jumps

    Args:
        scale_weight: Weight for scale loss (default: 1.0)
        velocity_weight: Weight for velocity reconstruction loss (default: 2.0)
        smoothness_weight: Weight for temporal smoothness (default: 0.1)
    """

    def __init__(
        self,
        scale_weight: float = 1.0,
        velocity_weight: float = 2.0,
        smoothness_weight: float = 0.1,
    ):
        super(DynamicsLoss, self).__init__()
        self.scale_weight = scale_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        prev_scale: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dynamics stage loss.

        Args:
            pred: Dict with velocity (B,3), translation_scale (B,1), motion_correction (B,3)
            gt: Dict with velocity_gt (B,3), translation_scale_gt (B,1)
            prev_scale: Previous timestep's predicted scale (B,1) for smoothness

        Returns:
            Dict with total loss and individual components
        """
        # Scale MSE
        scale_loss = F.mse_loss(pred['translation_scale'], gt['translation_scale_gt'])

        # Velocity reconstruction MSE
        velocity_loss = F.mse_loss(pred['velocity'], gt['velocity_gt'])

        # Temporal smoothness on scale
        smoothness_loss = torch.tensor(0.0, device=pred['velocity'].device)
        if prev_scale is not None:
            smoothness_loss = F.mse_loss(pred['translation_scale'], prev_scale)

        total = (
            self.scale_weight * scale_loss +
            self.velocity_weight * velocity_loss +
            self.smoothness_weight * smoothness_loss
        )

        return {
            'loss': total,
            'scale_loss': scale_loss,
            'velocity_loss': velocity_loss,
            'smoothness_loss': smoothness_loss,
        }


class JointLoss(nn.Module):
    """
    Stage 3 loss for joint fine-tuning.

    Components:
      - velocity_loss: SmoothL1(pred_vel, gt_vel) — primary metric
      - angular_velocity_loss: MSE(pred_omega, gt_omega)
      - geometry_reg: direction cosine + confidence (keep geometry reasonable)
      - scale_reg: keep scale reasonable

    Args:
        velocity_weight: Weight for velocity loss (default: 2.0)
        omega_weight: Weight for angular velocity loss (default: 0.5)
        geometry_reg_weight: Weight for geometry regularization (default: 0.1)
        scale_reg_weight: Weight for scale regularization (default: 0.1)
    """

    def __init__(
        self,
        velocity_weight: float = 2.0,
        omega_weight: float = 0.5,
        geometry_reg_weight: float = 0.1,
        scale_reg_weight: float = 0.1,
    ):
        super(JointLoss, self).__init__()
        self.velocity_weight = velocity_weight
        self.omega_weight = omega_weight
        self.geometry_reg_weight = geometry_reg_weight
        self.scale_reg_weight = scale_reg_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint stage loss.

        Args:
            pred: Dict with velocity (B,3), angular_velocity (B,3),
                  translation_direction (B,3), translation_scale (B,1), confidence (B,1)
            gt: Dict with velocity_gt (B,3), angular_velocity_gt (B,3),
                translation_direction_gt (B,3), translation_scale_gt (B,1), confidence_gt (B,1)

        Returns:
            Dict with total loss and individual components
        """
        # Primary velocity loss (SmoothL1 for robustness)
        velocity_loss = F.smooth_l1_loss(pred['velocity'], gt['velocity_gt'])

        # Angular velocity loss
        omega_loss = F.mse_loss(pred['angular_velocity'], gt['angular_velocity_gt'])

        # Geometry regularization: direction should still be reasonable
        cos_sim = F.cosine_similarity(pred['translation_direction'], gt['translation_direction_gt'], dim=-1)
        direction_reg = (1.0 - cos_sim).mean()
        with torch.cuda.amp.autocast(enabled=False):
            confidence_reg = F.binary_cross_entropy(
                pred['confidence'].float(), gt['confidence_gt'].float()
            )
        geometry_reg = direction_reg + confidence_reg

        # Scale regularization
        scale_reg = F.mse_loss(pred['translation_scale'], gt['translation_scale_gt'])

        total = (
            self.velocity_weight * velocity_loss +
            self.omega_weight * omega_loss +
            self.geometry_reg_weight * geometry_reg +
            self.scale_reg_weight * scale_reg
        )

        return {
            'loss': total,
            'velocity_loss': velocity_loss,
            'omega_loss': omega_loss,
            'geometry_reg': geometry_reg,
            'scale_reg': scale_reg,
        }
