"""
Stage-specific loss functions for vel_net_v2 multi-stage training.

Stage 1 (Rotation):    OmegaLoss on RotationNet predictions
Stage 2 (Translation): direction + confidence + scale + velocity
Stage 3 (Joint):       OmegaLoss + velocity SmoothL1 + regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class OmegaLoss(nn.Module):
    """
    Robust angular velocity loss with temporal consistency.

    Components:
      - Pointwise: SmoothL1(omega_pred, omega_gt)
      - Temporal difference: SmoothL1(delta_pred, delta_gt)

    The temporal difference term preserves sharp transitions (e.g., sudden
    rotation changes) that pointwise-only losses would smooth out.

    Can be called per-timestep with optional prev_omega arguments for
    temporal difference computation.

    Args:
        delta_weight: Weight for temporal difference loss (default: 0.3)
    """

    def __init__(self, delta_weight: float = 0.3):
        super(OmegaLoss, self).__init__()
        self.delta_weight = delta_weight

    def forward(
        self,
        omega_pred: torch.Tensor,
        omega_gt: torch.Tensor,
        prev_omega_pred: Optional[torch.Tensor] = None,
        prev_omega_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute omega loss (per-timestep).

        Args:
            omega_pred: (B, 3) predicted angular velocity
            omega_gt: (B, 3) ground truth angular velocity
            prev_omega_pred: (B, 3) previous timestep prediction (optional)
            prev_omega_gt: (B, 3) previous timestep GT (optional)

        Returns:
            Dict with total loss and components
        """
        # Pointwise robust loss
        L_point = F.smooth_l1_loss(omega_pred, omega_gt)

        # Temporal difference loss
        L_delta = torch.tensor(0.0, device=omega_pred.device)
        if prev_omega_pred is not None and prev_omega_gt is not None:
            delta_pred = omega_pred - prev_omega_pred
            delta_gt = omega_gt - prev_omega_gt
            L_delta = F.smooth_l1_loss(delta_pred, delta_gt)

        total = L_point + self.delta_weight * L_delta

        return {
            'loss': total,
            'omega_point_loss': L_point,
            'omega_delta_loss': L_delta,
        }


class TranslationLoss(nn.Module):
    """
    Stage 2 loss for translation branch + dynamics training.

    Combines direction/confidence losses (from TranslationBranch) with
    scale/velocity losses (from DynamicsBranch).

    Components:
      - direction_loss: 1 - cosine_similarity(pred_dir, gt_dir)
      - confidence_loss: BCE(pred_conf, gt_conf)
      - scale_loss: MSE(pred_scale, gt_speed)
      - velocity_loss: MSE(dir*scale + correction, gt_vel)
      - temporal_smoothness: penalize scale jumps

    Args:
        direction_weight: Weight for direction loss (default: 1.0)
        confidence_weight: Weight for confidence loss (default: 0.5)
        scale_weight: Weight for scale loss (default: 1.0)
        velocity_weight: Weight for velocity reconstruction loss (default: 2.0)
        smoothness_weight: Weight for temporal smoothness (default: 0.1)
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
        scale_weight: float = 1.0,
        velocity_weight: float = 2.0,
        smoothness_weight: float = 0.1,
    ):
        super(TranslationLoss, self).__init__()
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.scale_weight = scale_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        prev_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute translation + dynamics loss.

        Args:
            pred: Dict with translation_direction (B,3), confidence (B,1),
                  velocity (B,3), translation_scale (B,1)
            gt: Dict with translation_direction_gt (B,3), confidence_gt (B,1),
                velocity_gt (B,3), translation_scale_gt (B,1)
            prev_scale: Previous timestep's predicted scale (B,1)

        Returns:
            Dict with total loss and individual components
        """
        # Direction cosine loss
        cos_sim = F.cosine_similarity(
            pred['translation_direction'], gt['translation_direction_gt'], dim=-1
        )
        direction_loss = (1.0 - cos_sim).mean()

        # Confidence BCE
        with torch.cuda.amp.autocast(enabled=False):
            confidence_loss = F.binary_cross_entropy(
                pred['confidence'].float(), gt['confidence_gt'].float()
            )

        # Scale MSE
        scale_loss = F.mse_loss(pred['translation_scale'], gt['translation_scale_gt'])

        # Velocity reconstruction MSE
        velocity_loss = F.mse_loss(pred['velocity'], gt['velocity_gt'])

        # Temporal smoothness on scale
        smoothness_loss = torch.tensor(0.0, device=pred['velocity'].device)
        if prev_scale is not None:
            smoothness_loss = F.mse_loss(pred['translation_scale'], prev_scale)

        total = (
            self.direction_weight * direction_loss +
            self.confidence_weight * confidence_loss +
            self.scale_weight * scale_loss +
            self.velocity_weight * velocity_loss +
            self.smoothness_weight * smoothness_loss
        )

        return {
            'loss': total,
            'direction_loss': direction_loss,
            'confidence_loss': confidence_loss,
            'scale_loss': scale_loss,
            'velocity_loss': velocity_loss,
            'smoothness_loss': smoothness_loss,
        }


class JointLoss(nn.Module):
    """
    Stage 3 loss for joint fine-tuning.

    Components:
      - velocity_loss: SmoothL1(pred_vel, gt_vel) — primary metric
      - omega_loss: OmegaLoss (SmoothL1 + temporal diff)
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
        self.omega_loss_fn = OmegaLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        prev_omega_pred: Optional[torch.Tensor] = None,
        prev_omega_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint stage loss.

        Args:
            pred: Dict with velocity (B,3), angular_velocity (B,3),
                  translation_direction (B,3), translation_scale (B,1), confidence (B,1)
            gt: Dict with velocity_gt (B,3), angular_velocity_gt (B,3),
                translation_direction_gt (B,3), translation_scale_gt (B,1), confidence_gt (B,1)
            prev_omega_pred: (B, 3) previous timestep omega prediction (optional)
            prev_omega_gt: (B, 3) previous timestep omega GT (optional)

        Returns:
            Dict with total loss and individual components
        """
        # Primary velocity loss (SmoothL1 for robustness)
        velocity_loss = F.smooth_l1_loss(pred['velocity'], gt['velocity_gt'])

        # Angular velocity loss (robust + temporal)
        omega_dict = self.omega_loss_fn(
            pred['angular_velocity'], gt['angular_velocity_gt'],
            prev_omega_pred=prev_omega_pred,
            prev_omega_gt=prev_omega_gt,
        )
        omega_loss = omega_dict['loss']

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


# Keep old names for backward compatibility in imports
GeometryLoss = OmegaLoss
DynamicsLoss = TranslationLoss
