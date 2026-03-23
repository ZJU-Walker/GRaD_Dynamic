"""
Geometry Branch (TranslationBranch): Estimate translation direction from RGB + optical flow + IMU.

Angular velocity prediction has been moved to RotationNet (flow-based).
This branch now receives omega_pred from RotationNet as an input feature
instead of predicting omega itself.

Input (per timestep):
  - rgb_features: (B, 32)      # from CompactEncoder FC (RGB only)
  - flow_features: (B, 64)     # from FlowEncoder (dense optical flow)
  - imu_accel: (B, 3)          # body-frame accelerometer
  - omega_pred: (B, 3)         # predicted omega from RotationNet
  - prev_vel: (B, 3)           # auto-regressive velocity estimate

Per-step obs: [rgb_feat(32) + flow_feat(64) + accel(3) + omega_pred(3) + prev_vel(3)] = 105 dims

Output:
  - translation_direction: (B, 3) unit vector
  - confidence: (B, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class GeometryBranch(nn.Module):
    """
    Estimate translation direction from visual + IMU + predicted omega inputs.

    Architecture:
      - InputNorm: LayerNorm(105)
      - Projector MLP: 105 → 128 → 128
      - GRU: 2 layers, hidden_dim=128
      - Translation direction head: MLP 128→64→3 + L2 normalize
      - Confidence head: MLP 128→32→1 + sigmoid

    Args:
        rgb_feat_dim: RGB feature dimension (default: 32)
        flow_feat_dim: Optical flow feature dimension (default: 64)
        hidden_dim: GRU hidden dimension (default: 128)
        gru_layers: Number of GRU layers (default: 2)
    """

    OBS_DIM = 105  # rgb(32) + flow(64) + accel(3) + omega_pred(3) + prev_vel(3)

    def __init__(
        self,
        rgb_feat_dim: int = 32,
        flow_feat_dim: int = 64,
        hidden_dim: int = 128,
        gru_layers: int = 2,
    ):
        super(GeometryBranch, self).__init__()

        self.rgb_feat_dim = rgb_feat_dim
        self.flow_feat_dim = flow_feat_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        input_dim = rgb_feat_dim + flow_feat_dim + 3 + 3 + 3  # 105

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Projector MLP
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        # Translation direction head (output normalized to unit vector)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset GRU hidden state. Call at the start of each sequence."""
        device = next(self.parameters()).device
        self._hidden_state = torch.zeros(
            self.gru_layers, batch_size, self.hidden_dim, device=device
        )

    def forward_step(
        self,
        rgb_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        imu_accel: torch.Tensor,
        omega_pred: torch.Tensor,
        prev_vel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process single timestep.

        Args:
            rgb_feat: (B, 32) RGB features from CompactEncoder
            flow_feat: (B, 64) flow features from FlowEncoder
            imu_accel: (B, 3) body-frame accelerometer
            omega_pred: (B, 3) predicted angular velocity from RotationNet
            prev_vel: (B, 3) previous velocity estimate

        Returns:
            Dict with:
              - translation_direction: (B, 3) unit vector
              - confidence: (B, 1)
        """
        B = rgb_feat.size(0)

        # Initialize hidden state if needed
        if not hasattr(self, '_hidden_state') or self._hidden_state is None:
            self.reset_hidden_state(B)
        if self._hidden_state.size(1) != B:
            self.reset_hidden_state(B)

        # Concatenate inputs (omega_pred replaces imu_gyro)
        obs = torch.cat([rgb_feat, flow_feat, imu_accel, omega_pred, prev_vel], dim=1)  # (B, 105)

        # Normalize + project
        obs_norm = self.input_norm(obs)
        projected = self.projector(obs_norm).unsqueeze(1)  # (B, 1, hidden_dim)

        # GRU step
        out, self._hidden_state = self.gru(projected, self._hidden_state)
        h = self._hidden_state[-1]  # (B, hidden_dim)

        # Heads
        direction_raw = self.direction_head(h)  # (B, 3)
        direction_norm = F.normalize(direction_raw, p=2, dim=-1)  # unit vector

        confidence = self.confidence_head(h)  # (B, 1)

        return {
            'translation_direction': direction_norm,
            'confidence': confidence,
        }
