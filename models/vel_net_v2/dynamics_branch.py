"""
Dynamics Branch: Recover metric velocity scale using temporal history + control inputs.

Takes geometry branch outputs + IMU + actions to estimate the scalar speed
and a small velocity correction vector.

Input (per timestep):
  - translation_direction: (B, 3) from geometry branch
  - angular_velocity: (B, 3) from geometry branch
  - confidence: (B, 1) from geometry branch
  - imu_accel: (B, 3) body frame accelerometer
  - imu_gyro: (B, 3) body frame gyroscope
  - prev_vel: (B, 3) previous velocity estimate
  - action: (B, 4) [roll_rate, pitch_rate, yaw_rate, thrust]

Per-step obs: [dir(3) + omega(3) + conf(1) + accel(3) + gyro(3) + prev_vel(3) + action(4)] = 20 dims

Output:
  - translation_scale: (B, 1) positive scalar
  - motion_correction: (B, 3) residual velocity correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DynamicsBranch(nn.Module):
    """
    Recover metric velocity scale using physics cues and control history.

    Architecture:
      - InputNorm: LayerNorm(20)
      - Projector MLP: 20 → 128 → 128
      - GRU: 2 layers, hidden_dim=128
      - Scale head: MLP 128→64→1 + softplus (positive scale)
      - Correction head: MLP 128→64→3 (residual velocity correction)

    Args:
        hidden_dim: GRU hidden dimension (default: 128)
        gru_layers: Number of GRU layers (default: 2)
    """

    OBS_DIM = 20  # dir(3) + omega(3) + conf(1) + accel(3) + gyro(3) + prev_vel(3) + action(4)

    def __init__(
        self,
        hidden_dim: int = 128,
        gru_layers: int = 2,
    ):
        super(DynamicsBranch, self).__init__()

        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        input_dim = 20  # 3+3+1+3+3+3+4

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

        # Scale head: softplus ensures positive output
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        # Initialize scale head bias so softplus outputs ~1.0 m/s initially
        # softplus(x) ≈ x for x >> 0, so bias ≈ 1.0
        with torch.no_grad():
            self.scale_head[-1].bias.fill_(1.0)

        # Correction head: small residual velocity correction
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset GRU hidden state. Call at the start of each sequence."""
        device = next(self.parameters()).device
        self._hidden_state = torch.zeros(
            self.gru_layers, batch_size, self.hidden_dim, device=device
        )

    def forward_step(
        self,
        translation_direction: torch.Tensor,
        angular_velocity: torch.Tensor,
        confidence: torch.Tensor,
        imu_accel: torch.Tensor,
        imu_gyro: torch.Tensor,
        prev_vel: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Process single timestep.

        Args:
            translation_direction: (B, 3) unit vector from geometry branch
            angular_velocity: (B, 3) from geometry branch
            confidence: (B, 1) from geometry branch
            imu_accel: (B, 3) body frame accelerometer
            imu_gyro: (B, 3) body frame gyroscope
            prev_vel: (B, 3) previous velocity estimate
            action: (B, 4) [roll_rate, pitch_rate, yaw_rate, thrust]

        Returns:
            Dict with:
              - translation_scale: (B, 1) positive scalar
              - motion_correction: (B, 3) residual correction
        """
        B = translation_direction.size(0)

        # Initialize hidden state if needed
        if not hasattr(self, '_hidden_state') or self._hidden_state is None:
            self.reset_hidden_state(B)
        if self._hidden_state.size(1) != B:
            self.reset_hidden_state(B)

        # Concatenate inputs
        obs = torch.cat([
            translation_direction,  # 3
            angular_velocity,       # 3
            confidence,             # 1
            imu_accel,              # 3
            imu_gyro,               # 3
            prev_vel,               # 3
            action,                 # 4
        ], dim=1)  # (B, 20)

        # Normalize + project
        obs_norm = self.input_norm(obs)
        projected = self.projector(obs_norm).unsqueeze(1)  # (B, 1, hidden_dim)

        # GRU step
        out, self._hidden_state = self.gru(projected, self._hidden_state)
        h = self._hidden_state[-1]  # (B, hidden_dim)

        # Heads
        scale_raw = self.scale_head(h)  # (B, 1)
        translation_scale = F.softplus(scale_raw)  # positive

        motion_correction = self.correction_head(h)  # (B, 3)

        return {
            'translation_scale': translation_scale,
            'motion_correction': motion_correction,
        }
