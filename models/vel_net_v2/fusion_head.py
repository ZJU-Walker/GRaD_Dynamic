"""
Fusion Head: Combine geometry + dynamics outputs into final velocity estimate.

Combines:
  - translation_direction (unit vector) * translation_scale + motion_correction → linear velocity
  - angular_velocity from RotationNet (pass-through)

Output:
  - velocity: (B, 3) final [v_x, v_y, v_z] in body frame
  - angular_velocity: (B, 3) final [omega_x, omega_y, omega_z] in body frame
"""

import torch
import torch.nn as nn
from typing import Dict


class FusionHead(nn.Module):
    """
    Fuse geometry and dynamics branch outputs into final velocity.

    The fusion is straightforward:
      linear_vel = direction * scale + correction
      angular_vel = omega from RotationNet (pass-through)

    No learnable parameters in the default mode.
    """

    def __init__(self):
        super(FusionHead, self).__init__()

    def forward(
        self,
        geometry_outputs: Dict[str, torch.Tensor],
        dynamics_outputs: Dict[str, torch.Tensor],
        omega: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse geometry and dynamics outputs.

        Args:
            geometry_outputs: Dict with:
              - translation_direction: (B, 3) unit vector
              - confidence: (B, 1)
            dynamics_outputs: Dict with:
              - translation_scale: (B, 1) positive scalar
              - motion_correction: (B, 3)
            omega: (B, 3) angular velocity from RotationNet

        Returns:
            Dict with:
              - velocity: (B, 3) final body-frame velocity
              - angular_velocity: (B, 3) from RotationNet (pass-through)
              - translation_direction: (B, 3) from geometry (for logging)
              - translation_scale: (B, 1) from dynamics (for logging)
              - motion_correction: (B, 3) from dynamics (for logging)
              - confidence: (B, 1) from geometry (for logging)
        """
        direction = geometry_outputs['translation_direction']   # (B, 3)
        scale = dynamics_outputs['translation_scale']           # (B, 1)
        correction = dynamics_outputs['motion_correction']      # (B, 3)

        # Linear velocity = direction * scale + correction
        velocity = direction * scale + correction  # (B, 3)

        # Angular velocity from RotationNet (pass-through)
        angular_velocity = omega  # (B, 3)

        return {
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            # Pass-through for logging/loss computation
            'translation_direction': direction,
            'translation_scale': scale,
            'motion_correction': correction,
            'confidence': geometry_outputs['confidence'],
        }
