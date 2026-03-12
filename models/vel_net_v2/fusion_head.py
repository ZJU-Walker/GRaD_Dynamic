"""
Fusion Head: Combine geometry + dynamics outputs into final velocity estimate.

Combines:
  - translation_direction (unit vector) * translation_scale + motion_correction → linear velocity
  - angular_velocity from geometry branch (pass-through)

Output:
  - velocity: (B, 3) final [v_x, v_y, v_z] in body frame
  - angular_velocity: (B, 3) final [ω_x, ω_y, ω_z] in body frame
"""

import torch
import torch.nn as nn
from typing import Dict


class FusionHead(nn.Module):
    """
    Fuse geometry and dynamics branch outputs into final velocity.

    The fusion is straightforward:
      linear_vel = direction * scale + correction
      angular_vel = geometry angular_velocity (pass-through)

    No learnable parameters in the default mode.
    """

    def __init__(self):
        super(FusionHead, self).__init__()

    def forward(
        self,
        geometry_outputs: Dict[str, torch.Tensor],
        dynamics_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse geometry and dynamics outputs.

        Args:
            geometry_outputs: Dict with:
              - translation_direction: (B, 3) unit vector
              - angular_velocity: (B, 3)
              - confidence: (B, 1)
            dynamics_outputs: Dict with:
              - translation_scale: (B, 1) positive scalar
              - motion_correction: (B, 3)

        Returns:
            Dict with:
              - velocity: (B, 3) final body-frame velocity
              - angular_velocity: (B, 3) final body-frame angular velocity
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

        # Angular velocity is pass-through from geometry
        angular_velocity = geometry_outputs['angular_velocity']  # (B, 3)

        return {
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            # Pass-through for logging/loss computation
            'translation_direction': direction,
            'translation_scale': scale,
            'motion_correction': correction,
            'confidence': geometry_outputs['confidence'],
        }
