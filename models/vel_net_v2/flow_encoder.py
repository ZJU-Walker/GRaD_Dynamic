"""
Flow Encoder: CNN encoder for dense optical flow fields.

Takes RAFT-Small output (B, 2, H, W) and produces a compact feature vector (B, flow_dim).
Trainable end-to-end with the geometry branch.
RAFT-Small itself is frozen (pretrained); only this encoder is trainable.
"""

import torch
import torch.nn as nn


class FlowEncoder(nn.Module):
    """
    Encode dense optical flow field into compact feature vector.

    Architecture:
      - Conv2d(2, 32, 3, stride=2, padding=1) + ReLU
      - Conv2d(32, 64, 3, stride=2, padding=1) + ReLU
      - AdaptiveAvgPool2d(1) → (B, 64, 1, 1)
      - Flatten → Linear(64, flow_dim)

    Args:
        flow_dim: Output feature dimension (default: 64)

    Input: (B, 2, H, W) dense optical flow from RAFT-Small
    Output: (B, flow_dim) flow feature vector
    """

    def __init__(self, flow_dim: int = 64):
        super(FlowEncoder, self).__init__()

        self.flow_dim = flow_dim

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, flow_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Encode optical flow field.

        Args:
            flow: Dense optical flow (B, 2, H, W)

        Returns:
            Flow features (B, flow_dim)
        """
        x = self.conv(flow)         # (B, 64, H/4, W/4)
        x = self.pool(x)            # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc(x)              # (B, flow_dim)
        return x

    def get_trainable_params(self):
        """Return all parameters (all are trainable)."""
        return self.parameters()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
