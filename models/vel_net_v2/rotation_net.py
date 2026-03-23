"""
RotationNet: Predict body-frame angular velocity from optical flow spatial structure.

Optical flow naturally encodes rotation:
  - Roll  → circular flow pattern
  - Pitch → vertical gradient
  - Yaw   → horizontal gradient

Unlike the old GeometryBranch omega_head (which fed gyro GT through a GRU and
regressed-to-mean), this module predicts omega directly from the flow's spatial
structure using depthwise-separable convolutions + attention pooling.

Input:  (B, 2, H, W) raw optical flow  (coord channels added internally)
Output: (B, 3) body-frame angular velocity [omega_x, omega_y, omega_z]
"""

import torch
import torch.nn as nn
from typing import Optional


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class RotationNet(nn.Module):
    """
    Predict body-frame angular velocity from optical flow spatial structure.

    Architecture (~14k params):
      SpatialFlowEncoder:
        DepthwiseSeparableConv(5, 32, k=3, s=2) + ELU     # H/2 x W/2
        DepthwiseSeparableConv(32, 64, k=3, s=2) + ELU    # H/4 x W/4
        DepthwiseSeparableConv(64, 64, k=3, s=2) + ELU    # H/8 x W/8

      AttentionPooling:
        Conv2d(64, 1, k=1) -> softmax -> weighted sum -> (B, 64)

      OmegaMLP (instantaneous estimate):
        Linear(64, 32) + ELU
        Linear(32, 3) -> omega_inst (B, 3)

      TemporalRefiner (small residual GRU):
        GRU(input_size=3, hidden_size=32, num_layers=1)
        Linear(32, 3) -> delta_omega (B, 3)

      Output: omega = omega_inst + delta_omega (residual connection)

    Input:  (B, 2, H, W)  raw optical flow from RAFT-Small
    Output: (B, 3)         body-frame angular velocity
    """

    def __init__(self):
        super(RotationNet, self).__init__()

        # Spatial encoder: flow(2) + coord_channels(x, y, r²) = 5 input channels
        self.spatial_encoder = nn.Sequential(
            DepthwiseSeparableConv(5, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            DepthwiseSeparableConv(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )

        # Attention pooling
        self.attention_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Omega MLP (instantaneous estimate)
        self.omega_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 3),
        )

        # Temporal refiner (small residual GRU)
        self.gru = nn.GRU(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.gru_fc = nn.Linear(32, 3)

        # Initialize GRU FC near zero for clean residual at init
        with torch.no_grad():
            self.gru_fc.weight.fill_(0.0)
            self.gru_fc.bias.fill_(0.0)

        # Coord channel cache (populated on first forward)
        self._coord_cache: dict = {}

    def _get_coord_channels(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate coordinate channels (x, y, r²), cached per resolution."""
        key = (H, W, str(device))
        if key not in self._coord_cache:
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            r2 = xx ** 2 + yy ** 2
            coords = torch.stack([xx, yy, r2], dim=0)  # (3, H, W)
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def reset_hidden_state(self, batch_size: int = 1):
        """Reset GRU hidden state. Call at the start of each sequence."""
        device = next(self.parameters()).device
        self._hidden_state = torch.zeros(1, batch_size, 32, device=device)

    def forward(self, flow_raw: torch.Tensor) -> torch.Tensor:
        """
        Predict angular velocity from raw optical flow.

        Args:
            flow_raw: (B, 2, H, W) raw optical flow from RAFT-Small

        Returns:
            omega: (B, 3) predicted body-frame angular velocity
        """
        B, _, H, W = flow_raw.shape

        # Initialize hidden state if needed
        if not hasattr(self, '_hidden_state') or self._hidden_state is None:
            self.reset_hidden_state(B)
        if self._hidden_state.size(1) != B:
            self.reset_hidden_state(B)

        # Add coordinate channels: (B, 2, H, W) → (B, 5, H, W)
        coords = self._get_coord_channels(H, W, flow_raw.device)
        coords_batch = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 3, H, W)
        x = torch.cat([flow_raw, coords_batch], dim=1)  # (B, 5, H, W)

        # Spatial encoder
        spatial_feat = self.spatial_encoder(x)  # (B, 64, H/8, W/8)

        # Attention pooling
        attn_logits = self.attention_conv(spatial_feat)  # (B, 1, H/8, W/8)
        B_s, _, Hs, Ws = attn_logits.shape
        attn_weights = torch.softmax(attn_logits.view(B_s, 1, -1), dim=-1)  # (B, 1, Hs*Ws)
        spatial_flat = spatial_feat.view(B_s, 64, -1)  # (B, 64, Hs*Ws)
        pooled = torch.bmm(spatial_flat, attn_weights.transpose(1, 2)).squeeze(-1)  # (B, 64)

        # Instantaneous omega estimate
        omega_inst = self.omega_mlp(pooled)  # (B, 3)

        # Temporal refinement (residual GRU)
        gru_in = omega_inst.unsqueeze(1)  # (B, 1, 3)
        gru_out, self._hidden_state = self.gru(gru_in, self._hidden_state)
        delta_omega = self.gru_fc(gru_out.squeeze(1))  # (B, 3)

        omega = omega_inst + delta_omega  # Residual connection

        return omega

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
