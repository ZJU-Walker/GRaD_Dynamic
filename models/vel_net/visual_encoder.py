"""
Visual Encoder for RGB and Depth images.

Uses MobileNetV3-Small backbone with trainable projection head.
Same training approach as original GRaD_Nav_internal_Local project:
- Backbone: FROZEN (pretrained ImageNet weights)
- FC layer: TRAINABLE (learns jointly with vel_net)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class CompactEncoder(nn.Module):
    """
    MobileNetV3-Small based encoder for visual features.

    Training approach:
    - Backbone (MobileNetV3): FROZEN (pretrained ImageNet weights)
    - FC layer (576 → output_dim): TRAINABLE (learns jointly with vel_net)

    Args:
        input_channels: Number of input channels (3 for RGB, 1 for depth)
        output_dim: Output feature dimension (default: 32)

    Input: (B, C, H, W) - images, any resolution (adaptive pooling)
    Output: (B, output_dim) feature vector
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 32):
        super(CompactEncoder, self).__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Load MobileNetV3-Small with pretrained weights
        backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

        # Modify first conv layer for different input channels (e.g., depth)
        if input_channels != 3:
            # Original first conv: Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            original_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize with mean of original weights for single channel
            if input_channels == 1:
                with torch.no_grad():
                    backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Extract feature layers (remove classifier)
        self.features = backbone.features

        # Adaptive pooling to handle any input size
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection head: 576 (MobileNetV3-Small output) → output_dim
        # This is the TRAINABLE part
        self.fc = nn.Linear(576, output_dim)

        # Freeze backbone (pretrained weights)
        for param in self.features.parameters():
            param.requires_grad = False

        # fc layer remains trainable (default requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input image (B, C, H, W)
               - For RGB: (B, 3, H, W), values in [0, 1] or [0, 255]
               - For Depth: (B, 1, H, W), metric depth values

        Returns:
            Feature vector (B, output_dim)
        """
        # Normalize input if needed
        if x.max() > 1.0:
            x = x / 255.0

        # Pass through frozen backbone
        with torch.no_grad():
            features = self.features(x)
            features = self.pool(features)

        # Flatten
        features = features.view(features.size(0), -1)  # (B, 576)

        # Pass through trainable projection head
        output = self.fc(features)  # (B, output_dim)

        return output

    def get_trainable_params(self):
        """Return only trainable parameters (FC layer)."""
        return self.fc.parameters()

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class DualEncoder(nn.Module):
    """
    Combined RGB and Depth encoder.

    Contains two CompactEncoders:
    - RGB encoder: 3 channels → 32 dims
    - Depth encoder: 1 channel → 32 dims

    Total output: 64 dims (32 RGB + 32 Depth)
    """

    def __init__(self, rgb_dim: int = 32, depth_dim: int = 32):
        super(DualEncoder, self).__init__()

        self.rgb_encoder = CompactEncoder(input_channels=3, output_dim=rgb_dim)
        self.depth_encoder = CompactEncoder(input_channels=1, output_dim=depth_dim)

        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode RGB and depth images.

        Args:
            rgb: RGB image (B, 3, H, W) or (B, H, W, 3)
            depth: Depth image (B, 1, H, W) or (B, H, W) or (B, H, W, 1)

        Returns:
            Tuple of (rgb_features, depth_features), each (B, output_dim)
        """
        # Handle channel-last format for RGB
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)  # (B, H, W, 3) → (B, 3, H, W)

        # Handle different depth formats
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)  # (B, H, W, 1) → (B, 1, H, W)

        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)

        return rgb_features, depth_features

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode only RGB image."""
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        return self.rgb_encoder(rgb)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode only depth image."""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        return self.depth_encoder(depth)

    def get_trainable_params(self):
        """Return trainable parameters from both encoders."""
        return list(self.rgb_encoder.get_trainable_params()) + list(self.depth_encoder.get_trainable_params())

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return self.rgb_encoder.num_trainable_params() + self.depth_encoder.num_trainable_params()


def preprocess_image(image: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess image for encoder.

    Args:
        image: Input image tensor
        target_size: Target size (H, W) for resizing

    Returns:
        Preprocessed image tensor
    """
    import torch.nn.functional as F

    # Ensure float
    if image.dtype != torch.float32:
        image = image.float()

    # Normalize to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Resize if needed
    if image.shape[-2:] != target_size:
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

    return image


if __name__ == '__main__':
    # Test encoder
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Testing CompactEncoder on {device}")

    # Test single encoder
    print("\n--- Single CompactEncoder ---")
    rgb_encoder = CompactEncoder(input_channels=3, output_dim=32).to(device)
    depth_encoder = CompactEncoder(input_channels=1, output_dim=32).to(device)

    print(f"RGB encoder - Total params: {rgb_encoder.num_total_params():,}")
    print(f"RGB encoder - Trainable params: {rgb_encoder.num_trainable_params():,}")
    print(f"Depth encoder - Total params: {depth_encoder.num_total_params():,}")
    print(f"Depth encoder - Trainable params: {depth_encoder.num_trainable_params():,}")

    # Test forward pass with different input sizes
    batch_size = 4
    for H, W in [(64, 64), (224, 224), (360, 640)]:
        rgb = torch.randn(batch_size, 3, H, W, device=device)
        depth = torch.randn(batch_size, 1, H, W, device=device)

        rgb_feat = rgb_encoder(rgb)
        depth_feat = depth_encoder(depth)

        print(f"Input ({H}x{W}): RGB {rgb.shape} → {rgb_feat.shape}, Depth {depth.shape} → {depth_feat.shape}")

    # Test DualEncoder
    print("\n--- DualEncoder ---")
    dual_encoder = DualEncoder(rgb_dim=32, depth_dim=32).to(device)
    print(f"DualEncoder - Trainable params: {dual_encoder.num_trainable_params():,}")

    # Test with channel-last format
    rgb_hwc = torch.randn(batch_size, 224, 224, 3, device=device)
    depth_hw = torch.randn(batch_size, 224, 224, device=device)

    rgb_feat, depth_feat = dual_encoder(rgb_hwc, depth_hw)
    print(f"Channel-last input: RGB {rgb_hwc.shape} → {rgb_feat.shape}, Depth {depth_hw.shape} → {depth_feat.shape}")

    # Test gradient flow
    print("\n--- Gradient Test ---")
    rgb = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
    rgb_feat = rgb_encoder(rgb)
    loss = rgb_feat.sum()
    loss.backward()

    # Check gradients
    fc_grad = rgb_encoder.fc.weight.grad
    print(f"FC layer gradient exists: {fc_grad is not None}")
    if fc_grad is not None:
        print(f"FC layer gradient norm: {fc_grad.norm().item():.6f}")

    # Check backbone is frozen
    backbone_grad = list(rgb_encoder.features.parameters())[0].grad
    print(f"Backbone gradient exists: {backbone_grad is not None}")

    print("\nVisual Encoder test passed!")
