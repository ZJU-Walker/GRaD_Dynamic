"""
Velocity Observation History Buffer.

Manages a rolling window of velocity network observations for temporal processing.
"""

import torch
from typing import Optional


class VelObsHistBuffer:
    """
    Rolling buffer for velocity observation history.

    Maintains a fixed-size buffer of recent observations for feeding into
    the velocity network's GRU encoder.

    Args:
        batch_size: Number of parallel environments/samples
        obs_dim: Dimension of each observation (default: 49)
        buffer_size: Number of timesteps to store (default: 5)
        device: Device for tensor storage ('cpu' or 'cuda:X')

    Example:
        >>> buffer = VelObsHistBuffer(batch_size=32, obs_dim=49, buffer_size=5)
        >>> for t in range(100):
        ...     new_obs = get_observation()  # (32, 49)
        ...     buffer.update(new_obs)
        ...     history = buffer.get_concatenated()  # (32, 245)
        ...     # or
        ...     history_3d = buffer.get_buffer_3d()  # (32, 5, 49)
    """

    def __init__(self,
                 batch_size: int,
                 obs_dim: int = 49,
                 buffer_size: int = 5,
                 device: str = 'cpu'):
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.device = device

        # Initialize buffer with zeros: (batch_size, buffer_size, obs_dim)
        # Buffer order: oldest to newest (index 0 = oldest, index -1 = newest)
        self.buffer = torch.zeros(
            (batch_size, buffer_size, obs_dim),
            device=self.device
        )

    def update(self, new_obs: torch.Tensor) -> None:
        """
        Add new observation to buffer, discarding oldest.

        Args:
            new_obs: New observation tensor (batch_size, obs_dim)
        """
        assert new_obs.shape == (self.batch_size, self.obs_dim), \
            f"Expected shape ({self.batch_size}, {self.obs_dim}), got {new_obs.shape}"

        # Detach to prevent gradient accumulation through buffer
        self.buffer = self.buffer.detach()

        # Shift buffer: discard oldest (index 0), make room for newest
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)

        # Clone to avoid in-place operation issues
        self.buffer = self.buffer.clone()

        # Insert new observation at the end (newest position)
        self.buffer[:, -1, :] = new_obs.clone().detach()

    def get_concatenated(self) -> torch.Tensor:
        """
        Get flattened history for networks expecting 2D input.

        Returns:
            Flattened buffer (batch_size, buffer_size * obs_dim)
        """
        return self.buffer.detach().view(self.batch_size, -1)

    def get_buffer_3d(self) -> torch.Tensor:
        """
        Get 3D buffer for networks expecting sequence input (e.g., GRU).

        Returns:
            Buffer tensor (batch_size, buffer_size, obs_dim)
        """
        return self.buffer.detach().clone()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset buffer to zeros.

        Args:
            env_ids: Optional tensor of environment indices to reset.
                     If None, resets all environments.
        """
        if env_ids is None:
            self.buffer.zero_()
        else:
            self.buffer[env_ids] = 0.0

    def to(self, device: str) -> 'VelObsHistBuffer':
        """
        Move buffer to specified device.

        Args:
            device: Target device ('cpu' or 'cuda:X')

        Returns:
            Self for chaining
        """
        self.buffer = self.buffer.to(device)
        self.device = device
        return self

    def clone(self) -> 'VelObsHistBuffer':
        """
        Create a deep copy of the buffer.

        Returns:
            New VelObsHistBuffer with copied data
        """
        new_buffer = VelObsHistBuffer(
            batch_size=self.batch_size,
            obs_dim=self.obs_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )
        new_buffer.buffer = self.buffer.clone()
        return new_buffer

    def resize_batch(self, new_batch_size: int) -> None:
        """
        Resize buffer for different batch size (resets data).

        Args:
            new_batch_size: New batch size
        """
        self.batch_size = new_batch_size
        self.buffer = torch.zeros(
            (new_batch_size, self.buffer_size, self.obs_dim),
            device=self.device
        )

    def __repr__(self) -> str:
        return (f"VelObsHistBuffer(batch_size={self.batch_size}, "
                f"obs_dim={self.obs_dim}, buffer_size={self.buffer_size}, "
                f"device={self.device})")


if __name__ == '__main__':
    # Example usage
    batch_size = 4
    obs_dim = 49
    buffer_size = 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    buffer = VelObsHistBuffer(batch_size, obs_dim, buffer_size, device=device)

    # Simulate 10 steps
    for step in range(10):
        new_obs = torch.randn((batch_size, obs_dim), device=device)
        buffer.update(new_obs)

        if step % 3 == 0:
            concatenated = buffer.get_concatenated()
            buffer_3d = buffer.get_buffer_3d()
            print(f"Step {step}: Concat shape={concatenated.shape}, 3D shape={buffer_3d.shape}")

    # Test reset
    buffer.reset()
    print(f"After reset, buffer sum: {buffer.buffer.sum().item()}")

    # Test partial reset
    buffer.update(torch.ones((batch_size, obs_dim), device=device))
    buffer.reset(env_ids=torch.tensor([0, 2]))
    print(f"Env 0 sum: {buffer.buffer[0].sum().item()}")  # Should be 0
    print(f"Env 1 sum: {buffer.buffer[1].sum().item()}")  # Should be > 0
