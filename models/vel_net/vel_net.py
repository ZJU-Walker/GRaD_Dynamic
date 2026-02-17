"""
Velocity Network (VELO_NET) for drone velocity estimation with Direct Delta-V Prediction.

Auto-regressive GRU-based network that directly predicts velocity changes.
Uses previous velocity prediction as input during inference.

DIRECT DELTA-V MODE:
The network directly predicts velocity change from visual and IMU features:
    v_t = v_{t-1} + Network(obs)

Where:
- obs: Includes normalized IMU velocity as an input feature
- Network(obs): Directly predicts delta_v (velocity change)

IMU velocity is included in the observation as a feature to help the network
learn velocity estimation with noisy IMU readings.

Architecture:
    Input (76 dims) → LayerNorm → Projector MLP → GRU → Head MLP → Delta-V (3D)
    Final: vel_pred = prev_vel + delta_v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable, Tuple


def get_activation(act_name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
    }
    if act_name not in activations:
        print(f"Warning: Unknown activation '{act_name}', using ELU")
        return nn.ELU()
    return activations[act_name]


class VELO_NET(nn.Module):
    """
    Velocity estimation network with direct delta-v prediction.

    Uses GRU to process observation history and outputs velocity change (delta_v).
    Implements direct prediction: vel = prev_vel + delta_v

    Args:
        num_obs: Observation dimension per timestep (default: 76 with IMU, no action)
        stack_size: Number of stacked timesteps in history (default: 1 for testing)
        num_latent: Latent dimension for head MLP (default: 64)
        activation: Activation function name (default: 'elu')
        hidden_dim: GRU and projector hidden dimension (default: 256)
        gru_layers: Number of GRU layers (default: 3)
        dt: Time step (default: 1/30 = 33ms), used for acceleration computation
        device: Device to place the model on (default: 'cpu')
    """

    # Observation structure indices (76 dims total with IMU velocity, no action)
    # [0:6]   - Rot6D (rotation)
    # [6:9]   - Previous velocity (auto-regressive term)
    # [9:41]  - RGB features (32 dims)
    # [41:73] - Depth features (32 dims)
    # [73:76] - IMU velocity (3 dims)
    PREV_VEL_START_IDX = 6
    PREV_VEL_END_IDX = 9
    IMU_VEL_START_IDX = 73
    IMU_VEL_END_IDX = 76

    def __init__(self,
                 num_obs: int = 76,
                 stack_size: int = 1,
                 num_latent: int = 64,
                 activation: str = 'elu',
                 hidden_dim: int = 256,
                 gru_layers: int = 3,
                 dt: float = 1.0 / 30.0,
                 device: str = 'cpu'):
        super(VELO_NET, self).__init__()

        self.num_obs = num_obs
        self.stack_size = stack_size
        self.num_latent = num_latent
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.dt = dt
        self.device = device

        act_fn = get_activation(activation)

        # Input normalization (handles mixed-scale features: rotations ~1, RPM ~1000)
        self.input_norm = nn.LayerNorm(num_obs)

        # Feature projector: maps normalized obs to hidden_dim before GRU
        self.projector = nn.Sequential(
            nn.Linear(num_obs, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GRU encoder processes sequence: (B, stack_size, hidden_dim) → (B, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # MLP head outputs velocity distribution parameters
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            act_fn,
            nn.Linear(256, num_latent * 2),
            act_fn,
        )

        # Velocity output heads (mean and log-variance)
        self.vel_mu = nn.Linear(num_latent * 2, 3)
        self.vel_var = nn.Linear(num_latent * 2, 3)

    def reset_hidden_state(self, batch_size: int = 1):
        """
        Reset GRU hidden state for step-by-step processing.

        Call this at the start of each new sequence.

        Args:
            batch_size: Batch size for hidden state
        """
        device = next(self.parameters()).device
        self._hidden_state = torch.zeros(
            self.gru_layers, batch_size, self.hidden_dim, device=device
        )

    def encode_step(self, obs: torch.Tensor, prev_vel_raw: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode single timestep with direct delta-v prediction.

        DIRECT DELTA-V: vel_pred = prev_vel + delta_v

        Use this for step-by-step training with scheduled sampling.
        Call reset_hidden_state() at the start of each sequence.

        Args:
            obs: Single observation (B, num_obs) with normalized accel in last 3 dims
            prev_vel_raw: Unused, kept for API compatibility.

        Returns:
            Tuple of (delta_v_mu, delta_v_logvar), each shape (B, 3)
            Note: delta_v_mu is the VELOCITY CHANGE, not absolute velocity!
                  Caller must add: vel_pred = prev_vel + delta_v_mu
        """
        B = obs.size(0)

        # Initialize hidden state if not exists
        if not hasattr(self, '_hidden_state') or self._hidden_state is None:
            self.reset_hidden_state(B)

        # Ensure hidden state batch size matches
        if self._hidden_state.size(1) != B:
            self.reset_hidden_state(B)

        # Normalize observation (network sees normalized version)
        obs_norm = self.input_norm(obs)  # (B, num_obs)

        # Project
        projected = self.projector(obs_norm).unsqueeze(1)  # (B, 1, hidden_dim)

        # GRU step with persistent hidden state
        out, self._hidden_state = self.gru(projected, self._hidden_state)

        # Head MLP outputs DELTA_V (velocity change)
        x = self.head(self._hidden_state[-1])  # Use last layer's hidden
        delta_v_mu = self.vel_mu(x)
        delta_v_logvar = self.vel_var(x)

        return delta_v_mu, delta_v_logvar

    def physics_integrate(self, prev_vel: torch.Tensor, accel: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        [DEPRECATED] Physics-based velocity integration (dead reckoning).

        Note: This method is no longer used in direct delta-v mode.
        Kept for backward compatibility with old checkpoints.

        Args:
            prev_vel: Previous velocity (B, 3) in m/s
            accel: Acceleration/IMU reading (B, 3) in m/s^2
            dt: Time step (uses self.dt if None)

        Returns:
            vel_physics: Physics-predicted velocity (B, 3)
        """
        if dt is None:
            dt = self.dt
        return prev_vel + accel * dt

    def fuse_imu_vision(self, prev_vel: torch.Tensor, accel: torch.Tensor,
                        correction: torch.Tensor, dt: float = None) -> torch.Tensor:
        """
        [DEPRECATED] Fuse IMU prediction with visual correction.

        Note: This method is no longer used in direct delta-v mode.
        Kept for backward compatibility with old checkpoints.
        Use direct delta-v instead: vel_pred = prev_vel + delta_v

        Args:
            prev_vel: Previous velocity (B, 3) in m/s
            accel: IMU acceleration (B, 3) in m/s^2
            correction: Visual correction delta (B, 3) from network
            dt: Time step (uses self.dt if None)

        Returns:
            vel_pred: Fused velocity prediction (B, 3)
        """
        vel_physics = self.physics_integrate(prev_vel, accel, dt)
        return vel_physics + correction

    def encode(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation history to velocity distribution parameters.

        Args:
            obs_history: Observation history tensor
                - Shape (B, T, num_obs) for sequence input
                - Shape (B, T * num_obs) for flattened input

        Returns:
            Tuple of (vel_mu, vel_logvar), each shape (B, 3)
        """
        # Handle flattened input: (B, T × num_obs) → (B, T, num_obs)
        if obs_history.ndim == 2:
            B = obs_history.size(0)
            obs_history = obs_history.view(B, self.stack_size, self.num_obs)

        B, T, D = obs_history.shape

        # Normalize each timestep's observation
        obs_normalized = self.input_norm(obs_history)  # (B, T, num_obs)

        # Project each timestep to hidden_dim
        # Reshape for projection: (B*T, num_obs) → (B*T, hidden_dim) → (B, T, hidden_dim)
        obs_flat = obs_normalized.view(B * T, D)
        projected = self.projector(obs_flat).view(B, T, self.hidden_dim)

        # GRU forward pass
        gru_out, h = self.gru(projected)  # h: (num_layers, B, hidden_dim)
        last_hidden = h[-1]  # (B, hidden_dim)

        # Head MLP
        x = self.head(last_hidden)
        vel_mu = self.vel_mu(x)
        vel_logvar = self.vel_var(x)

        return vel_mu, vel_logvar

    def forward(self, obs_history: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with reparameterization for training.

        Args:
            obs_history: Observation history (B, T, num_obs) or (B, T*num_obs)

        Returns:
            Tuple of:
                - [vel]: List containing sampled velocity (B, 3)
                - [vel_mu, vel_logvar]: Distribution parameters
        """
        vel_mu, vel_logvar = self.encode(obs_history)
        vel = self.reparameterize(vel_mu, vel_logvar)
        return [vel], [vel_mu, vel_logvar]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.

        Args:
            mu: Mean of distribution (B, 3)
            logvar: Log-variance of distribution (B, 3)

        Returns:
            Sampled velocity (B, 3)
        """
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(self,
                obs_history: torch.Tensor,
                vel_gt: torch.Tensor,
                thrust: Optional[torch.Tensor] = None,
                pinn_weight: float = 0.1,
                **kwargs) -> torch.Tensor:
        """
        Compute training loss (supervised + optional PINN).

        Args:
            obs_history: Observation history (B, T, num_obs) or (B, T*num_obs)
            vel_gt: Ground truth velocity (B, 3)
            thrust: Optional thrust history for PINN loss (B, T, 3)
            pinn_weight: Weight for physics loss (default: 0.1)

        Returns:
            Loss tensor (B,)
        """
        estimation, latent_params = self.forward(obs_history)
        v = estimation[0]
        vel_mu, vel_logvar = latent_params

        # Supervised loss (MSE for better magnitude sensitivity)
        # MSE penalizes large errors quadratically, improving magnitude accuracy
        v = torch.clamp(v, min=-1e2, max=1e2)
        vel_gt = torch.clamp(vel_gt, min=-1e2, max=1e2)
        vel_loss = F.mse_loss(v, vel_gt, reduction='none').mean(-1)
        loss = torch.clamp(vel_loss, max=1e4)

        # PINN loss (physics-informed regularization)
        # FIXED: Gradients flow through physics loss (no torch.no_grad())
        pinn_loss = torch.zeros_like(loss)
        if thrust is not None and obs_history.ndim == 3:
            vel_pred_seq = self.predict_sequence(obs_history)  # (B, T, 3)
            pinn_loss = self.physics_loss_fn(vel_pred_seq, thrust)

        total_loss = loss + pinn_weight * pinn_loss
        return total_loss

    def predict_sequence(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity sequence for PINN loss computation.

        Processes each timestep sequentially through GRU to get
        velocity predictions at each step.

        Args:
            obs_history: Observation history (B, T, num_obs)

        Returns:
            Velocity predictions (B, T, 3)
        """
        B, T, D = obs_history.shape
        hidden = None
        preds = []

        for t in range(T):
            # Get single timestep and normalize
            obs_t = obs_history[:, t, :]  # (B, num_obs)
            obs_t_norm = self.input_norm(obs_t)

            # Project to hidden_dim
            projected_t = self.projector(obs_t_norm).unsqueeze(1)  # (B, 1, hidden_dim)

            # GRU step
            out, hidden = self.gru(projected_t, hidden)

            # Predict velocity from current hidden state
            x = self.head(hidden[-1])
            vel = self.vel_mu(x)  # Deterministic (no sampling)
            preds.append(vel)

        return torch.stack(preds, dim=1)  # (B, T, 3)

    def physics_loss_fn(self,
                        vel_pred_seq: torch.Tensor,
                        thrust: torch.Tensor,
                        mass: float = 1.2,
                        gravity: float = -9.81,
                        drag_coeff: float = 0.1,
                        dt: float = 0.05) -> torch.Tensor:
        """
        Compute physics residual loss (Newton's second law).

        Enforces: mass * acceleration = thrust + gravity + drag

        Args:
            vel_pred_seq: Predicted velocity sequence (B, T, 3)
            thrust: Thrust/force history (B, T, 3)
            mass: Drone mass in kg (default: 1.2)
            gravity: Gravitational acceleration (default: -9.81)
            drag_coeff: Drag coefficient (default: 0.1)
            dt: Time step in seconds (default: 0.05)

        Returns:
            Physics loss (B,)
        """
        B, T, _ = vel_pred_seq.shape

        # Compute estimated acceleration using finite differences
        acc_est = (vel_pred_seq[:, 1:, :] - vel_pred_seq[:, :-1, :]) / dt  # (B, T-1, 3)

        # Construct expected total force
        F_gravity = torch.tensor([0, 0, mass * gravity], device=vel_pred_seq.device)
        vel_mid = vel_pred_seq[:, :-1, :]  # Velocities for drag computation
        F_drag = -drag_coeff * vel_mid
        F_total = thrust[:, :-1, :] + F_gravity + F_drag

        # Physics residual: mass * a - F = 0
        physics_residual = mass * acc_est - F_total  # (B, T-1, 3)
        loss = torch.mean(physics_residual ** 2, dim=-1).mean(dim=-1)  # (B,)

        return loss

    @torch.no_grad()
    def inference(self,
                  get_sensor_obs_fn: Callable[[int], torch.Tensor],
                  num_steps: int,
                  init_vel: Optional[torch.Tensor] = None,
                  batch_size: int = 1) -> List[torch.Tensor]:
        """
        Auto-regressive inference for real-time use.

        Uses predicted velocity from previous step as input to current step.

        Args:
            get_sensor_obs_fn: Function(t) -> sensor_obs (B, num_obs-3)
                               Returns observation WITHOUT prev_vel slot
            num_steps: Number of steps to predict
            init_vel: Initial velocity estimate (B, 3), default zeros
            batch_size: Batch size if init_vel not provided

        Returns:
            List of velocity predictions, each (B, 3)
        """
        device = next(self.parameters()).device

        if init_vel is None:
            prev_vel = torch.zeros(batch_size, 3, device=device)
        else:
            prev_vel = init_vel.to(device)

        predictions = []
        hidden = None

        for t in range(num_steps):
            # Get current sensor observation (without prev_vel)
            sensor_obs = get_sensor_obs_fn(t)  # (B, num_obs-3) or full obs

            # Insert prev_vel into observation at the correct position
            obs = self._insert_prev_vel(sensor_obs, prev_vel)  # (B, num_obs)

            # Normalize and project
            obs_norm = self.input_norm(obs)
            projected = self.projector(obs_norm).unsqueeze(1)  # (B, 1, hidden_dim)

            # GRU step (maintains hidden state across steps)
            out, hidden = self.gru(projected, hidden)

            # Predict velocity
            x = self.head(hidden[-1])
            current_vel = self.vel_mu(x)
            predictions.append(current_vel)

            # Update for next step
            prev_vel = current_vel.detach()

        return predictions

    def _insert_prev_vel(self,
                         sensor_obs: torch.Tensor,
                         prev_vel: torch.Tensor) -> torch.Tensor:
        """
        Insert previous velocity into observation at correct position.

        Args:
            sensor_obs: Sensor observation (B, num_obs) with placeholder for prev_vel
                        OR (B, num_obs-3) without prev_vel
            prev_vel: Previous velocity (B, 3)

        Returns:
            Full observation (B, num_obs)
        """
        B = sensor_obs.shape[0]

        if sensor_obs.shape[1] == self.num_obs:
            # Full observation with placeholder - replace prev_vel slots
            obs = sensor_obs.clone()
            obs[:, self.PREV_VEL_START_IDX:self.PREV_VEL_END_IDX] = prev_vel
            return obs
        elif sensor_obs.shape[1] == self.num_obs - 3:
            # Observation without prev_vel - insert it
            obs = torch.cat([
                sensor_obs[:, :self.PREV_VEL_START_IDX],  # Before prev_vel
                prev_vel,                                  # prev_vel
                sensor_obs[:, self.PREV_VEL_START_IDX:]   # After prev_vel position
            ], dim=1)
            return obs
        else:
            raise ValueError(
                f"Expected sensor_obs dim {self.num_obs} or {self.num_obs-3}, "
                f"got {sensor_obs.shape[1]}"
            )

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        iteration: int = 0, extra: Optional[dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            iteration: Current iteration number
            extra: Extra data to include in checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_obs': self.num_obs,
                'stack_size': self.stack_size,
                'num_latent': self.num_latent,
                'hidden_dim': self.hidden_dim,
                'gru_layers': self.gru_layers,
                'dt': self.dt,
            },
            'iteration': iteration,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if extra is not None:
            checkpoint.update(extra)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu') -> Tuple['VELO_NET', dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model to

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = cls(
            num_obs=config['num_obs'],
            stack_size=config['stack_size'],
            num_latent=config['num_latent'],
            hidden_dim=config['hidden_dim'],
            gru_layers=config['gru_layers'],
            dt=config.get('dt', 1.0 / 30.0),  # Default for backward compatibility
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint
