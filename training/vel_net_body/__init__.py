"""
Velocity Network Training Pipeline (Body Frame).

Contains:
- DataCollector: Collect flight data from geometric controller
- VelNetDataset: PyTorch Dataset for training (transforms to body frame)
- VelNetTrainer: Training loop with curriculum learning (body frame)

Body Frame Convention (FLU - Forward-Left-Up):
- v_x: velocity along drone's forward axis
- v_y: velocity along drone's left axis
- v_z: velocity along drone's up axis

Usage:
    # Collect training data (same as world frame - transformation happens at training)
    python training/vel_net_body/train_vel_net_body.py collect --map gate_mid --n_sequences 30

    # Train model
    python training/vel_net_body/train_vel_net_body.py train --data_dir data/vel_net/sequences
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'DataCollector',
    'collect_sequences',
    'VelNetDataset',
    'VelNetDatasetWithEncoder',
    'VelNetTrainer',
    'autoregressive_test',
]


def __getattr__(name):
    """Lazy import to avoid loading heavy modules until needed."""
    if name in ('DataCollector', 'collect_sequences'):
        from .data_collector import DataCollector, collect_sequences
        return DataCollector if name == 'DataCollector' else collect_sequences
    elif name in ('VelNetDataset', 'VelNetDatasetWithEncoder'):
        from .dataset import VelNetDataset, VelNetDatasetWithEncoder
        return VelNetDataset if name == 'VelNetDataset' else VelNetDatasetWithEncoder
    elif name in ('VelNetTrainer', 'autoregressive_test'):
        from .trainer import VelNetTrainer, autoregressive_test
        return VelNetTrainer if name == 'VelNetTrainer' else autoregressive_test
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
