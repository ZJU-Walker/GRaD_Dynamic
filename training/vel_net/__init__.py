"""
Velocity Network Training Pipeline.

Contains:
- DataCollector: Collect flight data from geometric controller
- VelNetDataset: PyTorch Dataset for training
- VelNetTrainer: Training loop with curriculum learning

Usage:
    # Collect training data
    python training/vel_net/train_vel_net.py collect --map gate_mid --n_sequences 30

    # Train model
    python training/vel_net/train_vel_net.py train --data_dir data/vel_net/sequences
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
