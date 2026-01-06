#!/usr/bin/env python3
"""
Velocity Network Training Script.

Main entry point for data collection and training.

Usage:
    # Collect training data
    python training/vel_net/train_vel_net.py collect \
        --map gate_mid \
        --n_sequences 30 \
        --freq 30 \
        --output_dir data/vel_net/sequences

    # Train model
    python training/vel_net/train_vel_net.py train \
        --data_dir data/vel_net/sequences \
        --epochs 500 \
        --batch_size 64 \
        --wandb

    # Auto-regressive test
    python training/vel_net/train_vel_net.py test \
        --checkpoint checkpoints/vel_net/best.pt \
        --test_seq data/vel_net/sequences/seq_0000
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch

from models.vel_net import VELO_NET
from models.vel_net.visual_encoder import DualEncoder


def collect_command(args):
    """Run data collection."""
    from training.vel_net.data_collector import collect_sequences

    # Only enable velocity variation if --vary_velocity flag is set
    v_avg_range = (args.v_min, args.v_max) if args.vary_velocity else None

    collect_sequences(
        output_dir=args.output_dir,
        map_name=args.map,
        n_sequences=args.n_sequences,
        collection_freq=args.freq,
        v_avg=args.v_avg,
        v_avg_range=v_avg_range,
        smoothing=args.smoothing,
        device=args.device,
    )


def train_command(args):
    """Run training."""
    from training.vel_net.dataset import create_dataloaders
    from training.vel_net.trainer import VelNetTrainer, autoregressive_test

    device = args.device

    print(f"\n{'='*60}")
    print(f"Velocity Network Training")
    print(f"{'='*60}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Wandb: {args.wandb}")
    print(f"{'='*60}\n")

    # Create encoder
    encoder = DualEncoder(rgb_dim=32, depth_dim=32).to(device)
    print(f"Encoder created: {encoder.num_trainable_params():,} trainable params")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        encoder=encoder,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        device=device,
    )

    # Create model
    model = VELO_NET(
        num_obs=81,
        stack_size=1,
        num_latent=args.num_latent,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        device=device,
    ).to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} params")

    # Create trainer
    trainer = VelNetTrainer(
        model=model,
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        pinn_weight=args.pinn_weight,
        stage_patience=args.stage_patience,
        early_stop_patience=args.early_stop_patience,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(n_epochs=args.epochs)

    # Auto-regressive test on first validation sequence
    print("\nRunning auto-regressive test...")
    import glob
    sequences = sorted(glob.glob(str(Path(args.data_dir) / "seq_*")))
    if sequences:
        test_seq = sequences[-1]  # Use last sequence for testing
        ar_metrics = autoregressive_test(
            model=model,
            encoder=encoder,
            test_sequence_path=test_seq,
            device=device,
        )

    print(f"\nTraining complete! Best model saved to {args.checkpoint_dir}/best.pt")


def test_command(args):
    """Run auto-regressive test."""
    from training.vel_net.trainer import autoregressive_test

    device = args.device

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create encoder
    encoder = DualEncoder(rgb_dim=32, depth_dim=32).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Create model
    config = checkpoint['model_config']
    model = VELO_NET(
        num_obs=config['num_obs'],
        stack_size=config['stack_size'],
        hidden_dim=config['hidden_dim'],
        gru_layers=config['gru_layers'],
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  Epoch: {checkpoint['epoch']}, Stage: {checkpoint['stage']}")

    # Run test
    metrics = autoregressive_test(
        model=model,
        encoder=encoder,
        test_sequence_path=args.test_seq,
        device=device,
        max_steps=args.max_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Velocity Network Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # =========================================================================
    # Collect command
    # =========================================================================
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--output_dir', type=str, default='data/vel_net/sequences',
                                help='Output directory for sequences')
    collect_parser.add_argument('--map', type=str, default='gate_mid',
                                choices=['gate_mid', 'gate_left', 'gate_right'],
                                help='Map name')
    collect_parser.add_argument('--n_sequences', type=int, default=30,
                                help='Number of sequences to collect')
    collect_parser.add_argument('--freq', type=float, default=30.0,
                                help='Collection frequency (Hz)')
    collect_parser.add_argument('--v_avg', type=float, default=0.5,
                                help='Average velocity (m/s), same as waypoint_nav_geometric.py')
    collect_parser.add_argument('--vary_velocity', action='store_true',
                                help='Enable velocity variation (random v_avg per sequence)')
    collect_parser.add_argument('--v_min', type=float, default=0.5,
                                help='Min velocity for variation (only used with --vary_velocity)')
    collect_parser.add_argument('--v_max', type=float, default=2.0,
                                help='Max velocity for variation (only used with --vary_velocity)')
    collect_parser.add_argument('--smoothing', type=float, default=0.018,
                                help='B-spline corner smoothing factor')
    collect_parser.add_argument('--device', type=str, default='cuda:0',
                                help='PyTorch device')

    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser('train', help='Train velocity network')
    train_parser.add_argument('--data_dir', type=str, default='data/vel_net/sequences',
                              help='Path to sequences directory')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vel_net',
                              help='Checkpoint directory')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Resume from checkpoint')

    # Model config
    train_parser.add_argument('--hidden_dim', type=int, default=256,
                              help='Hidden dimension')
    train_parser.add_argument('--num_latent', type=int, default=64,
                              help='Latent dimension')
    train_parser.add_argument('--gru_layers', type=int, default=3,
                              help='Number of GRU layers')

    # Training config
    train_parser.add_argument('--epochs', type=int, default=500,
                              help='Maximum epochs')
    train_parser.add_argument('--batch_size', type=int, default=64,
                              help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                              help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5,
                              help='Weight decay')
    train_parser.add_argument('--grad_clip', type=float, default=1.0,
                              help='Gradient clipping')
    train_parser.add_argument('--val_ratio', type=float, default=0.1,
                              help='Validation ratio')
    train_parser.add_argument('--num_workers', type=int, default=4,
                              help='DataLoader workers')

    # Curriculum config
    train_parser.add_argument('--pinn_weight', type=float, default=0.1,
                              help='PINN loss weight')
    train_parser.add_argument('--stage_patience', type=int, default=20,
                              help='Epochs before A->B transition')
    train_parser.add_argument('--early_stop_patience', type=int, default=30,
                              help='Early stopping patience')

    # Logging
    train_parser.add_argument('--wandb', action='store_true',
                              help='Enable wandb logging')
    train_parser.add_argument('--wandb_project', type=str, default='vel_net',
                              help='Wandb project name')

    train_parser.add_argument('--device', type=str, default='cuda:0',
                              help='PyTorch device')

    # =========================================================================
    # Test command
    # =========================================================================
    test_parser = subparsers.add_parser('test', help='Auto-regressive test')
    test_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to checkpoint')
    test_parser.add_argument('--test_seq', type=str, required=True,
                             help='Path to test sequence')
    test_parser.add_argument('--max_steps', type=int, default=300,
                             help='Maximum test steps')
    test_parser.add_argument('--device', type=str, default='cuda:0',
                             help='PyTorch device')

    # =========================================================================
    # Parse and dispatch
    # =========================================================================
    args = parser.parse_args()

    if args.command == 'collect':
        collect_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'test':
        test_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
