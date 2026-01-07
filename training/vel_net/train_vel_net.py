#!/usr/bin/env python3
"""
Velocity Network Training Script.

Main entry point for data collection and training.

Usage:
    # Collect training data
    python training/vel_net/train_vel_net.py collect \
        --map gate_mid \
        --n_sequences 30 \
        --v_min 0.5 --v_max 2.0

    # Train model (with scheduled sampling)
    python training/vel_net/train_vel_net.py train \
        --data_dir data/vel_net/sequences \
        --epochs 200 \
        --batch_size 8 \
        --seq_length 64 \
        --tf_start_epoch 0 --tf_end_epoch 100 \
        --wandb

    # Auto-regressive test
    python training/vel_net/train_vel_net.py test \
        --checkpoint checkpoints/vel_net/best.pt \
        --test_seq data/vel_net/sequences/seq_0000

    # Evaluation flight
    python training/vel_net/train_vel_net.py eval \
        --checkpoint checkpoints/vel_net/best.pt \
        --map gate_mid --v_avg 1.0
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch

from models.vel_net import VELO_NET
from models.vel_net.visual_encoder import DualEncoder


def collect_command(args):
    """Run data collection."""
    from training.vel_net.data_collector import collect_sequences

    collect_sequences(
        output_dir=args.output_dir,
        map_name=args.map,
        n_sequences=args.n_sequences,
        collection_freq=args.freq,
        v_min=args.v_min,
        v_max=args.v_max,
        smoothing=args.smoothing,
        device=args.device,
    )


def train_command(args):
    """Run training with scheduled sampling."""
    from training.vel_net.dataset import create_dataloaders
    from training.vel_net.trainer import VelNetTrainer, autoregressive_test

    device = args.device

    print(f"\n{'='*60}")
    print(f"Velocity Network Training (Scheduled Sampling)")
    print(f"{'='*60}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Device: {device}")
    print(f"  Seq length: {args.seq_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  TF schedule: 100%% GT (epoch 0-{args.tf_start_epoch}) -> decay -> 0%% GT (epoch {args.tf_end_epoch}+)")
    print(f"  Wandb: {args.wandb}")
    print(f"{'='*60}\n")

    # Create encoder
    encoder = DualEncoder(rgb_dim=32, depth_dim=32).to(device)
    print(f"Encoder created: {encoder.num_trainable_params():,} trainable params")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        stride=args.stride,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
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
        tf_start_epoch=args.tf_start_epoch,
        tf_end_epoch=args.tf_end_epoch,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(
        n_epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
    )

    # Final auto-regressive test
    print("\nRunning final auto-regressive test...")
    import glob
    sequences = sorted(glob.glob(str(Path(args.data_dir) / "seq_*")))
    if sequences:
        test_seq = sequences[-1]
        ar_metrics = autoregressive_test(
            model=model,
            encoder=encoder,
            test_sequence_path=test_seq,
            vel_mean=trainer.vel_mean,
            vel_std=trainer.vel_std,
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
    print(f"  Epoch: {checkpoint['epoch']}")

    # Load velocity normalization stats
    vel_mean = checkpoint.get('vel_mean', torch.zeros(3))
    vel_std = checkpoint.get('vel_std', torch.ones(3))
    print(f"  Velocity norm: mean={vel_mean.cpu().numpy()}, std={vel_std.cpu().numpy()}")

    # Run test
    metrics = autoregressive_test(
        model=model,
        encoder=encoder,
        test_sequence_path=args.test_seq,
        vel_mean=vel_mean,
        vel_std=vel_std,
        device=device,
        max_steps=args.max_steps,
        save_plot=True,
        plot_path=args.plot_path,
    )


def eval_command(args):
    """Run evaluation flight with trained model."""
    from training.vel_net.evaluator import fly_and_evaluate

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
    print(f"  Epoch: {checkpoint['epoch']}")

    # Load velocity normalization stats
    vel_mean = checkpoint.get('vel_mean', torch.zeros(3))
    vel_std = checkpoint.get('vel_std', torch.ones(3))
    print(f"  Velocity norm: mean={vel_mean.cpu().numpy()}, std={vel_std.cpu().numpy()}")

    # Run evaluation flight
    fly_and_evaluate(
        model=model,
        encoder=encoder,
        vel_mean=vel_mean,
        vel_std=vel_std,
        map_name=args.map,
        v_avg=args.v_avg,
        output_dir=args.output_dir,
        device=device,
        max_steps=args.max_steps,
        smoothing=args.smoothing,
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
    collect_parser.add_argument('--v_min', type=float, default=0.5,
                                help='Min velocity (m/s)')
    collect_parser.add_argument('--v_max', type=float, default=2.0,
                                help='Max velocity (m/s)')
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

    # Sequence config
    train_parser.add_argument('--seq_length', type=int, default=64,
                              help='Sequence chunk length')
    train_parser.add_argument('--stride', type=int, default=32,
                              help='Stride between sequence chunks')

    # Training config
    train_parser.add_argument('--epochs', type=int, default=200,
                              help='Maximum epochs')
    train_parser.add_argument('--batch_size', type=int, default=8,
                              help='Batch size (number of sequences)')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                              help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5,
                              help='Weight decay')
    train_parser.add_argument('--grad_clip', type=float, default=1.0,
                              help='Gradient clipping')
    train_parser.add_argument('--val_ratio', type=float, default=0.1,
                              help='Validation ratio')
    train_parser.add_argument('--early_stop_patience', type=int, default=30,
                              help='Early stopping patience')

    # Scheduled sampling config
    train_parser.add_argument('--tf_start_epoch', type=int, default=0,
                              help='Epoch to start decaying (before = 100%% GT)')
    train_parser.add_argument('--tf_end_epoch', type=int, default=100,
                              help='Epoch to finish decaying (after = 0%% GT)')

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
    test_parser.add_argument('--plot_path', type=str, default=None,
                             help='Path to save plot')
    test_parser.add_argument('--device', type=str, default='cuda:0',
                             help='PyTorch device')

    # =========================================================================
    # Eval command
    # =========================================================================
    eval_parser = subparsers.add_parser('eval', help='Evaluate model with live flight')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to checkpoint')
    eval_parser.add_argument('--map', type=str, default='gate_mid',
                             choices=['gate_mid', 'gate_left', 'gate_right'],
                             help='Map name')
    eval_parser.add_argument('--v_avg', type=float, default=1.0,
                             help='Average velocity (m/s)')
    eval_parser.add_argument('--smoothing', type=float, default=0.018,
                             help='B-spline corner smoothing')
    eval_parser.add_argument('--max_steps', type=int, default=3000,
                             help='Maximum simulation steps')
    eval_parser.add_argument('--output_dir', type=str, default='output/vel_net_eval',
                             help='Output directory')
    eval_parser.add_argument('--device', type=str, default='cuda:0',
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
    elif args.command == 'eval':
        eval_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
