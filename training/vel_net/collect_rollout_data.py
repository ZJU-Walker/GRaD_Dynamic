"""
Collect rollout data for vel_net training from GradNav policy evaluation.

Runs the trained GradNav policy with dynamic obstacles and saves:
- RGB images (with augmented dynamic objects)
- Depth images (with augmented dynamic objects)
- Telemetry (position, velocity, quaternion, actions)

Data format matches existing vel_net training pipeline.

Usage:
    python training/vel_net/collect_rollout_data.py \
        --checkpoint /path/to/best_policy.pt \
        --cfg examples/cfg/gradnav/drone_dynamic_curriculum.yaml \
        --output_dir /scr/irislab/ke/data/vel_net \
        --num_episodes 50 \
        --freq 30
"""

import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import envs
from models.policy import actor
from models.policy.vae import VAE
from utils.common import seeding, print_info
from utils.running_mean_std import RunningMeanStd


class RolloutDataCollector:
    """Collect rollout data from GradNav policy for vel_net training."""

    def __init__(
        self,
        checkpoint_path: str,
        cfg_path: str,
        output_dir: str,
        device: str = 'cuda:0',
        collection_freq: float = 30.0,
        seed: int = 0,
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Add general section (like train_gradnav_dynamic.py does)
        self.cfg["params"]["general"] = {
            "device": device,
            "seed": seed,
            "render": True,
            "checkpoint": checkpoint_path,
        }

        # Calculate collection interval
        self.sim_freq = 1.0 / self.cfg["params"]["diff_env"].get("sim_dt", 0.02)  # Usually 50Hz
        self.collection_freq = collection_freq
        self.collection_interval = max(1, int(self.sim_freq / collection_freq))
        print_info(f"Sim freq: {self.sim_freq}Hz, Collection freq: {collection_freq}Hz, Interval: {self.collection_interval} steps")

        # Create environment
        self._create_env()

        # Load policy
        self._load_policy(checkpoint_path)

        # Data buffers
        self.reset_buffers()

    def _create_env(self):
        """Create the DynamicDroneEnv."""
        env_fn = getattr(envs, self.cfg["params"]["diff_env"]["name"])
        map_name = self.cfg["params"]["config"].get("map_name", 'gate_mid')
        env_hyper = self.cfg["params"].get("env_hyper", None)
        vel_net_cfg = self.cfg["params"].get("vel_net", None)
        dynamic_objects_cfg = self.cfg["params"].get("dynamic_objects", None)

        self.env = env_fn(
            num_envs=1,  # Single env for data collection
            device=self.device,
            render=True,  # Need render for RGB/depth
            seed=self.cfg["params"]["general"]["seed"],
            episode_length=self.cfg["params"]["diff_env"].get("episode_length", 250),
            stochastic_init=self.cfg["params"]["diff_env"].get("stochastic_env", True),
            MM_caching_frequency=self.cfg["params"]['diff_env'].get('MM_caching_frequency', 1),
            map_name=map_name,
            env_hyper=env_hyper,
            vel_net_cfg=vel_net_cfg,
            dynamic_objects_cfg=dynamic_objects_cfg,
            no_grad=True,
        )

        # Force Phase 2 with 100% spawn probability
        if hasattr(self.env, 'use_dynamic_objects') and self.env.use_dynamic_objects:
            self.env.current_phase = 2
            self.env.dynamic_spawn_prob = 1.0
            print_info(f"[Collector] Forced Phase 2 with 100% dynamic object spawn")
        else:
            print_info(f"[Collector] Dynamic objects not enabled")

        # Force 100% predicted velocity if enabled
        if hasattr(self.env, 'use_pred_vel_in_obs') and self.env.use_pred_vel_in_obs:
            self.env.gt_vel_ratio = 0.0
            print_info(f"[Collector] Using 100% predicted velocity")

        print_info(f"Environment created: {self.env.__class__.__name__}")

    def _load_policy(self, checkpoint_path: str):
        """Load the trained policy from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create actor network (same pattern as gradnav_dynamic.py)
        actor_name = self.cfg["params"]["network"].get("actor", 'ActorStochasticMLP')
        actor_fn = getattr(actor, actor_name)
        self.actor = actor_fn(
            self.env.num_obs,
            self.env.num_actions,
            self.cfg['params']['network'],
            device=self.device,
        )
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        # Create VAE (same pattern as gradnav_dynamic.py)
        kl_weight = self.cfg["params"]["vae"].get("kl_weight", 1.0)
        vae_encoder_dim = self.cfg["params"]["vae"].get("encoder_units", [256, 256, 256])
        vae_decoder_dim = self.cfg["params"]["vae"].get("decoder_units", [32, 64, 128, 256])
        self.vae = VAE(
            self.env.num_obs,
            self.env.num_history,
            self.env.num_latent,
            kld_weight=kl_weight,
            activation='elu',
            decoder_hidden_dims=vae_decoder_dim,
            encoder_hidden_dims=vae_encoder_dim,
        ).to(self.device)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae.eval()

        # Load visual net
        self.env.visual_net.load_state_dict(checkpoint['visual_net_state_dict'])
        self.env.visual_net.eval()

        # Load observation normalization
        self.obs_rms = None
        if 'obs_rms' in checkpoint and checkpoint['obs_rms'] is not None:
            self.obs_rms = checkpoint['obs_rms'].to(self.device)

        # Initialize observation history buffer
        self.obs_hist_buf = torch.zeros(1, self.env.num_history, self.env.num_obs, device=self.device)

        print_info(f"Policy loaded from: {checkpoint_path}")

    def reset_buffers(self):
        """Reset data collection buffers."""
        self.timestamps = []
        self.positions = []
        self.velocities = []
        self.orientations = []
        self.actions = []
        self.rgb_images = []
        self.depth_images = []

    def collect_frame(self, step: int, action: np.ndarray):
        """Collect a single frame of data."""
        # Get state from environment
        pos = self.env.state_joint_q[0, 0:3].cpu().numpy()  # Position
        quat = self.env.state_joint_q[0, 3:7].cpu().numpy()  # Quaternion (xyzw)
        vel = self.env.state_joint_qd[0, 3:6].cpu().numpy()  # Velocity (world frame)

        # Get images (use augmented if available for dynamic objects)
        if hasattr(self.env, 'augmented_rgb') and self.env.augmented_rgb is not None:
            rgb = self.env.augmented_rgb[0].cpu().numpy()
            depth = self.env.augmented_depth[0, :, :, 0].cpu().numpy()
        else:
            # Fallback: would need to access raw GS render
            print_info("[Warning] No augmented images available")
            return

        # Calculate timestamp
        timestamp = step * (1.0 / self.sim_freq)

        # Append to buffers
        self.timestamps.append(timestamp)
        self.positions.append(pos.copy())
        self.velocities.append(vel.copy())
        self.orientations.append(quat.copy())
        self.actions.append(action.copy())
        self.rgb_images.append(rgb.copy())
        self.depth_images.append(depth.copy())

    def save_sequence(self, seq_idx: int) -> str:
        """Save collected data to disk."""
        seq_dir = self.output_dir / f"seq_{seq_idx:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        # Save telemetry
        np.savez_compressed(
            seq_dir / "telemetry.npz",
            timestamps=np.array(self.timestamps, dtype=np.float32),
            positions=np.array(self.positions, dtype=np.float32),
            velocities=np.array(self.velocities, dtype=np.float32),
            orientations=np.array(self.orientations, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
        )

        # Save RGB images
        rgb_dir = seq_dir / "rgb"
        rgb_dir.mkdir(exist_ok=True)
        for i, rgb in enumerate(self.rgb_images):
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            Image.fromarray(rgb).save(rgb_dir / f"{i:06d}.png")

        # Save depth images
        depth_dir = seq_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        for i, depth in enumerate(self.depth_images):
            np.save(depth_dir / f"{i:06d}.npy", depth.astype(np.float32))

        num_frames = len(self.timestamps)
        self.reset_buffers()

        return str(seq_dir), num_frames

    @torch.no_grad()
    def run_episode(self) -> int:
        """Run a single episode and collect data.

        Returns:
            Number of frames collected
        """
        obs = self.env.reset()
        self.obs_hist_buf.zero_()

        done = False
        step = 0
        frames_collected = 0

        while not done:
            # Normalize observation
            raw_obs = obs.clone()
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            # Get action from policy
            actions = self.actor(obs, deterministic=True)
            actions = torch.tanh(actions)

            # Get VAE output
            vae_output, _ = self.vae.forward(self.obs_hist_buf)

            # Collect data before stepping (at collection frequency)
            if step % self.collection_interval == 0:
                action_np = actions[0].cpu().numpy()
                self.collect_frame(step, action_np)
                frames_collected += 1

            # Step environment
            obs, privilege_obs, history_obs, vel_obs, rew, done_tensor, extra_info = self.env.step(actions, vae_output)

            # Update history buffer
            self.obs_hist_buf = history_obs

            done = done_tensor[0].item()
            step += 1

        return frames_collected

    def collect_episodes(self, num_episodes: int):
        """Collect data from multiple episodes."""
        print_info(f"Collecting {num_episodes} episodes to {self.output_dir}")

        pbar = tqdm(range(num_episodes), desc="Collecting episodes")
        total_frames = 0

        for ep_idx in pbar:
            frames = self.run_episode()
            seq_dir, saved_frames = self.save_sequence(ep_idx)

            total_frames += saved_frames
            pbar.set_postfix({
                'frames': saved_frames,
                'total': total_frames,
                'seq': seq_dir.split('/')[-1],
            })

        print_info(f"Collection complete: {num_episodes} episodes, {total_frames} total frames")
        print_info(f"Data saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect rollout data for vel_net training")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained policy checkpoint')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='/scr/irislab/ke/data/vel_net',
                        help='Output directory for collected data')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--freq', type=float, default=30.0,
                        help='Data collection frequency in Hz')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()

    collector = RolloutDataCollector(
        checkpoint_path=args.checkpoint,
        cfg_path=args.cfg,
        output_dir=args.output_dir,
        device=args.device,
        collection_freq=args.freq,
        seed=args.seed,
    )

    collector.collect_episodes(args.num_episodes)


if __name__ == '__main__':
    main()
