# GRaD_Dynamic_onboard

Drone waypoint navigation system using A* path planning, B-spline trajectory generation, and SE(3) geometric controller. Includes velocity estimation network (vel_net) for learning-based state estimation.

## Overview

The system performs:
1. **A* Path Planning** - Finds collision-free paths through point cloud obstacles
2. **B-Spline Trajectory Generation** - Creates smooth, flyable trajectories with velocity/acceleration profiles
3. **Geometric Controller** - SE(3) controller for accurate trajectory tracking
4. **Velocity Network** - Auto-regressive GRU network for velocity estimation

---

## Quick Start

### Run Navigation Simulation

```bash
python controller/waypoint_nav_geometric.py --map gate_mid --v_avg 0.5 --save_3d_plot
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--map` | Map name (`gate_mid`, `gate_left`, `gate_right`) | `gate_mid` |
| `--v_avg` | Average flight velocity (m/s) | `1.5` |
| `--corner_smoothing` | B-spline corner smoothing | `0.18` |
| `--output` | Output video path | auto-generated |
| `--save_3d_plot` | Save 3D trajectory plot | off |
| `--save_traj_data` | Save trajectory data to `.npz` | off |
| `--traj_only` | Only generate trajectory (no flying) | off |

---

## Velocity Network Training

### 1. Data Collection

Collect flight data using the geometric controller:

```bash
# Collect 30 sequences with velocity variation [0.5, 2.0] m/s
python training/vel_net/train_vel_net.py collect \
    --map gate_mid \
    --n_sequences 30 \
    --freq 30 \
    --v_min 0.5 --v_max 2.0 \
    --output_dir data/vel_net/sequences

# Collect with fixed velocity (v_min == v_max)
python training/vel_net/train_vel_net.py collect \
    --map gate_mid \
    --n_sequences 10 \
    --v_min 1.0 --v_max 1.0
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--map` | Map name | `gate_mid` |
| `--n_sequences` | Number of sequences | `30` |
| `--freq` | Collection frequency (Hz) | `30` |
| `--v_min` | Min velocity (m/s) | `0.5` |
| `--v_max` | Max velocity (m/s). If != v_min, random per sequence | `2.0` |
| `--smoothing` | B-spline corner smoothing factor | `0.018` |
| `--output_dir` | Output directory | `data/vel_net/sequences` |

**Note:** If `v_min == v_max`, fixed velocity is used. Otherwise, random velocity is sampled from [v_min, v_max] for each sequence.

**Collection Progress Output:**
```
[ 1/10] v=1.23m/s: 100%|████████████████████| OK | 1467 frames | 29.3s
[ 2/10] v=0.87m/s: 100%|████████████████████| COLLISION | 453 frames | 9.1s
[ 3/10] v=1.95m/s: 100%|████████████████████| TIMEOUT | 1200 frames | 24.0s
```

**Output structure:**
```
data/vel_net/sequences/
├── seq_0000/
│   ├── telemetry.npz           # positions, velocities, orientations, actions
│   ├── rgb/                    # RGB images (*.png)
│   ├── depth/                  # Depth images (*.npy)
│   ├── astar_bspline.png       # A* path vs B-spline comparison (before flying)
│   ├── trajectory_profile.png  # Position, velocity, acceleration profiles (before flying)
│   └── trajectory.png          # Actual vs desired trajectory (after flying)
├── seq_0001/
│   └── ...
```

### 2. Training

Train the velocity network with curriculum learning:

```bash
# Train with wandb logging
python training/vel_net/train_vel_net.py train \
    --data_dir data/vel_net/sequences \
    --epochs 500 \
    --batch_size 64 \
    --wandb

# Train without wandb
python training/vel_net/train_vel_net.py train \
    --data_dir data/vel_net/sequences \
    --epochs 500 \
    --batch_size 64
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Path to sequences | `data/vel_net/sequences` |
| `--epochs` | Maximum epochs | `500` |
| `--batch_size` | Batch size | `64` |
| `--lr` | Learning rate | `1e-4` |
| `--wandb` | Enable wandb logging | off |
| `--checkpoint_dir` | Checkpoint directory | `checkpoints/vel_net` |
| `--resume` | Resume from checkpoint | None |

**Training Stages (Curriculum Learning):**

| Stage | Name | Loss | Description |
|-------|------|------|-------------|
| **A** | Imitation | MSE | Pure supervised learning - match ground truth velocity |
| **B** | PINN | MSE + Physics | Adds physics-informed constraints after Stage A plateaus |

- Training starts in **Stage A** with simple MSE loss
- After validation loss stops improving for `stage_patience` epochs (default: 20), transitions to **Stage B**
- **Stage B** adds physics-informed regularization to improve generalization
- Early stopping triggers after `early_stop_patience` epochs (default: 30) without improvement in Stage B

**Progress bar output:**
```
Training:  10%|████          | 50/500 [02:30<22:30] stage=A, loss=0.0234, val_mae=0.0156, lr=1.0e-04
Epoch  50 [A]: 100%|████████████████████| 542/542 [00:03] loss=0.0234, mse=0.0234
```

### 3. Evaluation

Test auto-regressive inference:

```bash
python training/vel_net/train_vel_net.py eval \
      --checkpoint checkpoints/vel_net/best.pt \
      --map gate_mid \
      --v_avg 1.0 \
      --output_dir output/vel_net_eval
```

---

## Velocity Network Architecture

### Observation Structure (81 dims)

| Index | Component | Dims | Description |
|-------|-----------|------|-------------|
| 0-5 | Rot6D | 6 | Rotation (first 2 cols of rotation matrix) |
| 6-9 | Action | 4 | Current [roll_rate, pitch_rate, yaw_rate, thrust] |
| 10-13 | Prev Action | 4 | Previous action |
| 14-16 | Prev Velocity | 3 | Auto-regressive term (GT for training, predicted for inference) |
| 17-48 | RGB Features | 32 | MobileNetV3 encoder output |
| 49-80 | Depth Features | 32 | MobileNetV3 encoder output |

### Model Architecture

```
Input (81 dims)
    ↓
LayerNorm
    ↓
Projector MLP (81 → 256)
    ↓
GRU (3 layers, 256 hidden)
    ↓
Head MLP (256 → 128)
    ↓
vel_mu / vel_var → Velocity (3D)
```

### Visual Encoder

- **Backbone**: MobileNetV3-Small (frozen, ImageNet pretrained)
- **FC Layer**: 576 → 32 (trainable, learns jointly with vel_net)

### Usage Example

```python
from models.vel_net import VELO_NET, DualEncoder, build_vel_observation_from_quat

# Create model and encoder
model = VELO_NET(num_obs=81, stack_size=1, device='cuda:0')
encoder = DualEncoder(rgb_dim=32, depth_dim=32)

# Encode images
rgb_feat, depth_feat = encoder(rgb_image, depth_image)

# Build observation
obs = build_vel_observation_from_quat(
    quat=quaternion,          # (B, 4) xyzw
    action=action,            # (B, 4)
    prev_action=prev_action,  # (B, 4)
    prev_vel=prev_vel,        # (B, 3)
    rgb_feat=rgb_feat,        # (B, 32)
    depth_feat=depth_feat,    # (B, 32)
)

# Predict velocity
vel_mu, vel_logvar = model.encode(obs)
```

---

## Project Structure

```
GRaD_Dynamic_onboard/
├── models/
│   └── vel_net/                      # Velocity estimation network
│       ├── vel_net.py                # VELO_NET model (81-dim input)
│       ├── visual_encoder.py         # MobileNetV3 encoder
│       ├── vel_obs_buffer.py         # History buffer
│       └── vel_obs_utils.py          # Rot6D, observation builders
│
├── training/
│   └── vel_net/                      # Training pipeline
│       ├── data_collector.py         # Flight data collection
│       ├── dataset.py                # PyTorch Dataset
│       ├── trainer.py                # Curriculum learning trainer
│       └── train_vel_net.py          # Main entry point
│
├── controller/
│   ├── waypoint_nav_geometric.py     # Main navigation script
│   ├── nav_helpers.py                # Helper functions
│   └── geometric_controller.py       # SE(3) controller
│
├── trajectory/
│   └── bspline_trajectory.py         # B-spline trajectory generation
│
├── envs/
│   └── drone_env.py                  # Simulation environment
│
└── utils/
    └── traj_planner_global.py        # A* path planner
```

---

## Coordinate Systems

| Space | Description | Example Start Position |
|-------|-------------|------------------------|
| Point Cloud | A* planning, waypoint definitions | [-6, 0, 1.2] |
| Simulation | Drone state, trajectory tracking | [0, 0, 1.2] |
| GS Rendering | Internal to env | Handled automatically |

**Conversion**: `sim_pos = pc_pos + [6, 0, 0]`

---

## Map Configurations

| Map | Start (PC space) | Waypoints | Destination |
|-----|------------------|-----------|-------------|
| `gate_mid` | [-6, 0, 1.2] | 4 waypoints | [7.5, -2, 1.2] |
| `gate_left` | [-6, 0, 1.2] | 3 waypoints | [7, -2, 1.2] |
| `gate_right` | [-6, 0, 1.3] | 5 waypoints | [7, -2, 1.3] |
