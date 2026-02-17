# GRaD_Dynamic_onboard

Drone waypoint navigation system using A* path planning, B-spline trajectory generation, SE(3) geometric controller, and velocity estimation network.

## Overview

The system performs:
1. **A* Path Planning** - Finds collision-free paths through point cloud obstacles
2. **B-Spline Trajectory Generation** - Creates smooth, flyable trajectories with velocity/acceleration profiles
3. **Geometric Controller** - SE(3) controller for accurate trajectory tracking
4. **Velocity Network** - Auto-regressive GRU network for velocity estimation (84-dim input with direct delta-v prediction)

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

Collect flight data using the geometric controller.

#### Preview Trajectories First

Before collecting data, preview trajectories to verify they're good:

```bash
# Preview a trajectory (saves video without collecting data)
# --map: GS/point cloud map (gate_mid, gate_left, gate_right, clutter, backroom, flightroom)
# --waypoints: trajectory configuration (gate_mid, gate_mid_high, zigzag, etc.)
python training/vel_net/train_vel_net.py collect \
    --map gate_mid --waypoints gate_mid --preview --v_avg 1.0

# Preview different waypoints on the same map
python training/vel_net/train_vel_net.py collect --map gate_mid --waypoints gate_mid_high --preview --v_avg 1.0
python training/vel_net/train_vel_net.py collect --map gate_mid --waypoints zigzag --preview --v_avg 1.0

# If --map is omitted, uses the default map from waypoints config
python training/vel_net/train_vel_net.py collect --waypoints gate_left --preview --v_avg 1.0
```

Videos saved to `output/preview_{waypoints}_v{v_avg}.mp4`.

#### Collect Training Data

```bash
# Collect 30 sequences with velocity variation [0.5, 2.0] m/s
# --map: GS/point cloud map to use for rendering
# --waypoints: trajectory configuration for waypoints
python training/vel_net/train_vel_net.py collect \
    --map gate_mid \
    --waypoints gate_mid \
    --n_sequences 30 \
    --freq 30 \
    --v_min 0.5 --v_max 2.0 \
    --output_dir data/vel_net/sequences

# Collect with action noise (recommended for robustness)
python training/vel_net/train_vel_net.py collect \
    --map gate_mid \
    --waypoints gate_mid \
    --n_sequences 30 \
    --v_min 0.5 --v_max 2.0 \
    --action_noise 0.1 \
    --output_dir data/vel_net/sequences_noisy

# Collect diverse trajectories with action noise
# Use same map with different waypoint configurations
python training/vel_net/train_vel_net.py collect \
    --map gate_mid --waypoints gate_mid --n_sequences 10 --v_min 0.5 --v_max 2.0 \
    --action_noise 0.1 --output_dir data/vel_net/sequences_diverse

python training/vel_net/train_vel_net.py collect \
    --map gate_mid --waypoints zigzag --n_sequences 10 --v_min 0.5 --v_max 2.0 \
    --action_noise 0.1 --output_dir data/vel_net/sequences_diverse

python training/vel_net/train_vel_net.py collect \
    --map gate_mid --waypoints gate_mid_high --n_sequences 10 --v_min 0.5 --v_max 2.0 \
    --action_noise 0.1 --output_dir data/vel_net/sequences_diverse

# Collect with waypoint randomization (±0.3m variation per sequence)
python training/vel_net/train_vel_net.py collect \
    --map gate_mid --waypoints gate_mid --n_sequences 100 --v_min 0.3 --v_max 1.5 \
    --action_noise 0.1 --waypoint_noise 0.1 \
    --output_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--map` | GS/point cloud map name (see below) | from waypoints config |
| `--waypoints` | Waypoint trajectory configuration (see below) | `gate_mid` |
| `--n_sequences` | Number of sequences | `30` |
| `--freq` | Collection frequency (Hz) | `30` |
| `--v_min` | Min velocity (m/s) | `0.5` |
| `--v_max` | Max velocity (m/s). If != v_min, random per sequence | `2.0` |
| `--smoothing` | B-spline corner smoothing factor | `0.018` |
| `--action_noise` | Action noise std (0.0-0.3) for robustness | `0.0` |
| `--waypoint_noise` | Waypoint position noise std (m), adds ±noise to each waypoint per sequence | `0.0` |
| `--output_dir` | Output directory | `data/vel_net/sequences` |
| `--preview` | Preview trajectory only (no data collection) | off |
| `--v_avg` | Average velocity for preview mode | `(v_min+v_max)/2` |

**Available GS Maps (`--map`):**
| Map | Description |
|-----|-------------|
| `gate_mid` | Main gate scene (center view) |
| `gate_left` | Gate scene (left view) |
| `gate_right` | Gate scene (right view) |
| `clutter` | Cluttered environment |
| `backroom` | Backroom scene |
| `flightroom` | Flight room scene |

**Available Waypoint Trajectories (`--waypoints`):**
| Trajectory | Default Map | Description |
|------------|-------------|-------------|
| `gate_mid` | gate_mid | Through center of gate, default trajectory |
| `gate_mid_high` | gate_mid | Higher altitude version of gate_mid |
| `gate_mid_low` | gate_mid | Lower altitude version of gate_mid |
| `gate_left` | gate_left | Through left side of gate |
| `gate_right` | gate_mid | Through right side of gate |
| `zigzag` | gate_mid | Zigzag pattern with lateral movements |
| `straight` | gate_mid | Straight line trajectory |
| `reverse` | gate_mid | Reverse direction of gate_mid |

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
python training/vel_net/precompute_features.py \
      --data_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata \
      --device cuda:0
# Train with wandb logging
 python training/vel_net/train_vel_net.py train \
      --data_dir /scr/irislab/ke/data/vel_net/sequences_0106 \
      --epochs 500 \
      --batch_size 16 \
      --seq_length 32 --stride 16 \
      --tf_start_epoch 10 --tf_end_epoch 60 \
      --wandb --checkpoint_dir /scr/irislab/ke/checkpoints/vel_net_0124_delta
# Train with body velocity
python training/vel_net_body/train_vel_net_body.py train \
      --data_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata \
      --checkpoint_dir checkpoints/vel_net_body \
      --epochs 500 \
      --batch_size 16 \
      --seq_length 32 --stride 16 \
      --tf_start_epoch 10 --tf_end_epoch 60 \
      --device cuda:0

# Train with full seq body
python training/vel_net_body/train_vel_net_body.py train \
      --data_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata \
      --checkpoint_dir checkpoints/vel_net_body_full_seq \
      --epochs 500 \
      --batch_size 1 \
      --grad_accum 8 \
      --seq_length 0 \
      --tf_start_epoch 20 --tf_end_epoch 150 \
      --device cuda:0
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Path to sequences | `data/vel_net/sequences` |
| `--epochs` | Maximum epochs | `500` |
| `--batch_size` | Batch size | `64` |
| `--lr` | Learning rate | `1e-4` |
| `--seq_length` | Frames per sequence chunk | `32` |
| `--stride` | Stride between chunks | `16` |
| `--tf_start_epoch` | Epoch to start TF decay | `10` |
| `--tf_end_epoch` | Epoch to end TF decay (0% GT after) | `60` |
| `--wandb` | Enable wandb logging | off |
| `--checkpoint_dir` | Checkpoint directory | `checkpoints/vel_net` |
| `--resume` | Resume from checkpoint | None |

**Training Output Explained:**
```
Epoch   3 | TF=1.00 | Loss=0.8234 | AR_MAE=0.1542 | LR=1.0e-04
```

| Metric | Meaning |
|--------|---------|
| **Epoch** | Current training iteration |
| **TF** | Teacher Forcing ratio - % of GT prev_vel used (1.0=100% GT, 0.0=100% predicted) |
| **Loss** | Training MSE on normalized velocities (all axes scaled to ~same range) |
| **AR_MAE** | Auto-Regressive MAE on validation set in **m/s** (real-world units, most important!) |
| **LR** | Current learning rate |

**Teacher Forcing Schedule:**
```
--tf_start_epoch 10 --tf_end_epoch 60

Epoch:    0         10        35        60        500
          |---------|---------|---------|---------|
TF:       1.0       1.0       0.5       0.0       0.0
          ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^
          100% GT   Start     Decaying  0% GT
          (easy)    decay               (realistic)
```

**What gets saved:**
- `best.pt` - Saved when **AR_MAE improves** (best validation performance)
- `epoch_N.pt` - Periodic checkpoints every 50 epochs
- `final.pt` - Final model after training

**Velocity Normalization:**
Training uses z-score normalization so all axes (vx, vy, vz) contribute equally to the loss. The normalization stats (`vel_mean`, `vel_std`) are saved in the checkpoint for inference.

### 3. Evaluation

Test auto-regressive inference:

```bash
# Evaluate on specific map and waypoints
python training/vel_net/train_vel_net.py eval \
      --checkpoint checkpoints/vel_net/best.pt \
      --map gate_mid \
      --waypoints gate_mid \
      --v_avg 1.0 \
      --output_dir output/vel_net_eval

# Evaluate with different waypoint trajectory
python training/vel_net/train_vel_net.py eval \
      --checkpoint checkpoints/vel_net/best.pt \
      --map gate_mid \
      --waypoints zigzag \
      --v_avg 1.5 \
      --output_dir output/vel_net_eval
```

### 4. Verification on Collected Data

Test vel_net on collected simulation or real-world data:

#### Simulation Data Verification

```bash
# Verify on a single simulation sequence
python test/verify_sim_data.py \
    --data_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata/seq_0000 \
    --checkpoint /scr/irislab/ke/checkpoints/vel_net_imu_fusion \
    --output_dir output/sim_data_verify

# Verify on different sequence
python test/verify_sim_data.py \
    --data_dir /scr/irislab/ke/data/vel_net/gate_mid_new_veldata/seq_0010 \
    --checkpoint /scr/irislab/ke/checkpoints/vel_net_imu_fusion \
    --output_dir output/sim_verify_seq10
```

#### Real-World Data Verification

```bash
# Verify on real-world flight data
python test/verify_real_world.py \
    --data_dir test/real_world_vel_data/2026-01-26_02-00-29 \
    --checkpoint /scr/irislab/ke/checkpoints/vel_net_imu_fusion \
    --output_dir output/real_world_verify
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Path to data sequence directory | required |
| `--checkpoint` | Checkpoint file or directory | required |
| `--output_dir` | Output directory for plots | `output/*_verify` |
| `--device` | PyTorch device | `cuda:0` |

**Output Files:**
| File | Description |
|------|-------------|
| `velocity_comparison.png` | 4-panel plot (X, Y, Z, magnitude) comparing GT vs predicted |
| `error_over_time.png` | Per-axis error evolution over time |
| `error_distribution.png` | Error histograms and box plots |
| `trajectory_3d.png` | 3D trajectory with error coloring |
| `metrics.txt` | Summary metrics (MAE, RMSE, per-axis) |
| `results.npz` | Raw data for further analysis |

**Data Format Differences:**

| Format | Sim Data | Real-World Data |
|--------|----------|-----------------|
| Telemetry | `telemetry.npz` | `fast_state_record.txt`, `fast_action_record.txt` |
| RGB | `000000.png` | `0001.jpg` |
| Depth | `000000.npy` | `0000_depth.npy` |
| Action order | [roll, pitch, yaw, thrust] | [thrust, roll, pitch, yaw] (auto-reordered) |

---

## Velocity Network Architecture

### Observation Structure (84 dims)

| Index | Component | Dims | Description |
|-------|-----------|------|-------------|
| 0-5 | Rot6D | 6 | Rotation (first 2 cols of rotation matrix) |
| 6-9 | Action | 4 | Current [roll_rate, pitch_rate, yaw_rate, thrust] |
| 10-13 | Prev Action | 4 | Previous action |
| 14-16 | Prev Velocity | 3 | Auto-regressive term (GT for training, predicted for inference) |
| 17-48 | RGB Features | 32 | MobileNetV3 encoder output |
| 49-80 | Depth Features | 32 | MobileNetV3 encoder output |
| 81-83 | IMU Accel | 3 | Linear acceleration from IMU |

### Model Architecture

```
Input (84 dims)
    ↓
LayerNorm
    ↓
Projector MLP (84 → 256)
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
from models.vel_net import VELO_NET, DualEncoder
from models.vel_net.vel_obs_utils import quaternion_to_rot6d

# Create model and encoder
model = VELO_NET(num_obs=84, stack_size=1, device='cuda:0')
encoder = DualEncoder(rgb_dim=32, depth_dim=32)

# Encode images
rgb_feat, depth_feat = encoder(rgb_image, depth_image)

# Normalize inputs (using stats from checkpoint)
prev_vel_norm = (prev_vel - vel_mean) / vel_std
imu_accel_norm = (imu_accel - accel_mean) / accel_std

# Build observation (84 dims)
rot6d = quaternion_to_rot6d(quaternion)  # (B, 6)
obs = torch.cat([
    rot6d,                    # (B, 6)
    action,                   # (B, 4)
    prev_action,              # (B, 4)
    prev_vel_norm,            # (B, 3) - normalized
    rgb_feat,                 # (B, 32)
    depth_feat,               # (B, 32)
    imu_accel_norm,           # (B, 3) - normalized
], dim=1)

# Predict velocity (direct delta-v mode)
delta_v_norm, _ = model.encode_step(obs)
delta_v = delta_v_norm * delta_std + delta_mean
velocity = prev_vel + delta_v  # No physics integration
```

---

## Project Structure

```
GRaD_Dynamic_onboard/
├── models/
│   └── vel_net/                      # Velocity estimation network
│       ├── vel_net.py                # VELO_NET model (84-dim input)
│       ├── visual_encoder.py         # MobileNetV3 DualEncoder
│       ├── vel_obs_buffer.py         # History buffer
│       └── vel_obs_utils.py          # Rot6D, observation builders
│
├── training/
│   └── vel_net/                      # Vel_net training pipeline
│       ├── data_collector.py         # Flight data collection
│       ├── dataset.py                # PyTorch Dataset
│       ├── trainer.py                # Curriculum learning trainer
│       ├── evaluator.py              # Evaluation utilities
│       └── train_vel_net.py          # Main entry point
│
├── test/
│   ├── verify_sim_data.py            # Verify vel_net on simulation data
│   ├── verify_real_world.py          # Verify vel_net on real-world data
│   └── real_world_vel_data/          # Real-world flight recordings
│
├── envs/
│   ├── drone_env.py                  # SimpleDroneEnv (base environment)
│   └── assets/
│       ├── quadrotor_dynamics.py     # Drone dynamics simulation
│       └── gs_data/                  # Gaussian splatting scene data
│
├── controller/
│   ├── waypoint_nav_geometric.py     # Main navigation script
│   ├── nav_helpers.py                # Helper functions
│   └── geometric_controller.py       # SE(3) controller
│
├── trajectory/
│   ├── bspline_trajectory.py         # B-spline trajectory generation
│   └── min_snap.py                   # Min-snap trajectory optimization
│
├── utils/
│   ├── traj_planner_global.py        # A* path planner
│   ├── gs_local.py                   # Gaussian splatting renderer
│   ├── point_cloud_util.py           # Point cloud utilities
│   └── rotation.py                   # Rotation utilities
│
└── checkpoints/
    └── vel_net_imu_fusion/           # Pretrained vel_net (84-dim)
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

## Map/Trajectory Configurations

### GS/Point Cloud Maps (`--map`)

| Map | Folder | Description |
|-----|--------|-------------|
| `gate_mid` | gate_mid_new | Main gate scene (center view) |
| `gate_left` | sv_917_3_left_nerfstudio | Gate scene (left view) |
| `gate_right` | sv_917_3_right_nerfstudio | Gate scene (right view) |
| `clutter` | sv_712_nerfstudio | Cluttered environment |
| `backroom` | sv_1018_2 | Backroom scene |
| `flightroom` | sv_1018_3 | Flight room scene |

### Waypoint Trajectories (`--waypoints`)

| Trajectory | Default Map | Start (PC space) | Waypoints | Destination | Description |
|------------|-------------|------------------|-----------|-------------|-------------|
| `gate_mid` | gate_mid | [-6, 0, 1.2] | 4 | [7.5, -2, 1.2] | Default center trajectory |
| `gate_mid_high` | gate_mid | [-6, 0, 1.6] | 4 | [7.5, -2, 1.6] | Higher altitude |
| `gate_mid_low` | gate_mid | [-6, 0, 0.8] | 4 | [7.5, -2, 0.8] | Lower altitude |
| `gate_left` | gate_left | [-6, 0, 1.2] | 3 | [7, -2, 1.2] | Left side of gate |
| `gate_right` | gate_mid | [-6, 0, 1.3] | 5 | [7, -2, 1.3] | Right side of gate |
| `zigzag` | gate_mid | [-6, 0, 1.2] | 6 | [7.5, -2, 1.2] | Lateral zigzag |
| `straight` | gate_mid | [-6, 0, 1.2] | 4 | [7.5, 0, 1.2] | Straight line |
| `reverse` | gate_mid | [7.5, -2, 1.2] | 4 | [-6, 0, 1.2] | Reverse of gate_mid |

**Note:** If `--map` is not specified, the default map from the waypoint config is used.


# GRaD Nav policy
## To train a gradnav policy:
```
python examples/train_gradnav.py     --cfg examples/cfg/gradnav/drone_test.yaml     --logdir checkpoints/gradnav_test     --device cuda:0

```

## To eval a gradnav policy:
```
 python examples/train_gradnav.py --cfg examples/cfg/gradnav/drone_test.yaml --checkpoint /home/irislab/ke/GRaD_Dynamic_onboard/checkpoints/gradnav_test_0212/gate_mid_new/02-12-2026-17-28-39/best_policy.pt  --play --render
 ```

 ## To continue training a gradnav policy (with bug potential):
 ```
 python examples/train_gradnav.py \
      --cfg examples/cfg/gradnav/drone_test.yaml \
      --logdir checkpoints/grad_nav_velnet_finetune \
      --checkpoint /home/irislab/ke/GRaD_Dynamic_onboard/checkpoints/grad_nav_velnet_finetune/gate_mid/01-12-2026-12-59-50/best_policy.pt \
      --device cuda:0
```
## To train with dynamic cylinder obstacle
```
python examples/train_gradnav_dynamic.py \
      --cfg examples/cfg/gradnav/drone_dynamic_cylinder.yaml \
      --logdir checkpoints/gradnav_dynamic_cylinder \
      --device cuda:0
```
## To eval
```
python examples/train_gradnav_dynamic.py \
      --cfg examples/cfg/gradnav/drone_dynamic_sphere.yaml \
      --checkpoint /home/irislab/ke/GRaD_Dynamic_onboard/checkpoints/dynamic_gradnav_cylinder_test_0115/gate_mid/01-15-2026-02-52-33/best_policy.pt \
      --play --render
```

## To train with curriculum learning (danger-aware retreat)
2-phase curriculum: Phase 1 (static) → Phase 2 (dynamic with danger-aware rewards)
```bash
python examples/train_gradnav_dynamic.py \
      --cfg examples/cfg/gradnav/drone_dynamic_curriculum.yaml \
      --logdir checkpoints/gradnav_curriculum \
      --device cuda:0
```

**Curriculum Schedule (configurable in YAML):**
- iter 0-200: Phase 1 (static only, original reward)
- iter 200-300: Phase 2 with 30% dynamic obstacle spawn
- iter 300-400: Phase 2 with 70% dynamic obstacle spawn
- iter 400+: Phase 2 with 100% dynamic obstacle spawn

**Phase 2 adds danger-aware rewards:**
- Retreat reward: encourages increasing distance when obstacle approaching
- No-forward penalty: discourages forward motion when in danger
- Backward penalty: small regularization to prevent always reversing

## To eval curriculum-trained policy
```bash
python examples/train_gradnav_dynamic.py \
      --cfg examples/cfg/gradnav/drone_dynamic_curriculum.yaml \
      --checkpoint /home/irislab/ke/GRaD_Dynamic_onboard/checkpoints/gradnav_dynamic_0216_using_vel_cuda_1_v1/gate_mid_new/02-16-2026-15-45-01/best_policy.pt \
      --play --render
```

---

# Dynamic Obstacle Support

## Overview

The system supports dynamic obstacles (sphere, box, cylinder) that can be injected into depth and RGB images via ray-casting. Objects can follow predefined trajectories or move with various patterns.

## Quick Test

```bash
# Test with trajectory navigation (0.3 m/s)
python controller/test_dynamic_obstacle.py --output output/dynamic_test.mp4 --max_steps 3000

# Test with shake motion
python controller/test_dynamic_obstacle.py --output output/shake_test.mp4 --max_steps 1000
```

## Components

### 1. DynamicObjectManager
Manages batched dynamic objects across multiple environments.

```python
from envs.dynamic_utils import DynamicObjectManager, TrajectoryPattern

manager = DynamicObjectManager(num_envs=1, device='cuda:0', max_objects_per_env=5)

# Spawn sphere with trajectory
pattern = TrajectoryPattern(trajectory_file='envs/assets/trajectories/human.csv', loop=True, device='cuda:0')
manager.spawn_object(env_id=0, position=pattern.get_position(0.0),
                     velocity=torch.zeros(3), radius=0.5, pattern=pattern, obj_type='sphere')

# Update positions each step
manager.update(dt=0.02)
```

### 2. DepthAugmentor
Ray-casting based depth/RGB injection with proper occlusion handling.

```python
from envs.dynamic_utils import DepthAugmentor

camera_params = {'fx': 128.0, 'fy': 128.0, 'cx': 128.0, 'cy': 72.0, 'width': 256, 'height': 144}
augmentor = DepthAugmentor(camera_params=camera_params, device='cuda:0')

# Inject objects into images
augmented_depth, augmented_rgb = augmentor.inject_objects_with_rgb(
    depth_maps=depth,           # (B, H, W, 1)
    rgb_images=rgb,             # (B, H, W, 3)
    T_world_to_camera=T_matrix, # (B, 4, 4) transformation matrix
    dynamic_manager=manager,
    use_shading=True
)
```

### 3. Movement Patterns

| Pattern | Description |
|---------|-------------|
| `LinearPattern` | Move in straight line with velocity |
| `CircularPattern` | Circular motion around center |
| `SinusoidalPattern` | Oscillating motion along axis |
| `RandomWalkPattern` | Random direction changes |
| `TrajectoryPattern` | Follow CSV trajectory file |

### 4. DynamicDroneEnv
Environment with built-in dynamic obstacle support.

```yaml
# examples/cfg/gradnav/drone_dynamic_test.yaml
params:
  diff_env:
    name: DynamicDroneEnv

  dynamic_objects:
    enabled: true
    max_objects_per_env: 1
    collision_threshold: 0.3
    camera_params:
      fx: 320.0
      fy: 320.0
      cx: 320.0
      cy: 180.0
      width: 640
      height: 360
    objects:
      - type: sphere
        radius: 0.5
        trajectory: "envs/assets/trajectories/human.csv"
        loop: true
```

## Coordinate System Notes

**Important:** GS coordinate system has Y and Z negated vs world frame.

```python
# Before injection, flip sphere positions to GS frame
original_positions = manager.positions.clone()
manager.positions[:, :, 1] = -original_positions[:, :, 1]  # Flip Y
manager.positions[:, :, 2] = -original_positions[:, :, 2]  # Flip Z

# Inject objects
augmented_depth, augmented_rgb = augmentor.inject_objects_with_rgb(...)

# Restore original positions
manager.positions = original_positions
```

## Camera Transform

Use `T_world_to_camera` 4x4 matrix for correct coordinate handling:

```python
def get_T_world_to_camera(pos, quat_xyzw, device):
    # Apply GS coordinate flip to position
    gs_pos = pos.cpu().numpy().copy()
    gs_pos[1] = -gs_pos[1]
    gs_pos[2] = -gs_pos[2]

    # Build rotation matrix with GS flip
    R_world_to_drone = Rotation.from_quat(quat_xyzw).as_matrix()
    flip_matrix = np.diag([1, -1, -1])
    R_world_to_drone_gs = R_world_to_drone @ flip_matrix

    # Drone to camera rotation
    R_drone_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    R_world_to_camera = R_world_to_drone_gs @ R_drone_to_camera

    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R_world_to_camera.T
    T[:3, 3] = -T[:3, :3] @ gs_pos

    return torch.tensor(T, device=device, dtype=torch.float32).unsqueeze(0)
```

## Supported Object Types

| Type | Parameters |
|------|------------|
| `sphere` | `radius` |
| `box` | `box_size` (3D), `box_rotation` (quaternion) |
| `cylinder` | `cylinder_radius`, `cylinder_height`, `cylinder_axis` (0/1/2) |