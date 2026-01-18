# GRaD_Dynamic_onboard

Drone waypoint navigation system using A* path planning, B-spline trajectory generation, SE(3) geometric controller, and velocity estimation network.

## Overview

The system performs:
1. **A* Path Planning** - Finds collision-free paths through point cloud obstacles
2. **B-Spline Trajectory Generation** - Creates smooth, flyable trajectories with velocity/acceleration profiles
3. **Geometric Controller** - SE(3) controller for accurate trajectory tracking
4. **Velocity Network** - Auto-regressive GRU network for velocity estimation (84-dim input with IMU fusion)

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
      --data_dir data/vel_net/sequences_0106 \
      --epochs 500 \
      --batch_size 16 \
      --seq_length 32 --stride 16 \
      --tf_start_epoch 10 --tf_end_epoch 60 \
      --wandb --checkpoint_dir checkpoints/vel_net_0106_norm_v2
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
python training/vel_net/train_vel_net.py eval \
      --checkpoint checkpoints/vel_net/best.pt \
      --map gate_mid \
      --v_avg 1.0 \
      --output_dir output/vel_net_eval
```

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

# Build observation (84 dims)
rot6d = quaternion_to_rot6d(quaternion)  # (B, 6)
obs = torch.cat([
    rot6d,                    # (B, 6)
    action,                   # (B, 4)
    prev_action,              # (B, 4)
    prev_vel,                 # (B, 3) - normalized
    rgb_feat,                 # (B, 32)
    depth_feat,               # (B, 32)
    imu_accel,                # (B, 3)
], dim=1)

# Predict velocity correction (physics-informed)
correction_norm, _ = model.forward(obs)
correction = correction_norm * delta_std + delta_mean
velocity = prev_vel + imu_accel * dt + correction
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

## Map Configurations

| Map | Start (PC space) | Waypoints | Destination |
|-----|------------------|-----------|-------------|
| `gate_mid` | [-6, 0, 1.2] | 4 waypoints | [7.5, -2, 1.2] |
| `gate_left` | [-6, 0, 1.2] | 3 waypoints | [7, -2, 1.2] |
| `gate_right` | [-6, 0, 1.3] | 5 waypoints | [7, -2, 1.3] |


# GRaD Nav policy
## To train a gradnav policy:
```
python examples/train_gradnav.py     --cfg examples/cfg/gradnav/drone_test.yaml     --logdir checkpoints/gradnav_test     --device cuda:0

```

## To eval a gradnav policy:
```
 python examples/train_gradnav.py --cfg examples/cfg/gradnav/drone_test.yaml --checkpoint /home/irislab/ke/GRaD_Dynamic_onboard/examples/logs/gradnav_test/gate_mid/gradnav_migration_test2/best_policy.pt  --play --render
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