# GRaD_Dynamic_onboard

Drone waypoint navigation system using A* path planning, B-spline trajectory generation, and SE(3) geometric controller.

## Overview

The system performs:
1. **A* Path Planning** - Finds collision-free paths through point cloud obstacles
2. **B-Spline Trajectory Generation** - Creates smooth, flyable trajectories with velocity/acceleration profiles
3. **Geometric Controller** - SE(3) controller for accurate trajectory tracking

## Quick Start

### Run Full Simulation (Path Planning + Flying + Video)

```bash
python controller/waypoint_nav_geometric.py --map gate_mid --corner_smoothing 0.018 --v_avg 0.5 --save_3d_plot
```

This will:
- Plan A* path through the `gate_mid` map
- Generate B-spline trajectory with corner smoothing
- Fly the drone at 0.5 m/s average velocity
- Save simulation video and 3D trajectory plot

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--map` | Map name (`gate_mid`, `gate_left`, `gate_right`, `simple_hover`) | `gate_mid` |
| `--v_avg` | Average flight velocity (m/s) | `1.5` |
| `--corner_smoothing` | B-spline corner smoothing (0=sharp, 0.5+=smooth) | `0.1` |
| `--output` | Output video path | `output/geometric_<map>_<timestamp>.mp4` |
| `--max_steps` | Maximum simulation steps | `3000` |
| `--fps` | Output video FPS | `25` |

### Output Options

| Option | Description |
|--------|-------------|
| `--save_plot` | Save trajectory comparison plot |
| `--save_3d_plot` | Save 3D trajectory plot with XY, XZ views and error |
| `--save_traj_profile` | Save trajectory profile (position/velocity/acceleration) |
| `--save_traj_data` | Save trajectory data to `.npz` file |
| `--traj_only` | Only generate trajectory (no simulation) |
| `--visualize_path` | Visualize A* path in Open3D before flying |

## Example Commands

### Generate trajectory only (no flying)
```bash
python controller/waypoint_nav_geometric.py --map gate_mid --traj_only
```
Outputs: trajectory profile and top-down/side view plots

### Slow flight with 3D plot
```bash
python controller/waypoint_nav_geometric.py --map gate_mid --v_avg 0.5 --corner_smoothing 0.018 --save_3d_plot
```

### Fast flight with all outputs
```bash
python controller/waypoint_nav_geometric.py --map gate_mid --v_avg 1.5 --save_plot --save_3d_plot --save_traj_data
```

### Visualize A* path before flying
```bash
python controller/waypoint_nav_geometric.py --map gate_mid --visualize_path
```

## Map Configurations

Waypoints are defined in `controller/waypoint_nav_geometric.py`:

| Map | Start | Waypoints | Destination |
|-----|-------|-----------|-------------|
| `gate_mid` | [-6, 0, 1.2] | 4 waypoints | [7, -2, 1.2] |
| `gate_left` | [-6, 0, 1.2] | 3 waypoints | [7, -2, 1.2] |
| `gate_right` | [-6, 0, 1.3] | 5 waypoints | [7, -2, 1.3] |

## Output Files

Running with `--save_3d_plot --save_traj_data` generates:
- `*_.mp4` - Simulation video (drone camera view)
- `*_3d_trajectory.png` - 3D trajectory plot with multiple views
- `*_trajectory_data.npz` - Trajectory data (actual, desired, errors, times)

## Velocity Network (vel_net)

Auto-regressive GRU network for drone velocity estimation with PINN loss.

### Architecture
```
Input (49 dims) → LayerNorm → Projector MLP → GRU (3 layers) → Head MLP → Velocity (3D)
```

### Observation Structure (49 dims)
| Component | Dims | Description |
|-----------|------|-------------|
| Rot6D | 6 | Rotation (first 2 cols of rotation matrix) |
| Action | 4 | Current thrust/body rates |
| Prev Action | 4 | Previous thrust/body rates |
| Prev Velocity | 3 | Auto-regressive term |
| RGB Features | 16 | Visual encoder output |
| Depth Features | 16 | Depth encoder output |

### Usage
```python
from models import VELO_NET, VelObsHistBuffer, build_vel_observation

# Create model
model = VELO_NET(num_obs=49, stack_size=5, device='cuda:0')

# Training
loss = model.loss_fn(obs_history, vel_gt, thrust=thrust_history)

# Inference (auto-regressive)
predictions = model.inference(get_sensor_obs_fn, num_steps=100, batch_size=1)
```

---

## Project Structure

```
models/
  └── vel_net/                     # Velocity estimation network
      ├── vel_net.py               # VELO_NET model
      ├── vel_obs_buffer.py        # History buffer
      └── vel_obs_utils.py         # Rot6D, observation builders

controller/
  ├── waypoint_nav_geometric.py    # Main navigation script
  ├── nav_helpers.py               # Helper functions
  └── geometric_controller.py      # SE(3) controller

trajectory/
  └── bspline_trajectory.py        # B-spline trajectory generation

utils/
  └── traj_planner_global.py       # A* path planner
```
