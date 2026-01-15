"""
Ray-casting based depth augmentation for dynamic objects (sphere, box, cylinder)
with proper occlusion handling
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

from .dynamic_objects import DynamicObjectManager


def transform_point_with_matrix(world_pos: torch.Tensor, T_world_to_camera: torch.Tensor):
    """
    Transform a point from world frame to camera frame using 4x4 transformation matrix

    Args:
        world_pos: (3,) or (N, 3) positions in world frame
        T_world_to_camera: (4, 4) transformation matrix

    Returns:
        camera_pos: Positions in camera frame
    """
    if world_pos.dim() == 1:
        world_pos = world_pos.unsqueeze(0)

    # Convert to homogeneous coordinates
    ones = torch.ones(world_pos.shape[0], 1, device=world_pos.device, dtype=world_pos.dtype)
    world_pos_homo = torch.cat([world_pos, ones], dim=1)

    # Apply transformation: p_cam = T @ p_world
    camera_pos_homo = (T_world_to_camera @ world_pos_homo.T).T

    # Extract 3D position
    camera_pos = camera_pos_homo[:, :3]

    # Remove batch dimension if input was unbatched
    if camera_pos.shape[0] == 1:
        camera_pos = camera_pos.squeeze(0)

    return camera_pos


def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x, y, z, w) to rotation matrix (3x3).
    Returns R_wc: rotation from camera frame to world frame.
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)

    x, y, z, w = q.unbind(-1)

    norm = torch.sqrt(x*x + y*y + z*z + w*w + 1e-8)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)

    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)

    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)

    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)

    return R[0] if R.shape[0] == 1 else R


def world_to_camera_transform(world_pos: torch.Tensor,
                              camera_pos: torch.Tensor,
                              camera_quat: torch.Tensor) -> torch.Tensor:
    """
    Convert world coordinates to camera coordinates.

    Args:
        world_pos: (3,) or (N,3) points in world frame
        camera_pos: (3,) camera origin in world frame
        camera_quat: (4,) camera orientation (x,y,z,w), world-frame

    Returns:
        (N, 3) points expressed in camera frame
    """
    if world_pos.ndim == 1:
        world_pos = world_pos.unsqueeze(0)

    R_wc = quat_to_rotmat_xyzw(camera_quat)
    rel = world_pos - camera_pos
    cam_pts = rel @ R_wc

    return cam_pts if cam_pts.shape[0] > 1 else cam_pts[0]


def ray_sphere_intersection(ray_origin: torch.Tensor,
                           ray_dir: torch.Tensor,
                           sphere_center: torch.Tensor,
                           sphere_radius: float):
    """
    Calculate ray-sphere intersection using quadratic formula

    Args:
        ray_origin: (3,) ray start point in camera frame
        ray_dir: (H*W, 3) normalized ray directions
        sphere_center: (3,) sphere center in camera frame
        sphere_radius: sphere radius

    Returns:
        t_values: (H*W,) distance to intersection (-1 if no intersection)
    """
    oc = ray_origin - sphere_center
    a = torch.sum(ray_dir * ray_dir, dim=1)
    b = 2.0 * torch.sum(ray_dir * oc, dim=1)
    c = torch.sum(oc * oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c
    t_values = torch.full((ray_dir.shape[0],), -1.0, device=ray_dir.device)

    valid_mask = discriminant >= 0

    if valid_mask.any():
        sqrt_disc = torch.sqrt(discriminant[valid_mask])
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]

        t1 = (-b_valid - sqrt_disc) / (2.0 * a_valid)
        t2 = (-b_valid + sqrt_disc) / (2.0 * a_valid)

        t_result = torch.where(t1 > 0, t1,
                              torch.where(t2 > 0, t2, torch.tensor(-1.0, device=ray_dir.device)))

        t_values[valid_mask] = t_result

    return t_values


class DepthAugmentor:
    """Ray-casting based depth augmentation for dynamic objects with occlusion handling"""

    def __init__(self, camera_params: dict, device: str = 'cuda'):
        """
        Args:
            camera_params: Dict with fx, fy, cx, cy, width, height
        """
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params['cx']
        self.cy = camera_params['cy']
        self.width = camera_params['width']
        self.height = camera_params['height']
        self.device = device

        self.ray_dirs = self._precompute_ray_directions()

        self.default_sphere_colors = [
            torch.tensor([1.0, 0.2, 0.2], device=device),  # Red
            torch.tensor([0.2, 1.0, 0.2], device=device),  # Green
            torch.tensor([0.2, 0.2, 1.0], device=device),  # Blue
            torch.tensor([1.0, 1.0, 0.2], device=device),  # Yellow
            torch.tensor([1.0, 0.5, 0.0], device=device),  # Orange
            torch.tensor([0.5, 0.2, 0.8], device=device),  # Purple
        ]

        self.default_light_dir = torch.tensor([0.5, -0.3, 0.7], device=device)
        self.default_light_dir = F.normalize(self.default_light_dir, dim=0)

    def _precompute_ray_directions(self):
        """Pre-compute normalized ray directions for all pixels"""
        H, W = self.height, self.width

        u = torch.arange(W, dtype=torch.float32, device=self.device)
        v = torch.arange(H, dtype=torch.float32, device=self.device)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')

        x = (u_grid - self.cx) / self.fx
        y = (v_grid - self.cy) / self.fy
        z = torch.ones_like(u_grid)

        rays = torch.stack([x, y, z], dim=-1)
        rays_flat = rays.reshape(-1, 3)
        rays_flat = rays_flat / torch.norm(rays_flat, dim=1, keepdim=True)

        return rays_flat

    def inject_sphere(self,
                     depth_map: torch.Tensor,
                     sphere_pos_world: torch.Tensor,
                     sphere_radius: float,
                     camera_pos: torch.Tensor = None,
                     camera_quat: torch.Tensor = None,
                     T_world_to_camera: torch.Tensor = None,
                     max_depth: float = 10.0):
        """
        Inject a single sphere into depth map using ray-casting with occlusion handling

        Args:
            depth_map: (H, W) current depth map from 3D-GS
            sphere_pos_world: (3,) sphere position in world frame
            sphere_radius: sphere radius
            camera_pos: (3,) camera position in world
            camera_quat: (4,) camera quaternion (x,y,z,w)
            max_depth: maximum depth to consider

        Returns:
            Modified depth map with sphere properly occluded
        """
        H, W = depth_map.shape

        depth_map = depth_map.to(self.device)
        sphere_pos_world = sphere_pos_world.to(self.device)

        # Support both camera_pos/quat and T_world_to_camera
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(self.device)
            sphere_pos_cam = transform_point_with_matrix(sphere_pos_world, T_world_to_camera)
        else:
            camera_pos = camera_pos.to(self.device)
            camera_quat = camera_quat.to(self.device)
            sphere_pos_cam = world_to_camera_transform(
                sphere_pos_world.unsqueeze(0) if sphere_pos_world.dim() == 1 else sphere_pos_world,
                camera_pos,
                camera_quat
            )

        if sphere_pos_cam.dim() > 1:
            sphere_pos_cam = sphere_pos_cam.squeeze()

        if sphere_pos_cam[2] < 0.1 or sphere_pos_cam[2] > max_depth:
            return depth_map

        ray_origin = torch.zeros(3, device=self.device)

        intersections = ray_sphere_intersection(
            ray_origin,
            self.ray_dirs,
            sphere_pos_cam,
            sphere_radius
        )

        intersection_depths = intersections.reshape(H, W)
        valid_mask = (intersection_depths > 0) & (intersection_depths < depth_map)
        depth_map[valid_mask] = intersection_depths[valid_mask]

        return depth_map

    def inject_box(self,
                   depth_map: torch.Tensor,
                   box_position: torch.Tensor,
                   box_size: torch.Tensor,
                   box_rotation: torch.Tensor = None,
                   camera_position: torch.Tensor = None,
                   camera_quaternion: torch.Tensor = None,
                   T_world_to_camera: torch.Tensor = None,
                   max_depth: float = 15.0) -> torch.Tensor:
        """Inject a box into the depth map with proper occlusion handling"""
        H, W = depth_map.shape
        device = depth_map.device

        # Support both camera_pos/quat and T_world_to_camera
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(device)
            box_pos_cam = transform_point_with_matrix(box_position, T_world_to_camera)
        else:
            box_pos_cam = world_to_camera_transform(
                box_position.unsqueeze(0) if box_position.dim() == 1 else box_position,
                camera_position,
                camera_quaternion
            )
        if box_pos_cam.dim() > 1:
            box_pos_cam = box_pos_cam.squeeze(0)

        if box_rotation is not None and not torch.allclose(box_rotation, torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)):
            x, y, z, w = box_rotation
            rot_matrix = torch.tensor([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], device=device, dtype=torch.float32)
            use_rotation = True
        else:
            rot_matrix = torch.eye(3, device=device, dtype=torch.float32)
            use_rotation = False

        half_size = box_size / 2.0
        focal = self.fx
        cx, cy = W / 2.0, H / 2.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        rays_x = (x_coords - cx) / focal
        rays_y = (y_coords - cy) / focal
        rays_z = torch.ones_like(rays_x)

        ray_dirs = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        if use_rotation:
            rot_matrix_inv = rot_matrix.T
            cam_to_box = -box_pos_cam
            cam_to_box_local = torch.matmul(rot_matrix_inv, cam_to_box)

            ray_dirs_flat = ray_dirs.reshape(-1, 3)
            ray_dirs_local = torch.matmul(ray_dirs_flat, rot_matrix_inv.T)
            ray_dirs_local = ray_dirs_local.reshape(H, W, 3)
        else:
            ray_dirs_local = ray_dirs
            cam_to_box_local = -box_pos_cam

        t_min = torch.full((H, W), -float('inf'), device=device)
        t_max = torch.full((H, W), float('inf'), device=device)

        for axis in range(3):
            ray_dir_axis = ray_dirs_local[..., axis]
            valid_rays = torch.abs(ray_dir_axis) > 1e-3

            ray_dir_safe = torch.where(valid_rays, ray_dir_axis, torch.ones_like(ray_dir_axis))
            t1 = (-half_size[axis] - cam_to_box_local[axis]) / ray_dir_safe
            t2 = (half_size[axis] - cam_to_box_local[axis]) / ray_dir_safe

            t_axis_min = torch.minimum(t1, t2)
            t_axis_max = torch.maximum(t1, t2)

            t_min = torch.where(valid_rays, torch.maximum(t_min, t_axis_min), t_min)
            t_max = torch.where(valid_rays, torch.minimum(t_max, t_axis_max), t_max)

        valid_intersections = (t_min <= t_max) & (t_max > 0) & (t_min < max_depth)
        intersection_depths = torch.where(t_min > 0, t_min, t_max)
        intersection_depths = torch.where(valid_intersections, intersection_depths, max_depth)

        valid_mask = valid_intersections & (intersection_depths < depth_map)
        depth_map[valid_mask] = intersection_depths[valid_mask]

        return depth_map

    def inject_cylinder(self,
                       depth_map: torch.Tensor,
                       cylinder_position: torch.Tensor,
                       cylinder_radius: float,
                       cylinder_height: float,
                       cylinder_axis: int = 2,
                       cylinder_rotation: torch.Tensor = None,
                       camera_position: torch.Tensor = None,
                       camera_quaternion: torch.Tensor = None,
                       T_world_to_camera: torch.Tensor = None,
                       max_depth: float = 15.0) -> torch.Tensor:
        """Inject a cylinder into the depth map with proper occlusion handling"""
        H, W = depth_map.shape
        device = depth_map.device

        # Support both camera_pos/quat and T_world_to_camera
        # Use detach() to prevent gradient tracking and memory buildup
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(device).detach()
            cyl_pos_cam = transform_point_with_matrix(cylinder_position.detach(), T_world_to_camera)
            # Extract rotation matrix from T_world_to_camera for cylinder rotation transform
            R_world_to_cam = T_world_to_camera[:3, :3]
        else:
            cyl_pos_cam = world_to_camera_transform(
                cylinder_position.detach().unsqueeze(0) if cylinder_position.dim() == 1 else cylinder_position.detach(),
                camera_position.detach(),
                camera_quaternion.detach()
            )
            # Get rotation matrix from camera quaternion (R_wc is camera-to-world, we need inverse)
            R_cam_to_world = quat_to_rotmat_xyzw(camera_quaternion.detach())
            R_world_to_cam = R_cam_to_world.T

        if cyl_pos_cam.dim() > 1:
            cyl_pos_cam = cyl_pos_cam.squeeze(0)

        # Cylinder rotation defines its local orientation (e.g., standing upright)
        # Use it as-is - the position transformation already handles camera movement
        cyl_rot_detached = cylinder_rotation.detach() if cylinder_rotation is not None else None
        if cyl_rot_detached is not None:
            x, y, z, w = cyl_rot_detached
            rot_matrix = torch.tensor([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], device=device, dtype=torch.float32)
            # Transform cylinder rotation from world to camera frame
            # rot_matrix = R_cam_from_cyl_local (transforms cylinder local → camera)
            rot_matrix = (rot_matrix @ R_world_to_cam).T
            use_rotation = True
        else:
            rot_matrix = torch.eye(3, device=device, dtype=torch.float32)
            use_rotation = False

        focal = self.fx
        cx, cy = W / 2.0, H / 2.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        rays_x = (x_coords - cx) / focal
        rays_y = (y_coords - cy) / focal
        rays_z = torch.ones_like(rays_x)

        ray_dirs = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        if use_rotation:
            rot_matrix_inv = rot_matrix.T
            cam_to_cyl = -cyl_pos_cam
            cam_to_cyl_local = torch.matmul(rot_matrix_inv, cam_to_cyl)

            ray_dirs_flat = ray_dirs.reshape(-1, 3)
            ray_dirs_local = torch.matmul(ray_dirs_flat, rot_matrix_inv.T)
            ray_dirs_local = ray_dirs_local.reshape(H, W, 3)
        else:
            ray_dirs_local = ray_dirs
            cam_to_cyl_local = -cyl_pos_cam

        half_height = cylinder_height / 2.0

        axis_map = {
            0: [1, 2, 0],
            1: [2, 0, 1],
            2: [0, 1, 2],
        }
        perm = axis_map[cylinder_axis]

        ray_dirs_perm = ray_dirs_local[..., perm]
        cam_to_cyl_perm = cam_to_cyl_local[perm]

        ray_ox = cam_to_cyl_perm[0].expand(H, W)
        ray_oy = cam_to_cyl_perm[1].expand(H, W)
        ray_oz = cam_to_cyl_perm[2].expand(H, W)
        ray_dx = ray_dirs_perm[..., 0]
        ray_dy = ray_dirs_perm[..., 1]
        ray_dz = ray_dirs_perm[..., 2]

        a = ray_dx * ray_dx + ray_dy * ray_dy
        b = 2 * (ray_ox * ray_dx + ray_oy * ray_dy)
        c = ray_ox * ray_ox + ray_oy * ray_oy - cylinder_radius * cylinder_radius

        discriminant = b * b - 4 * a * c
        valid_cylinder = (discriminant >= 0) & (torch.abs(a) > 1e-6)

        sqrt_disc = torch.sqrt(torch.abs(discriminant))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        z1 = ray_oz + t1 * ray_dz
        z2 = ray_oz + t2 * ray_dz

        valid_t1 = valid_cylinder & (t1 > 0) & (torch.abs(z1) <= half_height)
        valid_t2 = valid_cylinder & (t2 > 0) & (torch.abs(z2) <= half_height)

        t_top = (half_height - ray_oz) / ray_dz
        x_top = ray_ox + t_top * ray_dx
        y_top = ray_oy + t_top * ray_dy
        valid_top = (torch.abs(ray_dz) > 1e-6) & (t_top > 0) & (x_top*x_top + y_top*y_top <= cylinder_radius*cylinder_radius)

        t_bottom = (-half_height - ray_oz) / ray_dz
        x_bottom = ray_ox + t_bottom * ray_dx
        y_bottom = ray_oy + t_bottom * ray_dy
        valid_bottom = (torch.abs(ray_dz) > 1e-6) & (t_bottom > 0) & (x_bottom*x_bottom + y_bottom*y_bottom <= cylinder_radius*cylinder_radius)

        intersection_depths = torch.full((H, W), max_depth, device=device)

        intersection_depths = torch.where(valid_t1, torch.minimum(intersection_depths, t1), intersection_depths)
        intersection_depths = torch.where(valid_t2, torch.minimum(intersection_depths, t2), intersection_depths)
        intersection_depths = torch.where(valid_top, torch.minimum(intersection_depths, t_top), intersection_depths)
        intersection_depths = torch.where(valid_bottom, torch.minimum(intersection_depths, t_bottom), intersection_depths)

        valid_mask = (intersection_depths < max_depth) & (intersection_depths < depth_map)
        depth_map[valid_mask] = intersection_depths[valid_mask]

        return depth_map

    def compute_sphere_normals(self,
                              ray_dirs: torch.Tensor,
                              t_values: torch.Tensor,
                              sphere_pos_cam: torch.Tensor,
                              sphere_radius: float):
        """Compute surface normals for sphere intersection points"""
        ray_origin = torch.zeros(3, device=self.device)
        hit_points = ray_origin + t_values.unsqueeze(-1) * ray_dirs
        normals = (hit_points - sphere_pos_cam) / sphere_radius
        normals = F.normalize(normals, dim=-1)
        return normals

    def apply_lambertian_shading(self,
                                base_color: torch.Tensor,
                                normals: torch.Tensor,
                                light_dir: Optional[torch.Tensor] = None,
                                ambient: float = 0.3):
        """Apply Lambertian (diffuse) shading to give 3D appearance"""
        if light_dir is None:
            light_dir = self.default_light_dir

        light_dir = F.normalize(light_dir, dim=-1)
        ndotl = torch.clamp(torch.sum(normals * light_dir, dim=-1), 0, 1)
        shading_factor = ambient + (1 - ambient) * ndotl

        if base_color.dim() == 1:
            base_color = base_color.unsqueeze(0)

        shaded_colors = base_color * shading_factor.unsqueeze(-1)
        return shaded_colors

    def inject_sphere_rgb(self,
                         rgb_image: torch.Tensor,
                         depth_map: torch.Tensor,
                         sphere_pos_world: torch.Tensor,
                         sphere_radius: float,
                         sphere_color: torch.Tensor,
                         camera_pos: torch.Tensor = None,
                         camera_quat: torch.Tensor = None,
                         T_world_to_camera: torch.Tensor = None,
                         use_shading: bool = True,
                         debug: bool = False):
        """Inject a colored sphere into RGB image with shading and occlusion"""
        H, W, C = rgb_image.shape

        rgb_image = rgb_image.to(self.device)
        depth_map = depth_map.to(self.device)
        sphere_pos_world = sphere_pos_world.to(self.device)
        sphere_color = sphere_color.to(self.device)

        rgb_image = rgb_image.clone()

        # Support both camera_pos/quat and T_world_to_camera
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(self.device)
            sphere_pos_cam = transform_point_with_matrix(sphere_pos_world, T_world_to_camera)
        else:
            if camera_pos is not None:
                camera_pos = camera_pos.to(self.device)
            if camera_quat is not None:
                camera_quat = camera_quat.to(self.device)
            sphere_pos_cam = world_to_camera_transform(
                sphere_pos_world.unsqueeze(0) if sphere_pos_world.dim() == 1 else sphere_pos_world,
                camera_pos,
                camera_quat
            )
        if sphere_pos_cam.dim() > 1:
            sphere_pos_cam = sphere_pos_cam.squeeze()

        if sphere_pos_cam[2] < 0.1:
            return rgb_image

        ray_origin = torch.zeros(3, device=self.device)
        t_values = ray_sphere_intersection(
            ray_origin,
            self.ray_dirs,
            sphere_pos_cam,
            sphere_radius
        )

        t_image = t_values.reshape(H, W)
        visible_mask = (t_image > 0) & (t_image < depth_map.to(t_image.device))

        if not visible_mask.any():
            return rgb_image

        visible_flat = visible_mask.reshape(-1)

        if use_shading:
            normals = self.compute_sphere_normals(
                self.ray_dirs,
                t_values,
                sphere_pos_cam,
                sphere_radius
            )

            shaded_colors = self.apply_lambertian_shading(
                sphere_color,
                normals[visible_flat],
                ambient=0.3
            )

            rgb_flat = rgb_image.reshape(-1, 3)
            rgb_flat[visible_flat] = shaded_colors
            rgb_image = rgb_flat.reshape(H, W, 3)
        else:
            rgb_image[visible_mask] = sphere_color

        return rgb_image

    def inject_box_rgb(self,
                       rgb_image: torch.Tensor,
                       depth_map: torch.Tensor,
                       box_position: torch.Tensor,
                       box_size: torch.Tensor,
                       box_rotation: torch.Tensor,
                       box_color: torch.Tensor,
                       camera_position: torch.Tensor = None,
                       camera_quaternion: torch.Tensor = None,
                       T_world_to_camera: torch.Tensor = None,
                       use_shading: bool = True,
                       debug: bool = False) -> torch.Tensor:
        """Inject box into RGB image with proper shading and occlusion"""
        H, W, C = rgb_image.shape
        device = rgb_image.device

        # Support both camera_pos/quat and T_world_to_camera
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(device)
            box_pos_cam = transform_point_with_matrix(box_position, T_world_to_camera)
        else:
            box_pos_cam = world_to_camera_transform(
                box_position.unsqueeze(0) if box_position.dim() == 1 else box_position,
                camera_position,
                camera_quaternion
            )
        if box_pos_cam.dim() > 1:
            box_pos_cam = box_pos_cam.squeeze(0)

        if box_rotation is not None and not torch.allclose(box_rotation, torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)):
            x, y, z, w = box_rotation
            rot_matrix = torch.tensor([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], device=device, dtype=torch.float32)
            use_rotation = True
        else:
            rot_matrix = torch.eye(3, device=device, dtype=torch.float32)
            use_rotation = False

        half_size = box_size / 2.0
        focal = self.fx
        cx, cy = W / 2.0, H / 2.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        rays_x = (x_coords - cx) / focal
        rays_y = (y_coords - cy) / focal
        rays_z = torch.ones_like(rays_x)

        ray_dirs = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        if use_rotation:
            rot_matrix_inv = rot_matrix.T
            cam_to_box = -box_pos_cam
            cam_to_box_local = torch.matmul(rot_matrix_inv, cam_to_box)

            ray_dirs_flat = ray_dirs.reshape(-1, 3)
            ray_dirs_local = torch.matmul(ray_dirs_flat, rot_matrix_inv.T)
            ray_dirs_local = ray_dirs_local.reshape(H, W, 3)
        else:
            ray_dirs_local = ray_dirs
            cam_to_box_local = -box_pos_cam

        t_min = torch.full((H, W), -float('inf'), device=device)
        t_max = torch.full((H, W), float('inf'), device=device)

        hit_face = torch.zeros((H, W), dtype=torch.int, device=device)

        for axis in range(3):
            ray_dir_axis = ray_dirs_local[..., axis]
            valid_rays = torch.abs(ray_dir_axis) > 1e-3

            ray_dir_safe = torch.where(valid_rays, ray_dir_axis, torch.ones_like(ray_dir_axis))
            t1 = (-half_size[axis] - cam_to_box_local[axis]) / ray_dir_safe
            t2 = (half_size[axis] - cam_to_box_local[axis]) / ray_dir_safe

            t_axis_min = torch.minimum(t1, t2)
            t_axis_max = torch.maximum(t1, t2)

            new_t_min = torch.maximum(t_min, t_axis_min)
            face_changed = valid_rays & (new_t_min > t_min)
            hit_face = torch.where(face_changed, axis * 2 + (t1 > t2).int(), hit_face)

            t_min = torch.where(valid_rays, new_t_min, t_min)
            t_max = torch.where(valid_rays, torch.minimum(t_max, t_axis_max), t_max)

        valid_intersections = (t_min <= t_max) & (t_max > 0)
        intersection_depths = torch.where(t_min > 0, t_min, t_max)

        visible_mask = valid_intersections & (intersection_depths <= depth_map + 0.01)

        if use_shading and visible_mask.any():
            normals = torch.zeros((H, W, 3), device=device)

            face_normals_local = torch.tensor([
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

            if use_rotation:
                face_normals = torch.matmul(face_normals_local, rot_matrix.T)
            else:
                face_normals = face_normals_local

            for face_idx in range(6):
                face_mask = visible_mask & (hit_face == face_idx)
                if face_mask.any():
                    normals[face_mask] = face_normals[face_idx]

            visible_flat = visible_mask.reshape(-1)
            normals_flat = normals.reshape(-1, 3)

            shaded_colors = self.apply_lambertian_shading(
                box_color,
                normals_flat[visible_flat],
                ambient=0.3
            )

            rgb_flat = rgb_image.reshape(-1, 3)
            rgb_flat[visible_flat] = shaded_colors
            rgb_image = rgb_flat.reshape(H, W, 3)
        else:
            rgb_image[visible_mask] = box_color

        return rgb_image

    def inject_cylinder_rgb(self,
                           rgb_image: torch.Tensor,
                           depth_map: torch.Tensor,
                           cylinder_position: torch.Tensor,
                           cylinder_radius: float,
                           cylinder_height: float,
                           cylinder_axis: int,
                           cylinder_rotation: torch.Tensor,
                           cylinder_color: torch.Tensor,
                           camera_position: torch.Tensor = None,
                           camera_quaternion: torch.Tensor = None,
                           T_world_to_camera: torch.Tensor = None,
                           use_shading: bool = True,
                           debug: bool = False) -> torch.Tensor:
        """Inject cylinder into RGB image with proper shading and occlusion"""
        H, W, C = rgb_image.shape
        device = rgb_image.device

        # Support both camera_pos/quat and T_world_to_camera
        # Use detach() to prevent gradient tracking and memory buildup
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(device).detach()
            cyl_pos_cam = transform_point_with_matrix(cylinder_position.detach(), T_world_to_camera)
            # Extract rotation matrix from T_world_to_camera for cylinder rotation transform
            R_world_to_cam = T_world_to_camera[:3, :3]
        else:
            cyl_pos_cam = world_to_camera_transform(
                cylinder_position.detach().unsqueeze(0) if cylinder_position.dim() == 1 else cylinder_position.detach(),
                camera_position.detach(),
                camera_quaternion.detach()
            )
            # Get rotation matrix from camera quaternion (R_wc is camera-to-world, we need inverse)
            R_cam_to_world = quat_to_rotmat_xyzw(camera_quaternion.detach())
            R_world_to_cam = R_cam_to_world.T

        if cyl_pos_cam.dim() > 1:
            cyl_pos_cam = cyl_pos_cam.squeeze(0)

        # Cylinder rotation defines its local orientation (e.g., standing upright)
        # Use it as-is - the position transformation already handles camera movement
        cyl_rot_detached = cylinder_rotation.detach() if cylinder_rotation is not None else None
        # if cyl_rot_detached is not None and not torch.allclose(cyl_rot_detached, torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)):
        if cyl_rot_detached is not None:
            x, y, z, w = cyl_rot_detached
            # x, y, z, w = 0, 0, 0, 1
            rot_matrix = torch.tensor([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], device=device, dtype=torch.float32)
            # Transform cylinder rotation from world to camera frame
            # rot_matrix = R_cam_from_cyl_local (transforms cylinder local → camera)
            rot_matrix = (rot_matrix @ R_world_to_cam).T

            use_rotation = True
        else:
            rot_matrix = torch.eye(3, device=device, dtype=torch.float32)
            use_rotation = False

        focal = self.fx
        cx, cy = W / 2.0, H / 2.0

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        rays_x = (x_coords - cx) / focal
        rays_y = (y_coords - cy) / focal
        rays_z = torch.ones_like(rays_x)

        ray_dirs = torch.stack([rays_x, rays_y, rays_z], dim=-1)
        ray_dirs = F.normalize(ray_dirs, dim=-1)

        if use_rotation:
            rot_matrix_inv = rot_matrix.T
            cam_to_cyl = -cyl_pos_cam
            cam_to_cyl_local = torch.matmul(rot_matrix_inv, cam_to_cyl)

            ray_dirs_flat = ray_dirs.reshape(-1, 3)
            ray_dirs_local = torch.matmul(ray_dirs_flat, rot_matrix_inv.T)
            ray_dirs_local = ray_dirs_local.reshape(H, W, 3)
        else:
            ray_dirs_local = ray_dirs
            cam_to_cyl_local = -cyl_pos_cam

        half_height = cylinder_height / 2.0

        axis_map = {
            0: [1, 2, 0],
            1: [2, 0, 1],
            2: [0, 1, 2],
        }
        perm = axis_map[cylinder_axis]

        ray_dirs_perm = ray_dirs_local[..., perm]
        cam_to_cyl_perm = cam_to_cyl_local[perm]

        ray_ox = cam_to_cyl_perm[0].expand(H, W)
        ray_oy = cam_to_cyl_perm[1].expand(H, W)
        ray_oz = cam_to_cyl_perm[2].expand(H, W)
        ray_dx = ray_dirs_perm[..., 0]
        ray_dy = ray_dirs_perm[..., 1]
        ray_dz = ray_dirs_perm[..., 2]

        a = ray_dx * ray_dx + ray_dy * ray_dy
        b = 2 * (ray_ox * ray_dx + ray_oy * ray_dy)
        c = ray_ox * ray_ox + ray_oy * ray_oy - cylinder_radius * cylinder_radius

        discriminant = b * b - 4 * a * c
        valid_cylinder = (discriminant >= 0) & (torch.abs(a) > 1e-6)

        sqrt_disc = torch.sqrt(torch.abs(discriminant))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        z1 = ray_oz + t1 * ray_dz
        z2 = ray_oz + t2 * ray_dz

        valid_t1 = valid_cylinder & (t1 > 0) & (torch.abs(z1) <= half_height)
        valid_t2 = valid_cylinder & (t2 > 0) & (torch.abs(z2) <= half_height)

        t_top = (half_height - ray_oz) / ray_dz
        x_top = ray_ox + t_top * ray_dx
        y_top = ray_oy + t_top * ray_dy
        valid_top = (torch.abs(ray_dz) > 1e-6) & (t_top > 0) & (x_top*x_top + y_top*y_top <= cylinder_radius*cylinder_radius)

        t_bottom = (-half_height - ray_oz) / ray_dz
        x_bottom = ray_ox + t_bottom * ray_dx
        y_bottom = ray_oy + t_bottom * ray_dy
        valid_bottom = (torch.abs(ray_dz) > 1e-6) & (t_bottom > 0) & (x_bottom*x_bottom + y_bottom*y_bottom <= cylinder_radius*cylinder_radius)

        intersection_depths = torch.full((H, W), float('inf'), device=device)
        hit_surface = torch.zeros((H, W), dtype=torch.int, device=device)

        mask_t1 = valid_t1 & (t1 < intersection_depths)
        intersection_depths = torch.where(mask_t1, t1, intersection_depths)
        hit_surface = torch.where(mask_t1, 1, hit_surface)

        mask_t2 = valid_t2 & (t2 < intersection_depths)
        intersection_depths = torch.where(mask_t2, t2, intersection_depths)
        hit_surface = torch.where(mask_t2, 1, hit_surface)

        mask_top = valid_top & (t_top < intersection_depths)
        intersection_depths = torch.where(mask_top, t_top, intersection_depths)
        hit_surface = torch.where(mask_top, 2, hit_surface)

        mask_bottom = valid_bottom & (t_bottom < intersection_depths)
        intersection_depths = torch.where(mask_bottom, t_bottom, intersection_depths)
        hit_surface = torch.where(mask_bottom, 3, hit_surface)

        visible_mask = (intersection_depths < float('inf')) & (intersection_depths <= depth_map + 0.01)

        if use_shading and visible_mask.any():
            normals = torch.zeros((H, W, 3), device=device)

            side_mask = visible_mask & (hit_surface == 1)
            if side_mask.any():
                t_side = intersection_depths[side_mask]
                x_side = ray_ox[side_mask] + t_side * ray_dx[side_mask]
                y_side = ray_oy[side_mask] + t_side * ray_dy[side_mask]

                norm_x = x_side / cylinder_radius
                norm_y = y_side / cylinder_radius
                norm_z = torch.zeros_like(norm_x)

                normals_perm = torch.stack([norm_x, norm_y, norm_z], dim=-1)
                inv_perm = [perm.index(i) for i in range(3)]
                normals_local = normals_perm[..., inv_perm]

                if use_rotation:
                    normals_cam = torch.matmul(normals_local, rot_matrix.T)
                else:
                    normals_cam = normals_local

                normals[side_mask] = normals_cam

            top_mask = visible_mask & (hit_surface == 2)
            if top_mask.any():
                cap_normal_local = torch.zeros(3, device=device)
                cap_normal_local[perm[2]] = 1.0

                if use_rotation:
                    cap_normal = torch.matmul(rot_matrix, cap_normal_local)
                else:
                    cap_normal = cap_normal_local

                normals[top_mask] = cap_normal

            bottom_mask = visible_mask & (hit_surface == 3)
            if bottom_mask.any():
                cap_normal_local = torch.zeros(3, device=device)
                cap_normal_local[perm[2]] = -1.0

                if use_rotation:
                    cap_normal = torch.matmul(rot_matrix, cap_normal_local)
                else:
                    cap_normal = cap_normal_local

                normals[bottom_mask] = cap_normal

            visible_flat = visible_mask.reshape(-1)
            normals_flat = normals.reshape(-1, 3)

            shaded_colors = self.apply_lambertian_shading(
                cylinder_color,
                normals_flat[visible_flat],
                ambient=0.3
            )

            rgb_flat = rgb_image.reshape(-1, 3)
            rgb_flat[visible_flat] = shaded_colors
            rgb_image = rgb_flat.reshape(H, W, 3)
        else:
            rgb_image[visible_mask] = cylinder_color

        return rgb_image

    def inject_objects_with_rgb(self,
                               depth_maps: torch.Tensor,
                               rgb_images: torch.Tensor,
                               camera_poses: torch.Tensor = None,
                               T_world_to_camera: torch.Tensor = None,
                               dynamic_manager: DynamicObjectManager = None,
                               use_shading: bool = True,
                               custom_colors: Optional[List[torch.Tensor]] = None):
        """
        Inject all dynamic objects into both depth and RGB images

        Args:
            depth_maps: (B, H, W, 1) original depth maps
            rgb_images: (B, H, W, 3) original RGB images
            camera_poses: (B, 7) camera poses
            T_world_to_camera: (B, 4, 4) transformation matrices (optional)
            dynamic_manager: DynamicObjectManager instance
            use_shading: whether to apply shading
            custom_colors: optional list of colors for objects

        Returns:
            Tuple of (augmented_depth, augmented_rgb)
        """
        B, H, W, _ = depth_maps.shape

        depth_maps = depth_maps.to(self.device)
        rgb_images = rgb_images.to(self.device)
        if camera_poses is not None:
            camera_poses = camera_poses.to(self.device)
        if T_world_to_camera is not None:
            T_world_to_camera = T_world_to_camera.to(self.device)

        augmented_depth = depth_maps.clone()
        augmented_rgb = rgb_images.clone()

        sphere_colors = custom_colors if custom_colors else self.default_sphere_colors

        for env_id in range(B):
            # Support both camera_poses and T_world_to_camera
            if T_world_to_camera is not None:
                T_matrix = T_world_to_camera[env_id]
                cam_pos = None
                cam_quat = None
            else:
                cam_pos = camera_poses[env_id, :3]
                cam_quat = camera_poses[env_id, 3:7]
                T_matrix = None

            depth_2d = augmented_depth[env_id, :, :, 0]
            rgb_2d = augmented_rgb[env_id]

            original_depth_2d = depth_maps[env_id, :, :, 0].clone()

            for obj_idx in range(dynamic_manager.max_objects):
                if dynamic_manager.active[env_id, obj_idx]:
                    obj_pos = dynamic_manager.positions[env_id, obj_idx]
                    obj_color = sphere_colors[obj_idx % len(sphere_colors)]
                    obj_type = dynamic_manager.object_types[env_id][obj_idx]

                    if obj_type == 'sphere':
                        sphere_radius = dynamic_manager.radii[env_id, obj_idx].item()

                        depth_2d = self.inject_sphere(
                            depth_2d,
                            obj_pos,
                            sphere_radius,
                            cam_pos,
                            cam_quat,
                            T_matrix
                        )

                        rgb_2d = self.inject_sphere_rgb(
                            rgb_2d,
                            original_depth_2d,
                            obj_pos,
                            sphere_radius,
                            obj_color,
                            cam_pos,
                            cam_quat,
                            T_matrix,
                            use_shading,
                            debug=False
                        )
                    elif obj_type == 'box':
                        box_size = dynamic_manager.box_sizes[env_id, obj_idx]
                        box_rotation = dynamic_manager.box_rotations[env_id, obj_idx]

                        depth_2d = self.inject_box(
                            depth_2d,
                            obj_pos,
                            box_size,
                            box_rotation,
                            cam_pos,
                            cam_quat,
                            T_matrix
                        )

                        rgb_2d = self.inject_box_rgb(
                            rgb_2d,
                            original_depth_2d,
                            obj_pos,
                            box_size,
                            box_rotation,
                            obj_color,
                            cam_pos,
                            cam_quat,
                            T_matrix,
                            use_shading,
                            debug=False
                        )
                    elif obj_type == 'cylinder':
                        cylinder_params = dynamic_manager.cylinder_params[env_id, obj_idx]
                        cylinder_radius = cylinder_params[0].item()
                        cylinder_height = cylinder_params[1].item()
                        cylinder_axis = int(cylinder_params[2].item())
                        cylinder_rotation = dynamic_manager.cylinder_rotations[env_id, obj_idx]

                        depth_2d = self.inject_cylinder(
                            depth_2d,
                            obj_pos,
                            cylinder_radius,
                            cylinder_height,
                            cylinder_axis,
                            cylinder_rotation,
                            cam_pos,
                            cam_quat,
                            T_matrix
                        )

                        rgb_2d = self.inject_cylinder_rgb(
                            rgb_2d,
                            original_depth_2d,
                            obj_pos,
                            cylinder_radius,
                            cylinder_height,
                            cylinder_axis,
                            cylinder_rotation,
                            obj_color,
                            cam_pos,
                            cam_quat,
                            T_matrix,
                            use_shading,
                            debug=False
                        )

            augmented_depth[env_id, :, :, 0] = depth_2d
            augmented_rgb[env_id] = rgb_2d

        return augmented_depth, augmented_rgb
