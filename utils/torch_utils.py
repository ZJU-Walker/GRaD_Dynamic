"""
PyTorch Utilities

Provides common PyTorch helper functions for tensor operations and quaternion math.
"""

import torch


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    """Convert input to PyTorch tensor."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    """Normalize tensor along last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_mul(a, b):
    """Multiply two quaternions."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_apply(a, b):
    """Apply quaternion rotation to vector."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    """Rotate vector v by quaternion q."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_axis(q, axis: int = 0):
    """Get axis vector from quaternion."""
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def quat_conjugate(a):
    """Compute conjugate of quaternion."""
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    """Normalize quaternion to unit quaternion."""
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    """Create quaternion from angle and axis."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x):
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    """Compute inverse of transform."""
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    """Apply transform to vector."""
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    """Apply rotation to vector."""
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    """Combine two transforms."""
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    """Get basis vector rotated by quaternion."""
    return quat_rotate(q, v)

def grad_norm(params):
    """Calculate gradient norm across all parameters."""
    grad_norm = 0.
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad ** 2)
    return torch.sqrt(grad_norm)


def calculate_max_mean_gradient(parameters):
    """Calculate max and mean absolute gradient values."""
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad.view(-1))

    if gradients:
        gradients = torch.cat(gradients)
        mean_gradient = torch.mean(torch.abs(gradients))
        max_gradient = torch.max(torch.abs(gradients))
        return max_gradient, mean_gradient
    else:
        return None, None
