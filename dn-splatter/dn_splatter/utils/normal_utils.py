"""Utils for normals: depth-derived normals, Gaussian normals init/derivation, and rendering."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from dn_splatter.utils.camera_utils import get_means3d_backproj

try:
    from gsplat.rendering import rasterization
except ImportError:
    rasterization = None


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (wxyz convention) to rotation matrices."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return mat.reshape(quats.shape[:-1] + (3, 3))


def compute_normals_from_scales_quats(
    scales: Tensor,
    quats: Tensor,
    quat_to_rotmat_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Derive normals from smallest scale axis (geometry-derived from Gaussian shape)."""
    if quat_to_rotmat_fn is None:
        quat_to_rotmat_fn = quat_to_rotmat
    normals = F.one_hot(torch.argmin(scales, dim=-1), num_classes=3).float()
    rots = quat_to_rotmat_fn(quats)
    normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)
    return F.normalize(normals, dim=1)


def rotate_vector_to_vector(v1: Tensor, v2: Tensor) -> Tensor:
    """Returns a rotation matrix that rotates v1 to align with v2."""
    if v1.dim() == 1:
        v1 = v1[None, ...]
        v2 = v2[None, ...]
    N = v1.shape[0]
    u = v1 / torch.norm(v1, dim=-1, keepdim=True)
    Ru = v2 / torch.norm(v2, dim=-1, keepdim=True)
    I = torch.eye(3, 3, device=v1.device).unsqueeze(0).repeat(N, 1, 1)
    c = torch.bmm(u.view(N, 1, 3), Ru.view(N, 3, 1)).squeeze(-1)
    eps = 1.0e-10
    K = torch.bmm(Ru.unsqueeze(2), u.unsqueeze(1)) - torch.bmm(
        u.unsqueeze(2), Ru.unsqueeze(1)
    )
    ans = I + K + (K @ K) / (1 + c)[..., None]
    same = torch.abs(c - 1.0).squeeze(-1) < eps
    opposite = torch.abs(c + 1.0).squeeze(-1) < eps
    ans[same] = torch.eye(3, device=v1.device)
    ans[opposite] = -torch.eye(3, device=v1.device)
    return ans


def matrix_to_quaternion(rotation_matrix: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to a unit quaternion (wxyz)."""
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix[None, ...]
    traces = torch.vmap(torch.trace)(rotation_matrix)
    quaternion = torch.zeros(
        rotation_matrix.shape[0], 4,
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    for i in range(rotation_matrix.shape[0]):
        matrix = rotation_matrix[i]
        trace = traces[i]
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w, x = 0.25 * S, (matrix[2, 1] - matrix[1, 2]) / S
            y, z = (matrix[0, 2] - matrix[2, 0]) / S, (matrix[1, 0] - matrix[0, 1]) / S
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w, x = (matrix[2, 1] - matrix[1, 2]) / S, 0.25 * S
            y = (matrix[0, 1] + matrix[1, 0]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x, y = (matrix[0, 1] + matrix[1, 0]) / S, (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S
        else:
            S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S
        quaternion[i] = torch.tensor([w, x, y, z], dtype=matrix.dtype, device=matrix.device)
    return quaternion


def init_normals_from_seed(normals_seed: Tensor) -> Tensor:
    """Initialize normals from seed point normals (normalized)."""
    normals = normals_seed / torch.norm(normals_seed, dim=-1, keepdim=True)
    return normals


def init_normals_and_quats_from_seed(normals_seed: Tensor) -> Tuple[Tensor, Tensor]:
    """Initialize normals and quats from seed normals. Returns (normals, quats)."""
    normals = init_normals_from_seed(normals_seed)
    z_axis = torch.tensor(
        [0, 0, 1], dtype=torch.float, device=normals_seed.device
    ).repeat(normals_seed.shape[0], 1)
    mat = rotate_vector_to_vector(z_axis, normals)
    quats = matrix_to_quaternion(mat)
    return normals, quats


def init_normals_random(
    num_points: int,
    scales: Tensor,
    quats: Tensor,
    quat_to_rotmat_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Initialize normals randomly from scales and quats (smallest scale axis)."""
    normals = compute_normals_from_scales_quats(scales, quats, quat_to_rotmat_fn)
    return normals


def render_normals(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    camera_c2w: Tensor,
    rasterize_mode: str = "classic",
    tile_size: int = 16,
) -> Tensor:
    """Second rasterization pass to render normals (geometry-derived from scales/quats)."""
    if rasterization is None:
        raise ImportError("gsplat.rendering.rasterization required for render_normals")
    normals = compute_normals_from_scales_quats(scales, quats)
    viewdirs = (-means.detach() + camera_c2w[..., :3, 3]).detach()
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    dots = (normals * viewdirs).sum(-1)
    normals[dots < 0] = -normals[dots < 0]
    normals_cam = normals @ camera_c2w.squeeze(0)[:3, :3]
    normals_raster, _, _ = rasterization(
        means=means,
        quats=quats / quats.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales),
        opacities=opacities,
        colors=normals_cam,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        tile_size=tile_size,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=None,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=rasterize_mode,
    )
    normals_im = normals_raster.squeeze(0)
    norm = normals_im.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normals_im = normals_im / norm
    normals_im = ((normals_im + 1) / 2).clamp(0.0, 1.0)
    return normals_im


def pcd_to_normal(xyz: Tensor) -> Tensor:
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
    top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
    right_point = xyz[..., 1 : hd - 1, 2:wd, :]
    left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(
        xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
    ).permute(1, 2, 0)
    return xyz_normal


def normal_from_depth_image(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, cx, cy, img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals
