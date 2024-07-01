import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torchtyping import TensorType
from itertools import product

num_samples, num_cams = None, None


def rotvec_to_matrix(rotvec):
    return R.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(mat):
    return R.from_matrix(mat).as_rotvec()


def compose_rotvec(r1, r2):
    """
    #TODO: adapt to torch
    Compose two rotation euler vectors.
    """
    r1 = r1.cpu().numpy() if isinstance(r1, torch.Tensor) else r1
    r2 = r2.cpu().numpy() if isinstance(r2, torch.Tensor) else r2

    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum("...ij,...jk->...ik", R1, R2)
    return torch.from_numpy(matrix_to_rotvec(cR))


def quat_to_rotvec(quat, eps=1e-6):
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = (
        small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    )
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec


# batch*n
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).to(v.device)
    )
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag is True:
        return v, v_mag[:, 0]
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # [batch, 6]

    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # [batch, 6]
    y_raw = ortho6d[:, 3:6]  # [batch, 6]

    x = normalize_vector(x_raw)  # [batch, 6]
    z = cross_product(x, y_raw)  # [batch, 6]
    z = normalize_vector(z)  # [batch, 6]
    y = cross_product(z, x)  # [batch, 6]

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # [batch, 3, 3]
    return matrix


def invert_rotvec(rotvec: TensorType["num_samples", 3]):
    angle = torch.norm(rotvec, dim=-1)
    axis = rotvec / (angle.unsqueeze(-1) + 1e-6)
    inverted_rotvec = -angle.unsqueeze(-1) * axis
    return inverted_rotvec


def are_rotations(matrix: TensorType["num_samples", 3, 3]) -> TensorType["num_samples"]:
    """Check if a matrix is a rotation matrix."""
    # Check if the matrix is orthogonal
    identity = torch.eye(3, device=matrix.device)
    is_orthogonal = (
        torch.isclose(torch.bmm(matrix, matrix.transpose(1, 2)), identity, atol=1e-6)
        .all(dim=1)
        .all(dim=1)
    )

    # Check if the determinant is 1
    determinant = torch.det(matrix)
    is_determinant_one = torch.isclose(
        determinant, torch.tensor(1.0, device=matrix.device), atol=1e-6
    )

    return torch.logical_and(is_orthogonal, is_determinant_one)


def project_so3(
    matrix: TensorType["num_samples", 4, 4]
) -> TensorType["num_samples", 4, 4]:
    # Project rotation matrix to SO(3)
    # TODO: use torch
    rot = R.from_matrix(matrix[:, :3, :3].cpu().numpy()).as_matrix()

    projection = torch.eye(4).unsqueeze(0).repeat(matrix.shape[0], 1, 1).to(matrix)
    projection[:, :3, :3] = torch.from_numpy(rot).to(matrix)
    projection[:, :3, 3] = matrix[:, :3, 3]

    return projection


def pairwise_geodesic(
    R_x: TensorType["num_samples", "num_cams", 3, 3],
    R_y: TensorType["num_samples", "num_cams", 3, 3],
    reduction: str = "mean",
    block_size: int = 200,
):
    def arange(start, stop, step, endpoint=True):
        arr = torch.arange(start, stop, step)
        if endpoint and arr[-1] != stop - 1:
            arr = torch.cat((arr, torch.tensor([stop - 1], dtype=arr.dtype)))
        return arr

    # Geodesic distance
    # https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    num_samples, num_cams, _, _ = R_x.shape

    C = torch.zeros(num_samples, num_samples, device=R_x.device)
    chunk_indices = arange(0, num_samples + 1, block_size, endpoint=True)
    for i, j in product(
        range(chunk_indices.shape[0] - 1), range(chunk_indices.shape[0] - 1)
    ):
        start_x, stop_x = chunk_indices[i], chunk_indices[i + 1]
        start_y, stop_y = chunk_indices[j], chunk_indices[j + 1]
        r_x, r_y = R_x[start_x:stop_x], R_y[start_y:stop_y]

        # Compute rotations between each pair of cameras of each sample
        r_xy = torch.einsum("anjk,bnlk->abnjl", r_x, r_y)  # b, b, N, 3, 3

        # Compute axis-angle representations: angle is the geodesic distance
        traces = r_xy.diagonal(dim1=-2, dim2=-1).sum(-1)
        c = torch.acos(torch.clamp((traces - 1) / 2, -1, 1)) / torch.pi

        # Average distance between cameras over samples
        if reduction == "mean":
            C[start_x:stop_x, start_y:stop_y] = c.mean(-1)
        elif reduction == "sum":
            C[start_x:stop_x, start_y:stop_y] = c.sum(-1)

        # Check for NaN values in traces
        if torch.isnan(c).any():
            raise ValueError("NaN values detected in traces")

    return C
