import numpy as np
from matplotlib import colormaps
import rerun as rr
from rerun.components import Material
from scipy.spatial import transform


def color_fn(x, cmap="tab10"):
    return colormaps[cmap](x % colormaps[cmap].N)


def log_sample(
    root_name: str,
    traj: np.ndarray,
    K: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    caption: str,
):
    num_cameras = traj.shape[0]

    rr.log(root_name, rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log(
        f"{root_name}/trajectory/points",
        rr.Points3D(traj[:, :3, 3]),
        timeless=True,
    )
    rr.log(
        f"{root_name}/trajectory/line",
        rr.LineStrips3D(
            np.stack((traj[:, :3, 3][:-1], traj[:, :3, 3][1:]), axis=1),
            colors=[(1.0, 0.0, 1.0, 1.0)],
        ),
        timeless=True,
    )
    for k in range(num_cameras):
        rr.set_time_sequence("frame_idx", k)

        translation = traj[k][:3, 3]
        rotation_q = transform.Rotation.from_matrix(traj[k][:3, :3]).as_quat()
        rr.log(
            f"{root_name}/camera/image",
            rr.Pinhole(
                image_from_camera=K,
                width=K[0, -1] * 2,
                height=K[1, -1] * 2,
            ),
        )
        rr.log(
            f"{root_name}/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
            ),
        )
        rr.set_time_sequence("image", k)

        # Null vertices
        if vertices[k].sum() == 0:
            rr.log(f"{root_name}/human/human", rr.Clear(recursive=False))
            rr.log(f"{root_name}/camera/image/bbox", rr.Clear(recursive=False))
            continue

        rr.log(
            f"{root_name}/human/human",
            rr.Mesh3D(
                vertex_positions=vertices[k],
                indices=faces,
                vertex_normals=normals[k],
                mesh_material=Material(albedo_factor=color_fn(0)),
            ),
        )
    rr.log(
        f"{root_name}/caption",
        rr.TextDocument(caption, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )
