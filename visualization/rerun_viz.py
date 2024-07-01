import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import rerun as rr
import torch

from utils.random_utils import set_random_seed
from utils.rerun import log_sample
from visualization.common_viz import get_batch, init

# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 1.4
SAMPLE_ID = "2011_KAeAqaA0Llg_00005_00001"
PROMPT = "While the character moves right, the camera performs a boom bottom."

RENDER_SCALE = 4

# ------------------------------------------------------------------------------------- #


def get_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    num_frames, num_faces = vertices.shape[0], faces.shape[-2]
    faces = faces.expand(num_frames, num_faces, 3)

    verts_rgb = torch.ones_like(vertices)
    verts_rgb[:, :, 1] = 0
    textures = TexturesVertex(verts_features=verts_rgb)
    meshes = Meshes(verts=vertices, faces=faces, textures=textures)
    normals = meshes.verts_normals_padded()

    return normals, meshes


# ------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    diffuser, clip_model, dataset, device = init("config_viz")

    while True:
        # ----------------------------------------------------------------------------- #
        # Arguments
        SEED = int(input(f"Seed (default={SEED}): ") or SEED)
        W_GUIDANCE = float(input(f"Guidance (default={W_GUIDANCE}): ") or W_GUIDANCE)
        SAMPLE_ID = input(f"Sample ID (default={SAMPLE_ID}): ") or SAMPLE_ID
        PROMPT = input(f"Prompt (default='{PROMPT}'): ") or PROMPT
        # ----------------------------------------------------------------------------- #

        # Set arguments
        set_random_seed(SEED)
        diffuser.gen_seeds = np.array([SEED])
        diffuser.guidance_weight = W_GUIDANCE

        # Inference
        seq_feat = diffuser.net.model.clip_sequential
        batch = get_batch(PROMPT, SAMPLE_ID, clip_model, dataset, seq_feat, device)
        with torch.no_grad():
            out = diffuser.predict_step(batch, 0)

        # Run visualization
        padding_mask = out["padding_mask"][0].to(bool).cpu()
        padded_traj = out["gen_samples"][0].cpu()
        traj = padded_traj[padding_mask]
        padded_vertices = out["char_raw"]["char_vertices"][0]
        vertices = padded_vertices[padding_mask]
        faces = out["char_raw"]["char_faces"][0]
        normals, meshes = get_normals(vertices, faces)
        fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        caption = out["caption_raw"][0]

        accel = "cuda" if device.type == "cuda" else "cpu"

        rr.init(f"{SAMPLE_ID}", spawn=True)
        log_sample(
            root_name="world",
            traj=traj.numpy(),
            K=K,
            vertices=vertices.numpy(),
            faces=faces.numpy(),
            normals=normals.numpy(),
            caption=caption,
        )
