import os
import sys
import logging
from typing import Any, Dict

import bpy

sys.path.append(os.path.dirname(bpy.data.filepath))

from blender.src.render import render  # noqa
from blender.src.tools import delete_objs  # noqa

logger = logging.getLogger(__name__)

CAM_COLOR_PALETTES = [
    "Purples",
    "Greens",
    "Reds",
    "Oranges",
    "Greys",
]


class Renderer:
    def __init__(self):
        self.obj_names = None
        self.obj_to_keep = []
        self.traj_index = 0

    def render_cli(
        self, out: Dict[str, Any], selected_rate: float, keep_traj: bool, mode: str
    ):
        if keep_traj:
            suffix = f"_{str(self.traj_index).zfill(2)}"
            self.obj_to_keep.extend(
                [
                    x
                    for x in self.obj_names
                    if ("cam" + suffix in x) or ("curve" + suffix in x)
                ]
            )
            self.traj_index += 1

        if self.obj_names is not None:
            to_remove = list(set(self.obj_names).difference(set(self.obj_to_keep)))
            delete_objs(to_remove)
            delete_objs(["Plane", "myCurve", "Cylinder"])

        # Run visualization
        padding_mask = out["padding_mask"][0].to(bool).cpu()
        padded_traj = out["gen_samples"][0].cpu()
        traj = padded_traj[padding_mask].numpy()
        traj = traj[:, [0, 2, 1]]
        traj[:, 2] = -traj[:, 2]
        padded_vertices = out["char_raw"]["char_vertices"][0]
        vertices = padded_vertices[padding_mask].numpy()
        vertices = vertices[..., [0, 2, 1]]
        vertices[:, :, 2] = -vertices[:, :, 2]
        faces = out["char_raw"]["char_faces"][0].numpy()
        faces = faces[..., [0, 2, 1]]

        nframes = traj.shape[0]
        # Set the final frame of the playback
        if "video" in mode:
            bpy.context.scene.frame_end = nframes - 1
        num = int(selected_rate * nframes)
        self.obj_names = render(
            traj=traj,
            vertices=vertices,
            faces=faces,
            cam_color=CAM_COLOR_PALETTES[self.traj_index % len(CAM_COLOR_PALETTES)],
            mesh_color="Blues",
            traj_index=self.traj_index,
            # ---------------------------- #
            denoising=True,
            oldrender=True,
            res="low",
            exact_frame=0.5,
            num=num,
            mode=mode,
            init=False,
        )
