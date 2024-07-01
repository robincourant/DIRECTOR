import sys
import os

import bpy
import numpy as np
import torch

sys.path.append(os.path.dirname(bpy.data.filepath))

from blender.render_script import Renderer  # noqa
from utils.random_utils import set_random_seed  # noqa
from visualization.common_viz import get_batch, init  # noqa

# ------------------------------------------------------------------------------------- #

SEED = 33
W_GUIDANCE = 1.4
SAMPLE_ID = "2011_KAeAqaA0Llg_00005_00001"
PROMPT = "While the character moves right, the camera performs a boom bottom."

renderer = Renderer()

# ------------------------------------------------------------------------------------- #


# Define the Blender add-on
class GenerationAddon(bpy.types.Operator):
    bl_idname = "generation.generate"
    bl_label = "Generate"

    def execute(self, context):
        # Get the arguments from the UI
        seed = int(bpy.context.scene.seed)
        guidance_weight = float(bpy.context.scene.guidance_weight)
        sample_id = bpy.context.scene.sample_id
        prompt = bpy.context.scene.prompt

        # Set the arguments
        set_random_seed(seed)
        diffuser.gen_seeds = np.array([seed])
        diffuser.guidance_weight = guidance_weight

        # Inference
        seq_feat = diffuser.net.model.clip_sequential
        batch = get_batch(prompt, sample_id, clip_model, dataset, seq_feat, device)
        with torch.no_grad():
            out = diffuser.predict_step(batch, 0)

        renderer.render_cli(
            out,
            bpy.context.scene.selected_rate,
            bpy.context.scene.keep_traj,
            bpy.context.scene.render_mode.lower(),
        )

        return {"FINISHED"}


# Define the UI panel
class GENERATION_PT_generation_panel(bpy.types.Panel):
    bl_idname = "GENERATION_PT_generation_panel"
    bl_label = "DIRECTOR Generation"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_category = "Object"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "seed")
        layout.prop(context.scene, "guidance_weight")
        layout.prop(context.scene, "sample_id")
        layout.prop(context.scene, "prompt")
        layout.prop(context.scene, "selected_rate")
        layout.prop(context.scene, "keep_traj")
        layout.prop(context.scene, "render_mode")
        layout.operator("generation.generate", text="Generate")


# Register the add-on
def register():
    bpy.utils.register_class(GenerationAddon)
    bpy.utils.register_class(GENERATION_PT_generation_panel)
    bpy.types.Scene.seed = bpy.props.IntProperty(name="Seed", default=SEED)
    bpy.types.Scene.guidance_weight = bpy.props.FloatProperty(
        name="Guidance Weight", default=W_GUIDANCE
    )
    bpy.types.Scene.sample_id = bpy.props.StringProperty(
        name="Sample ID", default=SAMPLE_ID
    )
    bpy.types.Scene.prompt = bpy.props.StringProperty(
        name="Prompt", default=PROMPT
    )
    bpy.types.Scene.selected_rate = bpy.props.FloatProperty(
        name="Rate of poses", default=0.1
    )
    bpy.types.Scene.keep_traj = bpy.props.BoolProperty(
        name="Keep trajectory", default=False
    )
    bpy.types.Scene.render_mode = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose generation mode",
        items=[
            ("IMAGE", "Image", ""),
            ("VIDEO", "Video", ""),
            ("VIDEO_ACCUMULATE", "Video Accumulate", ""),
        ],
        default="IMAGE",
    )


def unregister():
    bpy.utils.unregister_class(GenerationAddon)
    bpy.utils.unregister_class(GENERATION_PT_generation_panel)
    del bpy.types.Scene.seed
    del bpy.types.Scene.guidance_weight
    del bpy.types.Scene.sample_id
    del bpy.types.Scene.prompt
    del bpy.types.Scene.selected_rate
    del bpy.types.Scene.keep_traj
    del bpy.types.Scene.render_mode


if __name__ == "__main__":
    register()

    # Initialize the models and dataset
    diffuser, clip_model, dataset, device = init("config_viz")
