from functools import partial
from typing import Any, Callable, Dict

import clip
import gradio as gr
from gradio_rerun import Rerun
import numpy as np
import trimesh
import rerun as rr
import torch

from utils.random_utils import set_random_seed
from utils.rerun import log_sample
from src.datasets.multimodal_dataset import MultimodalDataset
from src.training.diffuser import Diffuser
from visualization.common_viz import init, get_batch

# ------------------------------------------------------------------------------------- #

batch_size, num_cams, num_verts = None, None, None

SAMPLE_IDS = [
    "2011_KAeAqaA0Llg_00005_00001",
    "2011_F_EuMeT2wBo_00014_00001",
    "2011_MCkKihQrNA4_00014_00000",
]
LABEL_TO_IDS = {
    "right": 0,
    "static": 1,
    "complex": 2,
}
EXAMPLES = [
    "While the character moves right, the camera trucks right.",
    "While the character moves right, the camera performs a push in.",
    "While the character moves right, the camera performs a pull out.",
    "While the character stays static, the camera performs a boom bottom.",
    "While the character stays static, the camera performs a boom top.",
    "While the character moves to the right, the camera trucks right alongside them. "
    "Once the character comes to a stop, the camera remains static.",
    "While the character moves to the right, the camera remains static. "
    "Once the character comes to a stop, the camera pushes in.",
]
DEFAULT_TEXT = [
    "While the character moves right, the camera [...].",
    "While the character remains static, [...].",
    "While the character moves to the right, the camera [...]. "
    "Once the character comes to a stop, the camera [...].",
]

HEADER = """

<div align="center">
<h1 style='text-align: center'>E.T. the Exceptional Trajectories</h2>
<a href="https://robincourant.github.io/info/"><strong>Robin Courant</strong></a>
·
<a href="https://nicolas-dufour.github.io/"><strong>Nicolas Dufour</strong></a>
·
<a href="https://triocrossing.github.io/"><strong>Xi Wang</strong></a>
·
<a href="http://people.irisa.fr/Marc.Christie/"><strong>Marc Christie</strong></a>
·
<a href="https://vicky.kalogeiton.info/"><strong>Vicky Kalogeiton</strong></a>
</div>


<div align="center">
    <a href="https://www.lix.polytechnique.fr/vista/projects/2024_et_courant/" class="button"><b>[Webpage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/robincourant/DIRECTOR" class="button"><b>[DIRECTOR]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/robincourant/CLaTr" class="button"><b>[CLaTr]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/robincourant/the-exceptional-trajectories" class="button"><b>[Data]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

<br/>
"""

# ------------------------------------------------------------------------------------- #


def get_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    num_frames, num_faces = vertices.shape[0], faces.shape[-2]
    faces = faces.expand(num_frames, num_faces, 3)

    normals = [
        trimesh.Trimesh(vertices=v, faces=f, process=False).vertex_normals
        for v, f in zip(vertices, faces)
    ]
    normals = torch.from_numpy(np.stack(normals))

    return normals


def generate(
    prompt: str,
    seed: int,
    guidance_weight: float,
    sample_label: str,
    # ----------------------- ß#
    dataset: MultimodalDataset,
    device: torch.device,
    diffuser: Diffuser,
    clip_model: clip.model.CLIP,
) -> Dict[str, Any]:
    # Set arguments
    set_random_seed(seed)
    diffuser.gen_seeds = np.array([seed])
    diffuser.guidance_weight = guidance_weight

    # Inference
    sample_id = SAMPLE_IDS[LABEL_TO_IDS[sample_label]]
    seq_feat = diffuser.net.model.clip_sequential
    batch = get_batch(prompt, sample_id, clip_model, dataset, seq_feat, device)
    with torch.no_grad():
        out = diffuser.predict_step(batch, 0)

    # Run visualization
    padding_mask = out["padding_mask"][0].to(bool).cpu()
    padded_traj = out["gen_samples"][0].cpu()
    traj = padded_traj[padding_mask]
    padded_vertices = out["char_raw"]["char_vertices"][0]
    vertices = padded_vertices[padding_mask]
    faces = out["char_raw"]["char_faces"][0]
    normals = get_normals(vertices, faces)
    fx, fy, cx, cy = out["intrinsics"][0].cpu().numpy()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    caption = out["caption_raw"][0]

    rr.init(f"{sample_id}")
    rr.save(".tmp_gr.rrd")
    log_sample(
        root_name="world",
        traj=traj.numpy(),
        K=K,
        vertices=vertices.numpy(),
        faces=faces.numpy(),
        normals=normals.numpy(),
        caption=caption,
    )
    return "./.tmp_gr.rrd"


# ------------------------------------------------------------------------------------- #


def main(gen_fn: Callable):
    theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(HEADER)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Column(scale=2):
                    sample_str = gr.Dropdown(
                        choices=["static", "right", "complex"],
                        label="Character trajectory",
                        value="right",
                        interactive=True,
                    )
                    text = gr.Textbox(
                        placeholder="Type the camera motion you want to generate",
                        show_label=True,
                        label="Text prompt",
                        value=DEFAULT_TEXT[LABEL_TO_IDS[sample_str.value]],
                    )
                    seed = gr.Number(value=33, label="Seed")
                    guidance = gr.Slider(0, 10, value=1.4, label="Guidance", step=0.1)

                with gr.Column(scale=1):
                    btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                examples = gr.Examples(
                    examples=[[x, None, None] for x in EXAMPLES],
                    inputs=[text],
                )

        with gr.Row():
            output = Rerun()

        def load_example(example_id):
            processed_example = examples.non_none_processed_examples[example_id]
            return gr.utils.resolve_singleton(processed_example)

        def change_fn(change):
            sample_index = LABEL_TO_IDS[change]
            return gr.update(value=DEFAULT_TEXT[sample_index])

        sample_str.change(fn=change_fn, inputs=[sample_str], outputs=[text])

        inputs = [text, seed, guidance, sample_str]
        examples.dataset.click(
            load_example,
            inputs=[examples.dataset],
            outputs=examples.inputs_with_examples,
            show_progress=False,
            postprocess=False,
            queue=False,
        ).then(fn=gen_fn, inputs=inputs, outputs=[output])
        btn.click(fn=gen_fn, inputs=inputs, outputs=[output])
        text.submit(fn=gen_fn, inputs=inputs, outputs=[output])
    demo.launch(share=False)


# ------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    # Initialize the models and dataset
    diffuser, clip_model, dataset, device = init("config_viz")
    generate_sample = partial(
        generate,
        dataset=dataset,
        device=device,
        diffuser=diffuser,
        clip_model=clip_model,
    )

    main(generate_sample)
