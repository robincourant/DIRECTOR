from .scene import setup_scene  # noqa
from .floor import plot_floor
from .camera import Camera
from .sampler import get_frameidx
from .meshes_w_cams import MeshesWithCameras
from .materials import CAM_TO_MATERIAL, MESH_TO_MATERIAL


def render(
    traj,
    vertices,
    faces,
    traj_index=None,
    mesh_color="Blues",
    cam_color="Blues",
    cam_segments=None,
    mesh_segments=None,
    mode="image",
    exact_frame=None,
    num=8,
    denoising=True,
    oldrender=True,
    res="high",
    init=False,
):
    assert mode in ["image", "video", "video_accumulate"]

    if not init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender)

    data = MeshesWithCameras(
        cams=traj,
        vertices=vertices,
        faces=faces,
        traj_index=traj_index,
        mode=mode,
        cam_color=cam_color,
        mesh_color=mesh_color,
    )

    if not init:
        # Create a floor
        plot_floor(data.data)

    # initialize the camera
    Camera(first_root=data.get_root(0), mode=mode, is_mesh=True)

    nframes = len(data)
    frames_to_keep = num if mode == "image" else nframes
    frame_indices = get_frameidx(
        mode=mode,
        nframes=nframes,
        exact_frame=exact_frame,
        frames_to_keep=frames_to_keep,
    )
    frame_to_keep_indices = list(
        get_frameidx(
            mode=mode,
            nframes=nframes,
            exact_frame=exact_frame,
            frames_to_keep=num,
        )
    )
    nframes_to_render = len(frame_indices)

    imported_obj_names = []
    for index, frame_index in enumerate(frame_indices):
        if cam_segments is None or mesh_segments is None:
            if mode == "image":
                frac = index / (nframes_to_render - 1)
                cam_mat = data.get_cam_sequence_mat(frac)
                mesh_mat = data.get_mesh_sequence_mat(frac)
            else:  # Video mode, no color hue
                cam_mat = data.get_cam_sequence_mat(1.0)
                mesh_mat = data.get_mesh_sequence_mat(1.0)
        else:
            cam_mat = CAM_TO_MATERIAL[cam_segments[frame_index]]
            mesh_mat = MESH_TO_MATERIAL[mesh_segments[frame_index]]

        keep_frame = frame_index in frame_to_keep_indices
        mesh_name, cam_name = data.load_in_blender(
            frame_index, cam_mat, mesh_mat, mode, keep_frame
        )
        curve_name = data.show_cams(frame_index + 1, mode=mode)
        imported_obj_names.extend([mesh_name, cam_name, curve_name])

    return imported_obj_names
