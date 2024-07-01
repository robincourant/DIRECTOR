import bpy
import numpy as np


def mesh_detect(data):
    # heuristic
    if data.shape[1] > 1000:
        return True
    return False


# see this for more explanation
# https://gist.github.com/iyadahmed/7c7c0fae03c40bd87e75dc7059e35377
# This should be solved with new version of blender
class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0


def load_numpy_vertices_into_blender(
    vertices, faces, name, mat, frame_index, mode, keep_frame
):
    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces.view(ndarray_pydata))
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Set object material
    obj.active_material = mat

    if "video" in mode:
        # Initialize object as hidden
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_index - 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_index - 1)

        # Make object visible at the specified frame
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_index)
        obj.keyframe_insert(data_path="hide_render", frame=frame_index)

        if ("accumulate" not in mode) or ("accumulate" in mode and not keep_frame):
            # Hide object visible after the specified frame
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=frame_index + 1)
            obj.keyframe_insert(data_path="hide_render", frame=frame_index + 1)

    # Optional: Set shading to smooth
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    obj.active_material = mat
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action="DESELECT")
    return True


def delete_objs(names):
    if not isinstance(names, list):
        names = [names]
    # bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        for name in names:
            if obj.name.startswith(name) or obj.name.endswith(name):
                obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action="DESELECT")
