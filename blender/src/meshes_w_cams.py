import numpy as np
import bpy
import matplotlib
import trimesh

from .materials import body_material


def prepare_data(cams, vertices):
    # Remove the floor
    offset = vertices[..., 2].min()
    vertices[..., 2] -= offset
    cams[:, 2, 3] -= offset
    return cams, vertices


class MeshesWithCameras:
    def __init__(
        self,
        cams,
        vertices,
        mode,
        faces,
        traj_index=None,
        oldrender=True,
        mesh_color="Blues",
        cam_color="Blues",
        **kwargs,
    ):
        cams, vertices = prepare_data(cams, vertices)

        self.faces = faces
        self.data = vertices
        self.mode = mode
        self.oldrender = oldrender
        self.mesh_color = mesh_color
        self.cam_color = cam_color
        self.cams = cams
        self.traj_index = traj_index

        self.N = len(vertices)
        self.trajectory = vertices[:, :, [0, 1]].mean(1)

        self.cam_vertices, self.cam_faces = self.load_cam_marker()

    def load_cam_marker(self):
        cam_marker = trimesh.load_mesh("./blender/cam_marker.stl")
        cam_vertices = (cam_marker.vertices / cam_marker.vertices.max()) * 0.2
        cam_vertices[:, 2] *= -1
        cam_vertices[:, 2] += 0.2
        cam_faces = cam_marker.faces

        return cam_vertices, cam_faces

    def get_mesh_sequence_mat(self, frac):
        cmap = matplotlib.cm.get_cmap(self.mesh_color)
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end - begin) * frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_cam_sequence_mat(self, frac):
        cmap = matplotlib.cm.get_cmap(self.cam_color)
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end - begin) * frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.cams[index, :3, 3]

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, cam_mat, mesh_mat, mode, keep_frame):
        from .tools import load_numpy_vertices_into_blender

        suffix = "" if self.traj_index is None else f"_{str(self.traj_index).zfill(2)}"

        vertices = self.data[index]
        faces = self.faces
        mesh_name = f"{str(index).zfill(4)}_mesh{suffix}"
        load_numpy_vertices_into_blender(
            vertices, faces, mesh_name, mesh_mat, index, mode, keep_frame
        )

        h_cam_vertices = np.hstack(
            [self.cam_vertices, np.ones(self.cam_vertices.shape[0])[:, None]]
        )
        cam = np.eye(4)
        cam[:3] = self.cams[index]

        marker = (cam @ h_cam_vertices.T).T[:, :3]
        cam_name = f"{str(index).zfill(4)}_cam{suffix}"
        load_numpy_vertices_into_blender(
            marker, self.cam_faces, cam_name, cam_mat, index, mode, keep_frame
        )

        return mesh_name, cam_name

    def show_cams(self, index, mode):
        suffix = "" if self.traj_index is None else f"_{str(self.traj_index).zfill(2)}"
        name = f"{str(index).zfill(4)}_curve{suffix}"

        # create the Curve Datablock
        curveData = bpy.data.curves.new(name, type="CURVE")
        curveData.dimensions = "3D"
        curveData.resolution_u = 2

        # map coords to spline
        polyline = curveData.splines.new("POLY")
        polyline.points.add(len(self.cams[:index]) - 1)
        for i, coord in enumerate(self.cams[:index, :3, 3]):
            x, y, z = coord
            polyline.points[i].co = (x, y, z, 1)

        # create Object
        curveOB = bpy.data.objects.new(name, curveData)
        curveData.bevel_depth = 0.01

        bpy.context.collection.objects.link(curveOB)

        if "video" in mode:
            # Initialize object as hidden
            curveOB.hide_viewport = True
            curveOB.hide_render = True
            curveOB.keyframe_insert(data_path="hide_viewport", frame=index - 1)
            curveOB.keyframe_insert(data_path="hide_render", frame=index - 1)

            # Make object visible at the specified frame
            curveOB.hide_viewport = False
            curveOB.hide_render = False
            curveOB.keyframe_insert(data_path="hide_viewport", frame=index)
            curveOB.keyframe_insert(data_path="hide_render", frame=index)

            if "accumulate" not in mode:
                # Hide object after the specified frame
                curveOB.hide_viewport = True
                curveOB.hide_render = True
                curveOB.keyframe_insert(data_path="hide_viewport", frame=index + 1)
                curveOB.keyframe_insert(data_path="hide_render", frame=index + 1)

        # Optional: Set shading to smooth
        bpy.ops.object.select_all(action="DESELECT")
        curveOB.select_set(True)
        bpy.context.view_layer.objects.active = curveOB
        bpy.ops.object.shade_smooth()
        bpy.ops.object.select_all(action="DESELECT")

        return name

    def __len__(self):
        return self.N
