from typing import List

import matplotlib.cm as cm
from evo.core.trajectory import PosePath3D
from evo.tools import plot
from evo.tools.settings import SETTINGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cmap_to_color = {"Reds": "#e41a1c", "Blues": "#377eb8", "Greens": "#4daf4a"}


def rotation_from_vec(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def draw_trajectories(
    trajectories: PosePath3D,
    save_path: str = None,
    marker_scale: float = None,
    marker_interval: int = 10,
    colormaps: List[str] = None,
) -> plot.PlotCollection:

    if "human_mesh" in trajectories:
        draw_human = True
        human_vertices = trajectories["human_mesh"]
        human_cmap = colormaps[list(trajectories.keys()).index("human_mesh")]
    else:
        draw_human = False

    # Change y and z axis for realistic view
    trajectories = {
        name: PosePath3D(poses_se3=[x[[0, 2, 1, 3]] for x in traj.poses_se3])
        for name, traj in trajectories.items()
        if name != "human_mesh"
    }

    if marker_scale is None:
        xyz = np.array([traj.positions_xyz for traj in trajectories.values()]).reshape(
            -1, 3
        )
        marker_scale = 0.1 * max(xyz.max(axis=0) - xyz.min(axis=0))

    plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
    figsize = tuple(SETTINGS.plot_figsize)
    fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=figsize)
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=figsize)
    fig_traj = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode[SETTINGS.plot_mode_default]
    ax_traj = plot.prepare_axis(fig_traj, plot_mode)

    # Set axis limits
    xyz = np.concatenate([traj.positions_xyz for traj in trajectories.values()])
    if draw_human:
        xyz = np.concatenate([xyz, human_vertices[0].reshape(-1, 3)])
    min_x, min_y, min_z = np.min(xyz, axis=0)
    max_x, max_y, max_z = np.max(xyz, axis=0)
    ax_traj.set_xlim(min_x, max_x)
    ax_traj.set_ylim(min_y, max_y)
    ax_traj.set_zlim(min_z, max_z)

    for k, (name, traj) in enumerate(trajectories.items()):
        color = next(ax_traj._get_lines.prop_cycler)["color"]
        plot.traj_xyz(axarr_xyz, traj, color=color, label=name, alpha=0.5)
        if name != "human_sample":
            plot.traj_rpy(axarr_rpy, traj, color=color, label=name, alpha=0.5)
        if colormaps is not None:
            plot_traj_colormap(ax_traj, traj, plot_mode, colormaps[k])
        else:
            plot.traj(ax_traj, plot_mode, traj, color=color, label=name, alpha=0.5)

        draw_coordinate_axes(
            ax_traj,
            traj,
            plot_mode,
            color=cmap_to_color[colormaps[k]],
            scale=marker_scale if name != "human_sample" else 0,
            interval=marker_interval,
        )

    if draw_human:
        vertices = human_vertices[:, ::marker_interval]
        num_frames = vertices.shape[1]
        cmap = plt.get_cmap(human_cmap)(np.linspace(0.3, 0.7, num_frames))
        for frame_index in range(num_frames):
            ax_traj.scatter(
                vertices[0, frame_index, :, 0],
                vertices[0, frame_index, :, 1],
                vertices[0, frame_index, :, 2],
                color=cmap[frame_index],
                marker="o",
                s=marker_scale,
            )

    plot_collection.add_figure("trajectories", fig_traj)
    plot_collection.add_figure("xyz_view", fig_xyz)
    plot_collection.add_figure("rpy_view", fig_rpy)
    if save_path is not None:
        plot_collection.export(save_path, confirm_overwrite=True)

    return plot_collection


def draw_coordinate_axes(
    ax: plt.Figure,
    traj: PosePath3D,
    plot_mode: plot.PlotMode,
    color: str,
    scale: float = 0.1,
    interval: int = 1,
) -> None:
    """
    Draws a coordinate frame axis for each pose of a trajectory.
    :param ax: plot axis
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param plot_mode: PlotMode value
    :param scale: affects the size of the marker (1. * scale)
    :param x_color: color of the x-axis
    :param y_color: color of the y-axis
    :param z_color: color of the z-axis
    """
    if scale <= 0:
        return

    # Transform start/end vertices of each axis to global frame.
    traj_poses = np.array(traj.poses_se3)[::interval]
    num_cameras = len(traj_poses)
    unit_x = np.array([1 * scale, 0, 0, 1])
    unit_y = np.array([0, 1 * scale, 0, 1])
    unit_z = np.array([0, 0, 1 * scale, 1])

    # Transform start/end vertices of each axis to global frame.
    x_vertices = np.array([[p[:3, 3], p.dot(unit_x)[:3]] for p in traj_poses])
    y_vertices = np.array([[p[:3, 3], p.dot(unit_y)[:3]] for p in traj_poses])
    z_vertices = np.array([[p[:3, 3], p.dot(unit_z)[:3]] for p in traj_poses])

    markers = plot.colored_line_collection(
        x_vertices.reshape(-1, 3),
        np.array(((num_cameras - 1) * [color]) + ["g"]),
        plot_mode,
        step=2,
        linestyles="dotted",
    )
    ax.add_collection(markers)
    markers = plot.colored_line_collection(
        z_vertices.reshape(-1, 3),
        np.array(((num_cameras - 1) * [color]) + ["g"]),
        plot_mode,
        step=2,
        linestyles="solid",
    )
    ax.add_collection(markers)
    markers = plot.colored_line_collection(
        y_vertices.reshape(-1, 3),
        np.array(((num_cameras - 1) * [color]) + ["g"]),
        plot_mode,
        step=2,
        linestyles="dotted",
    )
    ax.add_collection(markers)


def plot_traj_colormap(
    ax: plt.Axes,
    traj: PosePath3D,
    plot_mode: plot.PlotMode,
    cmap_name: str = "jet",
) -> None:
    """
    color map a path/trajectory in xyz coordinates according to
    an array of values
    :param ax: plot axis
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param array: Nx1 array of values used for color mapping
    :param plot_mode: PlotMode
    :param min_map: lower bound value for color mapping
    :param max_map: upper bound value for color mapping
    :param title: plot title
    :param fig: plot figure. Obtained with plt.gcf() if none is specified
    """
    array = range(traj.num_poses)
    pos = traj.positions_xyz
    norm = mpl.colors.Normalize(vmin=0, vmax=traj.num_poses, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mapper.set_array(array)
    colors = [mapper.to_rgba(a) for a in array]
    line_collection = plot.colored_line_collection(pos, colors, plot_mode)
    ax.add_collection(line_collection)
    if SETTINGS.plot_xyz_realistic:
        plot.set_aspect_equal(ax)
    # ax.autoscale_view(True, True, True)


def draw_traj(
    trajectory,
    axes: str,
    marker_scale: float,
    plot_collection: plot.PlotCollection,
    colormap: bool = False,
):
    figsize = tuple(SETTINGS.plot_figsize)
    fig_traj = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode[axes]
    ax_traj = plot.prepare_axis(fig_traj, plot_mode)
    if colormap:
        num_poses = trajectory.num_poses
        plot.traj_colormap(
            ax_traj, trajectory, range(num_poses), plot_mode, 0, num_poses
        )
    else:
        color = next(ax_traj._get_lines.prop_cycler)["color"]
        plot.traj(ax_traj, plot_mode, trajectory, color=color, alpha=0.5)
    plot.draw_coordinate_axes(ax_traj, trajectory, plot_mode, marker_scale=marker_scale)
    plot_collection.add_figure(f"axes_{axes}", fig_traj)


def draw_plans(trajectory, save_path=None, marker_scale=0, colormap: bool = False):
    plot_collection = plot.PlotCollection("evo_traj - trajectory plot")

    # XYZ
    draw_traj(trajectory, "xyz", marker_scale, plot_collection, colormap)
    # XY
    draw_traj(trajectory, "xy", marker_scale, plot_collection, colormap)
    # XZ
    draw_traj(trajectory, "xz", marker_scale, plot_collection, colormap)
    # YZ
    draw_traj(trajectory, "yz", marker_scale, plot_collection, colormap)

    if save_path is not None:
        plot_collection.export(save_path, confirm_overwrite=True)

    return plot_collection
