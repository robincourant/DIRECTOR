from typing import List

import matplotlib.cm as cm
from evo.core.trajectory import PosePath3D
from evo.tools import plot
from evo.tools.settings import SETTINGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {"Greens": "#4daf4a", "Blues": "#377eb8", "Reds": "#e41a1c"}


def draw_trajectories(
    trajectories: PosePath3D,
    save_path: str = None,
    marker_scale: float = 0,
    colormaps: List[str] = None,
) -> plot.PlotCollection:
    plot_collection = plot.PlotCollection("evo_traj - trajectory plot")
    figsize = tuple(SETTINGS.plot_figsize)
    fig_xyz, axarr_xyz = plt.subplots(3, sharex="col", figsize=figsize)
    fig_rpy, axarr_rpy = plt.subplots(3, sharex="col", figsize=figsize)
    fig_traj = plt.figure(figsize=figsize)
    plot_mode = plot.PlotMode[SETTINGS.plot_mode_default]
    ax_traj = plot.prepare_axis(fig_traj, plot_mode)

    # Set axis limits
    xyz = np.concatenate([traj.positions_xyz for traj in trajectories.values()])
    min_x, min_y, min_z = np.min(xyz, axis=0)
    max_x, max_y, max_z = np.max(xyz, axis=0)
    ax_traj.set_xlim(min_x, max_x)
    ax_traj.set_ylim(min_y, max_y)
    ax_traj.set_zlim(min_z, max_z)

    for k, (name, traj) in enumerate(trajectories.items()):
        color = COLORS[colormaps[k]] if colormaps is not None else None
        plot.traj_xyz(axarr_xyz, traj, color=color, label=name, alpha=0.5)
        plot.traj_rpy(axarr_rpy, traj, color=color, label=name, alpha=0.5)
        if colormaps is not None:
            plot_traj_colormap(ax_traj, traj, plot_mode, colormaps[k])
        else:
            plot.traj(ax_traj, plot_mode, traj, color=color, label=name, alpha=0.5)

        plot.draw_coordinate_axes(ax_traj, traj, plot_mode, marker_scale=marker_scale)

    plot_collection.add_figure("trajectories", fig_traj)
    plot_collection.add_figure("xyz_view", fig_xyz)
    plot_collection.add_figure("rpy_view", fig_rpy)
    if save_path is not None:
        plot_collection.export(save_path, confirm_overwrite=True)

    return plot_collection


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
