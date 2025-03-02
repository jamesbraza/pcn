"""
MIT License.

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def plot_pcd_three_views(
    filename: str,
    pcds: Sequence[np.ndarray],
    titles: Sequence[str],
    suptitle="",
    sizes: Optional[List[float]] = None,
    cmap="Reds",
    zdir="y",
    xlim=(-0.3, 0.3),
    ylim=(-0.3, 0.3),
    zlim=(-0.3, 0.3),
):
    """Save three views of the point clouds with the specified titles to the input filename."""
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig: Figure = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax: Axes3D = fig.add_subplot(
                3, len(pcds), i * len(pcds) + j + 1, projection="3d"
            )
            ax.view_init(elev, azim)
            ax.scatter(
                pcd[:, 0],
                pcd[:, 1],
                pcd[:, 2],
                zdir=zdir,
                c=color,
                s=size,
                cmap=cmap,
                vmin=-1,
                vmax=0.5,
            )
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1
    )
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def show_pcd(points: np.ndarray) -> None:
    """Visualize a point cloud in an interactive window."""
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(points)
    except ValueError:
        # SEE: https://github.com/isl-org/Open3D/issues/2557#issuecomment-850853083
        pcd.points = o3d.utility.Vector3dVector(deepcopy(points))
    o3d.visualization.draw_geometries([pcd])
