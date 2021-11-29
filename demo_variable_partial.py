import libgcc_fix  # isort: skip

import importlib
import math
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from tensorpack import MapData, dataflow

from data.pcn_data import (
    NUM_GT_POINTS,
    PCNShapeNetDataset,
    PointCloudDataEntry,
    iter_synset,
)
from data_util import filter_pcd_by_plane, get_min_max
from demo import plot_pcd
from tf_util import chamfer, earth_mover


def plot_line_graph(
    xs: List[List[float]], ys: List[Union[list, np.ndarray]], labels: List[str]
) -> Figure:
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111)
    for x, y, label in zip(xs, ys, labels):
        ax.scatter(x, y, label=label)
    ax.set_xlabel("Percent Shown")
    ax.set_ylabel("Distance")
    ax.set_title("Percentage Shown vs Distance")
    ax.legend()
    return fig


def create_plots(
    partial: np.ndarray,
    complete: np.ndarray,
    ground_truth: np.ndarray,
    suptitle: Optional[str] = None,
) -> Figure:
    fig: Figure = plt.figure(figsize=(8, 4))
    plot_pcd(
        fig.add_subplot(131, projection="3d"),
        partial,
        title=f"Partial ({partial.shape[0]})",
    )
    plot_pcd(
        fig.add_subplot(132, projection="3d"),
        complete,
        title=f"Completed ({complete.shape[0]})",
    )
    plot_pcd(
        fig.add_subplot(133, projection="3d"),
        ground_truth,
        title=f"Ground Truth ({ground_truth.shape[0]})",
    )
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1
    )
    if suptitle is not None:
        fig.suptitle(suptitle)
    return fig


def main(
    model_type: str = "pcn_cd",
    checkpoint: str = "data/trained_models/pcn_cd",
    output_dir: str = "results",
) -> None:
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, NUM_GT_POINTS, 3))
    npts = tf.placeholder(tf.int32, (1,))
    model_module = importlib.import_module(".%s" % model_type, "models")
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))
    cd_op = chamfer(model.outputs, gt)
    emd_op = earth_mover(model.outputs, gt)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    df: MapData = dataflow.LMDBSerializer.load(
        PCNShapeNetDataset.get_lmdb_filepath(), shuffle=False
    )
    data_entry: PointCloudDataEntry = next(
        iter(iter_synset(df, name_substr="9ee32f514a4ee4a043c34c097f2ab3af"))
    )

    min_max = get_min_max(data_entry.partial, "x")
    step = 5
    range_min = math.floor(min_max[0] * 100)
    range_max = math.ceil(min_max[1] * 100) + step
    percentages_shown: List[float] = []
    cds: List[float] = []
    emds: List[float] = []
    for threshold in [i / 100 for i in range(range_min, range_max, step)]:
        partial = filter_pcd_by_plane(data_entry.partial, "x", threshold)
        if partial.shape[0] <= 0:
            # Skip analysis if input point cloud is empty
            continue
        complete: np.ndarray = sess.run(
            model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]}
        )[0]
        cd, emd = sess.run(
            [cd_op, emd_op],
            feed_dict={model.outputs: [complete], gt: [data_entry.ground_truth]},
        )

        percent_shown: float = partial.shape[0] / data_entry.partial.shape[0]
        percentages_shown.append(percent_shown), cds.append(cd), emds.append(emd)

        fig_title = (
            f"Percent Shown: {percent_shown * 100:.1f}%, "
            f"Chamfer Distance: {cd:.4f}, Earth Mover Distance: {emd:.4f}"
        )
        fig = create_plots(
            partial,
            complete,
            data_entry.ground_truth,
            suptitle=fig_title,
        )
        fig.savefig(f"{output_dir}/{fig_title}.png")
    fig = plot_line_graph([percentages_shown] * 2, [cds, emds], ["CD", "EMD"])
    fig.savefig(f"{output_dir}/summary.png")


if __name__ == "__main__":
    main()
