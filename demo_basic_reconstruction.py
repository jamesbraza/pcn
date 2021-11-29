import libgcc_fix  # isort: skip

import importlib

import tensorflow as tf
from tensorpack import MapData, dataflow
from vis import (  # From https://github.com/lynetcha/completion3d/blob/master/shared/vis.py#L23
    plot_xyz,
)

from data.pcn_data import (
    NUM_GT_POINTS,
    DataSubset,
    PCNShapeNetDataset,
    PointCloudDataEntry,
)
from tf_util import chamfer, earth_mover


def main(
    model_type: str = "pcn_cd",
    checkpoint: str = "data/trained_models/pcn_cd",
    log_dir: str = "log/basic",
) -> None:
    inputs = tf.placeholder(tf.float32, (1, None, 3), name="partial")
    gt = tf.placeholder(tf.float32, (1, NUM_GT_POINTS, 3), name="ground_truth")
    npts = tf.placeholder(tf.int32, (1,), name="num_points")
    model_module = importlib.import_module(".%s" % model_type, "models")
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))
    cd_fine_op = chamfer(model.outputs, gt)
    cd_coarse_op = chamfer(model.coarse, gt)
    emd_fine_op = earth_mover(model.outputs, gt)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    df: MapData = dataflow.LMDBSerializer.load(
        PCNShapeNetDataset.get_lmdb_filepath(DataSubset.VALIDATION), shuffle=False
    )
    data_entry: PointCloudDataEntry = PCNShapeNetDataset.extract_from_data_buffer(
        next(iter(df))
    )

    partial = data_entry.partial
    fine, coarse = sess.run(
        [model.outputs, model.coarse],
        feed_dict={inputs: [partial], npts: [partial.shape[0]]},
    )
    cd_fine, emd_fine = sess.run(
        [cd_fine_op, emd_fine_op],
        feed_dict={model.outputs: fine, gt: [data_entry.ground_truth]},
    )
    cd_coarse = sess.run(
        cd_coarse_op, feed_dict={model.coarse: coarse, gt: [data_entry.ground_truth]}
    )
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    _ = 0  # Debug here


if __name__ == "__main__":
    main()
