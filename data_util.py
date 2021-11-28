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
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorpack import DataFlow, dataflow


def resample_pcd(pcd: np.ndarray, n: int) -> np.ndarray:
    """Drop or duplicate points so that the input point cloud has exactly n points."""
    num_points: int = pcd.shape[0]
    idx = np.random.permutation(num_points)  # Indices to keep
    if idx.shape[0] < n:
        # If we aren't keeping enough indices, randomly select more
        idx = np.concatenate([idx, np.random.randint(num_points, size=n - num_points)])
    # Drop excess points, returning the exact amount requested
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):  # noqa: D101
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    """
    Modified version of tensorpack.dataflow.BatchData.

    Why does this exist?  It seems the author didn't want to deal with
    subclassing complexity, so just copied and the manually "overrode".
    """

    def __init__(
        self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False
    ):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        """Concatenate input points along the 0-th dimension Stack all other data along the 0-th dimension."""
        ids = np.stack([x[0] for x in data_holder])
        inputs = [
            resample_pcd(x[1], self.input_size)
            if x[1].shape[0] > self.input_size
            else x[1]
            for x in data_holder
        ]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(
            np.float32
        )
        npts = np.stack(
            [
                x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
                for x in data_holder
            ]
        ).astype(np.int32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(
            np.float32
        )
        return ids, inputs, npts, gts


def lmdb_dataflow(
    lmdb_path,
    batch_size: int,
    input_size: int,
    output_size: int,
    is_training: bool,
    test_speed: bool = False,
) -> Tuple[DataFlow, int]:
    df: DataFlow = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size


def get_queued_data(generator, dtypes, shapes, queue_capacity=10):
    assert len(dtypes) == len(shapes), "dtypes and shapes must have the same length"
    queue = tf.FIFOQueue(queue_capacity, dtypes, shapes)
    placeholders = [
        tf.placeholder(dtype, shape) for dtype, shape in zip(dtypes, shapes)
    ]
    enqueue_op = queue.enqueue(placeholders)
    close_op = queue.close(cancel_pending_enqueues=True)
    feed_fn = lambda: {  # noqa: E731
        placeholder: value for placeholder, value in zip(placeholders, next(generator))
    }
    queue_runner = tf.contrib.training.FeedingQueueRunner(
        queue, [enqueue_op], close_op, feed_fns=[feed_fn]
    )
    tf.train.add_queue_runner(queue_runner)
    return queue.dequeue()
