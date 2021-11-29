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

import numpy as np
import tensorflow as tf
from tensorflow_core import Tensor

from pc_distance import tf_approxmatch, tf_nndistance


def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features,
            num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope="fc_%d" % i,
        )
    outputs = tf.contrib.layers.fully_connected(
        features,
        layer_dims[-1],
        activation_fn=None,
        scope="fc_%d" % (len(layer_dims) - 1),
    )
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs,
            num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope="conv_%d" % i,
        )
    outputs = tf.contrib.layers.conv1d(
        inputs,
        layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope="conv_%d" % (len(layer_dims) - 1),
    )
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    with tf.variable_scope("pointwise_maxpool", reuse=tf.AUTO_REUSE):
        outputs = [
            tf.reduce_max(f, axis=1, keepdims=keepdims)
            for f in tf.split(inputs, npts, axis=1)
        ]
        return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    with tf.variable_scope("pointwise_unpool", reuse=tf.AUTO_REUSE):
        inputs = tf.split(inputs, inputs.shape[0], axis=0)
        outputs = [tf.tile(f, [1, npts[i], 1]) for i, f in enumerate(inputs)]
        return tf.concat(outputs, axis=1)


def chamfer(pcd1: Tensor, pcd2: Tensor) -> Tensor:
    with tf.variable_scope("chamfer", reuse=tf.AUTO_REUSE):
        dist1, _, dist2, __ = tf_nndistance.nn_distance(pcd1, pcd2)
        dist1 = tf.reduce_mean(tf.sqrt(dist1))
        dist2 = tf.reduce_mean(tf.sqrt(dist2))
        return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    with tf.variable_scope("emd", reuse=tf.AUTO_REUSE):
        num_points = tf.cast(pcd1.shape[1], tf.float32)
        match = tf_approxmatch.approx_match(pcd1, pcd2)
        cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
        return tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=["train_summary"])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=["valid_summary"])
    return update
