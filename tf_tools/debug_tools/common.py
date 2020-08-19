#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from tf_tools.config import SummaryConfig


def session_run(outputs, feed_dict):
    # print(f"outputs: {outputs}")
    # print(f"feed_dict: {feed_dict}")
    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        with tf.summary.FileWriter(SummaryConfig.outputs_path, sess.graph):
            sess.run(tf.global_variables_initializer())
            ret = sess.run(outputs, feed_dict=feed_dict)
    return ret


if __name__ == '__main__':
    pass
