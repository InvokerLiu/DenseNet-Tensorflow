#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : layers.py
#   Author      : LiuBo
#   Created date: 2019-04-24 17:16
#   Description :
#
# ================================================================

import tensorflow as tf


def conv2d(x, kernel_size, stride, filter_num, padding, name):
    """
    :param x: 
    :param kernel_size: 
    :param stride: 
    :param filter_num: 
    :param padding: 
    :param name: 
    :return: 
    """
    with tf.variable_scope(name):
        channel = int(x.get_shape()[-1])
        w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channel, filter_num],
                                            0, 0.1, dtype=tf.float32), name="w")
        b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[filter_num]), name="b")
        out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        out = tf.nn.bias_add(out, b)
        return out


def max_pool(x, kernel_size, stride, padding, name):
    """
    :param x:
    :param kernel_size:
    :param stride:
    :param padding:
    :param name:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def avg_pool(x, kernel_size, stride, padding, name):
    """
    :param x:
    :param kernel_size:
    :param stride:
    :param padding:
    :param name:
    :return:
    """
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding=padding, name=name)


def fc_connect(x, output_dim, name):
    """
    :param x:
    :param output_dim:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        input_dim = int(x.get_shape()[-1])
        w = tf.Variable(tf.truncated_normal([input_dim, output_dim], 0, 0.1, dtype=tf.float32), name="w")
        b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_dim]), name="b")
        out = tf.nn.xw_plus_b(x, w, b)
    return out
