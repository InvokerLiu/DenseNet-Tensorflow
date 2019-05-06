#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : blocks.py
#   Author      : LiuBo
#   Created date: 2019-04-24 18:30
#   Description :
#
# ================================================================

from core.layers import conv2d, avg_pool
import tensorflow as tf


def conv_block(x, filter_num, training, name, keep_prob):
    """
    :param x:
    :param filter_num:
    :param name:
    :param training:
    :param keep_prob:
    :return:
    """
    with tf.variable_scope(name):
        # 1x1 Convolution (Bottleneck layer)
        inter_channel = filter_num * 4
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = conv2d(x, kernel_size=1, stride=1, filter_num=inter_channel, padding="SAME", name="conv1")
        x = tf.nn.dropout(x, keep_prob=keep_prob)

        # 3x3 Convolution
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = conv2d(x, kernel_size=3, stride=1, filter_num=filter_num, padding="SAME", name="conv2")
        x = tf.nn.dropout(x, keep_prob=keep_prob)

        return x


def dense_block(x, layers_num, filter_num, growth_rate, training, name,
                keep_prob, grow_filters_num=True):
    """
    :param x:
    :param layers_num:
    :param filter_num:
    :param growth_rate:
    :param training:
    :param name:
    :param keep_prob:
    :param grow_filters_num:
    :return:
    """
    with tf.variable_scope(name):
        concat_feat = x
        for i in range(layers_num):
            conv_block_name = "conv_block" + str(i + 1)
            x = conv_block(concat_feat, growth_rate, training, name=conv_block_name,
                           keep_prob=keep_prob)
            concat_name = "concat" + str(i + 1)
            concat_feat = tf.concat([concat_feat, x], axis=3, name=concat_name)

            if grow_filters_num:
                filter_num += growth_rate

    return concat_feat, filter_num


def transition_block(x, filter_num, training, name, keep_prob, compression=1.0):
    """
    :param x:
    :param filter_num:
    :param training:
    :param name:
    :param compression:
    :param keep_prob:
    :return:
    """
    with tf.variable_scope(name):
        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = conv2d(x, kernel_size=1, stride=1, filter_num=int(filter_num * compression), padding="SAME", name="conv")

        x = tf.nn.dropout(x, keep_prob=keep_prob)

        x = avg_pool(x, kernel_size=2, stride=2, name="avg_pool", padding="SAME")

        return x
