#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : DenseNet64.py
#   Author      : LiuBo
#   Created date: 2019-05-08 19:40
#   Description :
#
# ================================================================

from core.layers import conv2d, avg_pool, fc_connect
from core.blocks import dense_block, transition_block
import tensorflow as tf


class DenseNet64(object):
    def __init__(self, dense_block_num=4, growth_rate=12, filter_num=32, reduction=0.0,
                 class_num=6, input_size=64, channels=3, training=True, name="dense_net64",
                 train_summary_dir="", test_summary_dir=""):
        """
        :param dense_block_num:
        :param growth_rate:
        :param filter_num:
        :param reduction:
        :param class_num:
        :param input_size:
        :param channels:
        :param training:
        :param name:
        :param train_summary_dir:
        :param test_summary_dir:
        """
        self.dense_block_num = dense_block_num
        self.growth_rate = growth_rate
        self.filter_num = filter_num
        self.reduction = reduction
        self.class_num = class_num
        self.input_size = input_size
        self.channels = channels
        self.layers_num = [3, 6, 9, 4]
        self.compression = 1 - reduction
        self.training = training
        self.deep_features = None
        with tf.variable_scope(name):
            self.__add_placeholders()
            self.__Y = self.__build_graph()

            if training:
                self.__loss, self.__accuracy, self.__train_step = self.__build_training_graph()
                tf.summary.scalar("loss", self.__loss)
                tf.summary.scalar("accuracy", self.__accuracy)
                self.__writer_train = tf.summary.FileWriter(train_summary_dir)
                self.__writer_test = tf.summary.FileWriter(test_summary_dir)
                self.__write_op = tf.summary.merge_all()
            else:
                self.__loss, self.__accuracy, self.__train_step = self.__build_inference_graph()
            self.__saver = tf.train.Saver(max_to_keep=3)

    def __add_placeholders(self):
        """
        :return:
        """
        self.__X = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.channels], name="input_X")
        self.__Y_ = tf.placeholder(tf.float32, [None, self.class_num], name="ground_truth")
        self.__is_training = tf.placeholder(tf.bool, name="is_training")
        self.__iter = tf.placeholder(tf.int32, name="iteration")
        self.__keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.__lr = tf.placeholder(tf.float32, name="learn_rate")

    def __build_graph(self):
        """
        :return:
        """
        x = conv2d(self.__X, kernel_size=3, stride=1, filter_num=self.filter_num, padding="SAME", name="conv1")
        x = tf.layers.batch_normalization(x, training=self.__is_training)
        x = tf.nn.relu(x)

        for block_idx in range(self.dense_block_num - 1):
            dense_block_name = "dense_block" + str(block_idx + 1)
            transition_block_name = "transition_block" + str(block_idx + 1)
            x, filter_num = dense_block(x, self.layers_num[block_idx], self.filter_num,
                                        self.growth_rate, training=self.__is_training,
                                        name=dense_block_name, keep_prob=self.__keep_prob)
            x = transition_block(x, self.filter_num, training=self.__is_training,
                                 name=transition_block_name, keep_prob=self.__keep_prob,
                                 compression=self.compression)

            self.filter_num = int(self.filter_num * self.compression)
        dense_block_name = "dense_block" + str(self.dense_block_num)
        x = conv2d(x, kernel_size=3, stride=1, filter_num=self.filter_num, padding="SAME", name="conv2")
        x = tf.layers.batch_normalization(x, training=self.__is_training)
        x = tf.nn.relu(x)
        x, filter_num = dense_block(x, self.layers_num[-1], self.filter_num, self.growth_rate,
                                    training=self.__is_training, name=dense_block_name,
                                    keep_prob=self.__keep_prob)

        x = tf.layers.batch_normalization(x, training=self.__is_training)
        x = tf.nn.relu(x)
        x = conv2d(x, kernel_size=3, stride=1, filter_num=self.filter_num, padding="SAME", name="conv3")
        x = tf.layers.batch_normalization(x, training=self.__is_training)
        x = tf.nn.relu(x)
        self.deep_features = x
        pool_size = x.get_shape()[1]
        x = avg_pool(x, kernel_size=pool_size, stride=1, padding="VALID", name="avg_pool")

        input_dim = x.get_shape()[-1]
        x = tf.reshape(x, [-1, input_dim])

        x = fc_connect(x, self.class_num, name="fc_connect")
        x = tf.nn.softmax(x)
        return x

    def __build_training_graph(self):
        """
        :return:
        """
        loss = tf.reduce_mean(-tf.reduce_sum(self.__Y_ * tf.log(self.__Y), reduction_indices=[1]))
        correct_prediction = tf.equal(tf.argmax(self.__Y, 1), tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        opt = tf.train.AdamOptimizer(learning_rate=self.__lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = opt.minimize(loss)
        return loss, accuracy, train_step

    def __build_inference_graph(self):
        """
        :return:
        """
        loss = tf.reduce_mean(-tf.reduce_sum(self.__Y_ * tf.log(self.__Y), reduction_indices=[1]))
        correct_prediction = tf.equal(tf.argmax(self.__Y, 1), tf.argmax(self.__Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return loss, accuracy, None

    def train_op(self, sess, x, y, iteration, learn_rate, keep_prob=1.0):
        """
        :param sess:
        :param x:
        :param y:
        :param iteration:
        :param learn_rate:
        :param keep_prob:
        :return:
        """
        _, write_data, acc, loss = sess.run([self.__train_step, self.__write_op, self.__accuracy, self.__loss],
                                            feed_dict={self.__X: x,
                                                       self.__Y_: y,
                                                       self.__iter: iteration,
                                                       self.__is_training: True,
                                                       self.__lr: learn_rate,
                                                       self.__keep_prob: keep_prob})
        self.__writer_train.add_summary(write_data, global_step=iteration)
        self.__writer_train.flush()
        return acc, loss

    def eval_op(self, sess, x, y, iteration=None):
        """
        :param sess:
        :param x:
        :param y:
        :param iteration: the number of iteration, if train, iteration is not None, if inference, iteration is None
        :return:
        """
        if iteration is None:
            it = 0
        else:
            it = iteration

        acc, loss = sess.run([self.__accuracy, self.__loss],
                             feed_dict={self.__X: x,
                                        self.__Y_: y,
                                        self.__iter: it,
                                        self.__is_training: False,
                                        self.__keep_prob: 1.0})

        if self.training:
            write_data = sess.run(self.__write_op,
                                  feed_dict={self.__X: x,
                                             self.__Y_: y,
                                             self.__iter: it,
                                             self.__is_training: False,
                                             self.__keep_prob: 1.0})
            self.__writer_test.add_summary(write_data, global_step=it)
            self.__writer_test.flush()

        return acc, loss

    def infer_op(self, sess, x):
        """
        :param sess:
        :param x:
        :return:
        """
        y = sess.run(self.__Y, feed_dict={self.__X: x,
                                          self.__iter: 0,
                                          self.__is_training: False,
                                          self.__keep_prob: 1.0})
        return y

    def save(self, sess, path, global_step):
        """
        :param sess:
        :param path:
        :param global_step:
        :return:
        """
        self.__saver.save(sess, path, global_step=global_step)

    def restore(self, sess, path):
        """
        :param sess:
        :param path:
        :return:
        """
        self.__saver.restore(sess, path)

    def get_deep_features(self, sess, input_data):
        deep_features = sess.run(self.deep_features, feed_dict={self.__X: input_data,
                                                                self.__keep_prob: 1.0,
                                                                self.__iter: 0,
                                                                self.__is_training: False})
        return deep_features


if __name__ == "__main__":
    dense_net = DenseNet64()
    variable = [v for v in tf.trainable_variables()]
    parm_cnt = 0
    for v in variable:
        print("   ", v.name, v.get_shape())
        parm_cnt_v = 1
        for i in v.get_shape().as_list():
            parm_cnt_v *= i
        parm_cnt += parm_cnt_v
    print("[*] Model parameter size: %.4fM" % (parm_cnt / 1024 / 1024))
