#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : Tester.py
#   Author      : LiuBo
#   Created date: 2019-05-08 19:00
#   Description :
#
# ================================================================
import tensorflow as tf
import time
import os


class Tester(object):
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def test(self, batch_size=16, restore_checkpoint=""):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        test_data_size = self.test_data.data_size
        n_batch_test = test_data_size // batch_size
        if n_batch_test * batch_size != test_data_size:
            n_batch_test += 1
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            self.model.restore(sess, tf.train.latest_checkpoint(restore_checkpoint))
            print("[*] Model initialized!")
            mean_test_acc = 0
            mean_test_loss = 0
            test_time = 0
            for i in range(n_batch_test):
                x, y = self.test_data.next_batch(batch_size)
                x = x / 127.5 - 1
                begin_time = time.time()
                acc, loss = self.model.eval_op(sess, x, y)
                end_time = time.time()
                mean_test_acc += acc * x.shape[0]
                mean_test_loss += loss * x.shape[0]
                test_time += (end_time - begin_time)
                print("\t\r iteration %3d/%3d: \t loss = %.4f, acc = %.4f, time = %.4f"
                      % (i + 1, n_batch_test, loss, acc, end_time - begin_time),
                      flush=True, end="")
            mean_test_acc /= test_data_size
            mean_test_loss /= test_data_size
            print("\n[*] Testing done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
                  % (mean_test_loss, mean_test_acc, test_time), flush=True)
            return mean_test_acc, mean_test_loss
