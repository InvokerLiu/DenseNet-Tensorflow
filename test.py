#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : test.py
#   Author      : LiuBo
#   Created date: 2019-04-24 23:30
#   Description :
#
# ================================================================

import tensorflow as tf
import time
import os
from DenseNet import DenseNet
import Utils.config as cf
from data_process.data import Data


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dense_net = DenseNet(reduction=cf.reduction, class_num=cf.class_num, name=cf.name_of_net,
                         train_summary_dir=cf.train_summary_dir, test_summary_dir=cf.test_summary_dir)
    print("[*] Model built!")
    test_dataset = Data("128_128jpg/test", is_shuffle_data=False)
    test_data_size = test_dataset.data_size
    n_batch_test = test_data_size // cf.batch_size
    if n_batch_test * cf.batch_size != test_data_size:
        n_batch_test += 1
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        dense_net.restore(sess, cf.restore_checkpoint)
        print("[*] Model initialized!")
        mean_test_acc = 0
        mean_test_loss = 0
        test_time = 0
        for i in range(n_batch_test):
            x, y = test_dataset.next_batch(cf.batch_size)
            begin_time = time.time()
            acc, loss = dense_net.eval_op(sess, x, y)
            end_time = time.time()
            mean_test_acc += acc * x.shape[0]
            mean_test_loss += loss * x.shape[0]
            test_time += (end_time - begin_time)
            print("\t\r iteration %3d/%3d: \t loss = %.4f, acc = %.4f, time = %.4f"
                  % (i + 1, n_batch_test, loss, acc, end_time - begin_time),
                  flush=True, end="")
        mean_test_acc /= test_data_size
        mean_test_loss /= test_data_size
        print("[*] Testing done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
              % (mean_test_loss, mean_test_acc, test_time), flush=True)
