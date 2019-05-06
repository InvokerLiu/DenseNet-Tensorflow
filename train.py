#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : train.py
#   Author      : LiuBo
#   Created date: 2019-04-24 23:08
#   Description :
#
# ================================================================

from TinyDenseNet import TinyDenseNet
import Utils.config as cf
import tensorflow as tf
import time
import os
import numpy
from data_process.data import Data

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[*] Model begin build!")
    dense_net = TinyDenseNet(reduction=cf.reduction, class_num=cf.class_num, name=cf.name_of_net,
                             train_summary_dir=cf.train_summary_dir, test_summary_dir=cf.test_summary_dir)
    print("[*] Model built!")
    #
    train_dataset = Data("128_128jpg/train", input_window=40, model_input_size=40)
    test_dataset = Data("128_128jpg/test", is_shuffle_data=False, input_window=40, model_input_size=40)
    #
    train_data_size = train_dataset.data_size
    test_data_size = test_dataset.data_size

    n_batch_train = train_data_size // cf.batch_size
    n_batch_test = test_data_size // cf.batch_size
    if n_batch_train * cf.batch_size != train_data_size:
        n_batch_train += 1
    if n_batch_test * cf.batch_size != test_data_size:
        n_batch_test += 1
    early_stopping_cnt = 0

    best_test_acc = 0

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        if cf.is_restore:
            dense_net.restore(sess, cf.restore_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
        print("[*] Model initialized!")

        for epoch in range(cf.epochs):
            print("[*] Epoch %3d/%3d, Training start..." % (epoch + 1, cf.epochs), flush=True)
            mean_train_acc = 0
            mean_train_loss = 0
            train_time = 0
            for i in range(n_batch_train):
                iteration = epoch * n_batch_train + i
                x, y = train_dataset.next_batch(cf.batch_size)
                x = x / 127.5 - 1
                begin_time = time.time()
                acc, loss = dense_net.train_op(sess, x, y, iteration, cf.learn_rate, cf.keep_prob)
                end_time = time.time()
                mean_train_acc += acc * x.shape[0]
                mean_train_loss += loss * x.shape[0]
                train_time += (end_time - begin_time)
                if numpy.isnan(loss):
                    print("[*] NaN Stopping!", flush=True)
                    exit(-1)

                print("\t\r epoch %3d/%3d, iteration %3d/%3d: \t loss = %.4f, acc = %.4f, time = %.4f"
                      % (epoch + 1, cf.epochs, i + 1, n_batch_train, loss, acc, end_time - begin_time),
                      flush=True, end="")
            mean_train_loss /= train_data_size
            mean_train_acc /= train_data_size
            print("[*] Epoch %3d/%3d, Training done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
                  % (epoch + 1, cf.epochs, mean_train_loss, mean_train_acc, train_time), flush=True)

            print("[*] Epoch %3d/%3d, Testing start..." % (epoch + 1, cf.epochs), flush=True)
            mean_test_acc = 0
            mean_test_loss = 0
            test_time = 0
            for i in range(n_batch_test):
                iteration = epoch * n_batch_train + i
                x, y = test_dataset.next_batch(cf.batch_size)
                x = x / 127.5 - 1
                begin_time = time.time()
                acc, loss = dense_net.eval_op(sess, x, y, iteration)
                end_time = time.time()
                mean_test_acc += acc * x.shape[0]
                mean_test_loss += loss * x.shape[0]
                test_time += (end_time - begin_time)
            mean_test_acc /= test_data_size
            mean_test_loss /= test_data_size
            print("[*] Epoch %3d/%3d, Testing done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
                  % (epoch + 1, cf.epochs, mean_test_loss, mean_test_acc, test_time), flush=True)
            if mean_test_acc >= best_test_acc:
                best_test_acc = mean_test_acc
                early_stopping_cnt = 0
                print("[*] Best test accuracy so far!")
                dense_net.save(sess, cf.checkpoint, global_step=epoch + 1)
                print("[*] Model saved at", cf.checkpoint, flush=True)
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt >= cf.early_stopping_n:
                    print("[*] Early Stopping!", flush=True)
                    exit(-1)
