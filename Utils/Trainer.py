#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : Trainer.py
#   Author      : LiuBo
#   Created date: 2019-05-08 19:00
#   Description :
#
# ================================================================
import tensorflow as tf
import time
import os
import numpy


class Trainer(object):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def train(self, epochs=100, batch_size=16, learn_rate=0.0001, keep_prob=1.0,
              early_stopping_n=5, checkpoint="", restore_checkpoint=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        train_data_size = self.train_data.data_size
        test_data_size = self.test_data.data_size

        n_batch_train = train_data_size // batch_size
        n_batch_test = test_data_size // batch_size
        if n_batch_train * batch_size != train_data_size:
            n_batch_train += 1
        if n_batch_test * batch_size != test_data_size:
            n_batch_test += 1
        early_stopping_cnt = 0
        best_test_acc = 0
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            if restore_checkpoint is not None:
                self.model.restore(sess, restore_checkpoint)
            else:
                sess.run(tf.global_variables_initializer())
            print("[*] Model initialized!")
            for epoch in range(epochs):
                print("[*] Epoch %3d/%3d, Training start..." % (epoch + 1, epochs), flush=True)
                mean_train_acc = 0
                mean_train_loss = 0
                train_time = 0
                for i in range(n_batch_train):
                    iteration = epoch * n_batch_train + i
                    x, y = self.train_data.next_batch(batch_size)
                    x = x / 127.5 - 1
                    begin_time = time.time()
                    acc, loss = self.model.train_op(sess, x, y, iteration, learn_rate, keep_prob)
                    end_time = time.time()
                    mean_train_acc += acc * x.shape[0]
                    mean_train_loss += loss * x.shape[0]
                    train_time += (end_time - begin_time)
                    if numpy.isnan(loss):
                        print("[*] NaN Stopping!", flush=True)
                        exit(-1)

                    print("\t\r epoch %3d/%3d, iteration %3d/%3d: \t loss = %.4f, acc = %.4f, time = %.4f"
                          % (epoch + 1, epochs, i + 1, n_batch_train, loss, acc, end_time - begin_time),
                          flush=True, end="")
                mean_train_loss /= train_data_size
                mean_train_acc /= train_data_size
                print("[*] Epoch %3d/%3d, Training done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
                      % (epoch + 1, epochs, mean_train_loss, mean_train_acc, train_time), flush=True)

                print("[*] Epoch %3d/%3d, Testing start..." % (epoch + 1, epochs), flush=True)
                mean_test_acc = 0
                mean_test_loss = 0
                test_time = 0
                for i in range(n_batch_test):
                    iteration = epoch * n_batch_train + i
                    x, y = self.test_data.next_batch(batch_size)
                    x = x / 127.5 - 1
                    begin_time = time.time()
                    acc, loss = self.model.eval_op(sess, x, y, iteration)
                    end_time = time.time()
                    mean_test_acc += acc * x.shape[0]
                    mean_test_loss += loss * x.shape[0]
                    test_time += (end_time - begin_time)
                mean_test_acc /= test_data_size
                mean_test_loss /= test_data_size
                print("[*] Epoch %3d/%3d, Testing done!\n\t loss = %.4f, acc = %.4f, time = %.4f"
                      % (epoch + 1, epochs, mean_test_loss, mean_test_acc, test_time), flush=True)
                if mean_test_acc >= best_test_acc:
                    best_test_acc = mean_test_acc
                    early_stopping_cnt = 0
                    print("[*] Best test accuracy so far!")
                    self.model.save(sess, checkpoint, global_step=epoch + 1)
                    print("[*] Model saved at", checkpoint, flush=True)
                else:
                    early_stopping_cnt += 1
                    if early_stopping_cnt >= early_stopping_n:
                        print("[*] Early Stopping!", flush=True)
                        return
