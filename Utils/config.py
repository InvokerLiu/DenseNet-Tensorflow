#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : layers.py
#   Author      : LiuBo
#   Created date: 2019-04-24 23:00
#   Description :
#
# ================================================================
# dense net reduction
reduction = 0.0

# number of classes
class_num = 6

# number of epoch
epochs = 1000

# name of neural network
name_of_net = "dense_net"

# dropout keep probability
keep_prob = 1.0

# leaning rate
learn_rate = 0.0001

# batch size
batch_size = 16

# directory of summary
train_summary_dir = "summary/train"
test_summary_dir = "summary/test"

# save and restore checkpoint
checkpoint = "checkpoint/dense_net"
restore_checkpoint = "checkpoint/dense_net-1"

early_stopping_n = 5

is_restore = False



