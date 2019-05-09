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
from nets.DenseNet64 import DenseNet64
from nets.DenseNet56 import DenseNet56
from nets.DenseNet48 import DenseNet48
from nets.DenseNet40 import DenseNet40
from nets.DenseNet32 import DenseNet32
from nets.DenseNet24 import DenseNet24
from nets.DenseNet16 import DenseNet16
from nets.DenseNet8 import DenseNet8
from Utils.Trainer import Trainer
from data_process.data import Data
import os


if __name__ == "__main__":
    input_size = 48
    train_summary_dir = "summary/train/" + str(input_size)
    if os.path.exists(train_summary_dir) is False:
        os.makedirs(train_summary_dir)
    test_summary_dir = "summary/test/" + str(input_size)
    if os.path.exists(test_summary_dir) is False:
        os.makedirs(test_summary_dir)
    checkpoint = "checkpoint/" + str(input_size)
    if os.path.exists(checkpoint) is False:
        os.makedirs(checkpoint)
    model = DenseNet48(train_summary_dir=train_summary_dir,
                       test_summary_dir=test_summary_dir)
    train_data = Data("128_128jpg/train", model_input_size=input_size, input_window=input_size)
    test_data = Data("128_128jpg/test", model_input_size=input_size, input_window=input_size, is_shuffle_data=False)
    trainer = Trainer(model, train_data, test_data)
    trainer.train(epochs=100, batch_size=32, learn_rate=0.0001, keep_prob=1.0,
                  early_stopping_n=5, checkpoint=checkpoint + "/")
