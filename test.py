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
from nets.DenseNet56 import DenseNet56
from nets.DenseNet48 import DenseNet48
from nets.DenseNet40 import DenseNet40
from nets.DenseNet32 import DenseNet32
from nets.DenseNet24 import DenseNet24
from nets.DenseNet16 import DenseNet16
from nets.DenseNet8 import DenseNet8
from Utils.Tester import Tester
from data_process.data import Data


if __name__ == "__main__":
    input_size = 56
    model = DenseNet56(train_summary_dir="summary/train/" + str(input_size),
                       test_summary_dir="summary/test/" + str(input_size), training=False)
    test_data = Data("128_128jpg/test", model_input_size=input_size, input_window=input_size, is_shuffle_data=False)
    tester = Tester(model, test_data)
    tester.test(batch_size=32, restore_checkpoint="checkpoint/" + str(input_size))
