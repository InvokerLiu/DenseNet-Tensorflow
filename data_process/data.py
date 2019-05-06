#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : data.py
#   Author      : LiuBo
#   Created date: 2019-04-25 09:21
#   Description :
#
# ================================================================

# -*-coding:utf-8-*-
import numpy as np
import os
from PIL import Image


class Data(object):
    def __init__(self, image_dir, channel_num=3, model_input_size=224, num_of_classes=6, is_shuffle_data=True,
                 input_window=128, one_hot=True):
        self.image_dir = image_dir
        self.is_shuffle_data = is_shuffle_data
        self.input_window = input_window
        self.one_hot = one_hot
        self.channel_num = channel_num
        self.model_input_size = model_input_size
        self.num_of_classes = num_of_classes
        self.file_names = list()
        self.ground_truth = list()
        self.current_index = 0
        self.__read_data__()
        self.data_size = len(self.file_names)
        if self.is_shuffle_data is True:
            self.__shuffle_data__()

    def __read_data__(self):
        for i in range(self.num_of_classes):
            temp_labels_data = []
            for k in range(self.num_of_classes):
                if k == i:
                    temp_labels_data.append(1)
                else:
                    temp_labels_data.append(0)
            temp_dir = self.image_dir + "/" + str(i + 1)
            for file in os.listdir(temp_dir):
                filename = temp_dir + "/" + file
                if os.path.isdir(filename):
                    continue
                extension = os.path.splitext(filename)
                if extension[-1] != ".jpg":
                    continue
                self.file_names.append(filename)
                if self.one_hot is True:
                    self.ground_truth.append(temp_labels_data)
                else:
                    self.ground_truth.append(i + 1)

    def __shuffle_data__(self):
        permutation = np.random.permutation(len(self.file_names))
        shuffle_files = [self.file_names[x] for x in permutation]
        shuffle_label = np.array(self.ground_truth)[permutation, :]
        self.file_names = shuffle_files
        self.ground_truth = shuffle_label

    def next_batch(self, batch_size):
        size = batch_size
        if self.current_index + batch_size >= len(self.file_names):
            size = len(self.file_names) - self.current_index
        input_data = list()
        labels = list()
        for i in range(self.current_index, self.current_index + size):
            image = Image.open(self.file_names[i])
            data = np.array(image)
            data_size = len(data)
            begin = int((data_size - self.input_window) / 2)
            sub_data = data[begin: begin + self.input_window]
            sub_data = sub_data[:, begin: begin + self.input_window]
            image = Image.fromarray(sub_data)
            image = image.resize((self.model_input_size, self.model_input_size), Image.BILINEAR)
            temp_data = np.array(image)
            input_data.append(temp_data)
            labels.append(self.ground_truth[i])

        if self.current_index + batch_size >= len(self.file_names):
            self.current_index = 0
        else:
            self.current_index += batch_size

        return np.array(input_data), np.array(labels)


if __name__ == "__main__":
    train = Data("../128_128jpg/train")
    test = Data("../128_128jpg/test")
    train_data = train.next_batch(32)
    print(train_data[0].shape)
