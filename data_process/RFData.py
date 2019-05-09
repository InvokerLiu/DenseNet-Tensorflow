#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : RFData.py
#   Author      : LiuBo
#   Created date: 2019-05-09 09:21
#   Description :
#
# ================================================================

import numpy as np
import struct


class Data(object):
    def __init__(self):
        self.feature = list()
        self.label = list()


class RFData(object):
    def __init__(self):
        self.train = Data()
        self.test = Data()

    def get_data_from_sample(self, train_sample_file, test_sample_file,
                             feature_file_list, class_num, feature_dimension):
        train_list = list()
        with open(train_sample_file, "r") as f:
            line = f.readline()
            line = line.strip('\n')
            temp_string = line.split(' ')
            for i in range(class_num):
                temp_list = list()
                for j in range(int(temp_string[i])):
                    line = f.readline()
                    line = line.strip('\n')
                    line = line.split(' ')
                    temp_list.append(int(line[0]))
                train_list.append(temp_list)
        test_list = list()
        with open(test_sample_file, "r") as f:
            line = f.readline()
            line = line.strip('\n')
            temp_string = line.split(' ')
            for i in range(class_num):
                temp_list = list()
                for j in range(int(temp_string[i])):
                    line = f.readline()
                    line = line.strip('\n')
                    line = line.split(' ')
                    temp_list.append(int(line[0]))
                test_list.append(temp_list)
        read_format = str(feature_dimension) + "f"
        for i in range(class_num):
            for j in range(len(train_list[i])):
                index = train_list[i][j]
                feature_data = None
                for k in range(len(feature_file_list)):
                    with open(feature_file_list[k], "rb") as f:
                        f.seek(index * feature_dimension * 4)
                        buf = f.read(feature_dimension * 4)
                        data = struct.unpack(read_format, buf)
                        if k == 0:
                            feature_data = np.array(data)
                        else:
                            feature_data = np.append(feature_data, data)
                self.train.feature.append(feature_data)
                self.train.label.append(i)

            for j in range(len(test_list[i])):
                index = test_list[i][j]
                feature_data = None
                for k in range(len(feature_file_list)):
                    with open(feature_file_list[k], "rb") as f:
                        f.seek(index * feature_dimension * 4)
                        buf = f.read(feature_dimension * 4)
                        data = struct.unpack(read_format, buf)
                        if k == 0:
                            feature_data = np.array(data)
                        else:
                            feature_data = np.append(feature_data, data)
                self.test.feature.append(feature_data)
                self.test.label.append(i)
        permutation = np.random.permutation(len(self.train.feature))
        shuffle_data = np.array(self.train.feature)[permutation]
        shuffle_label = np.array(self.train.label)[permutation]
        self.train.feature = shuffle_data
        self.train.label = shuffle_label

    def get_data_from_sample_txt(self, train_sample_file, test_sample_file,
                                 feature_file_list, class_num, feature_dimension_list):
        train_list = list()
        with open(train_sample_file, "r") as f:
            line = f.readline()
            line = line.strip('\n')
            temp_string = line.split(' ')
            for i in range(class_num):
                temp_list = list()
                for j in range(int(temp_string[i])):
                    line = f.readline()
                    line = line.strip('\n')
                    line = line.split(' ')
                    temp_list.append(int(line[0]))
                train_list.append(temp_list)
        test_list = list()
        with open(test_sample_file, "r") as f:
            line = f.readline()
            line = line.strip('\n')
            temp_string = line.split(' ')
            for i in range(class_num):
                temp_list = list()
                for j in range(int(temp_string[i])):
                    line = f.readline()
                    line = line.strip('\n')
                    line = line.split(' ')
                    temp_list.append(int(line[0]))
                test_list.append(temp_list)
        lines_list = list()
        for k in range(len(feature_file_list)):
            with open(feature_file_list[k]) as f:
                lines = f.readlines()
                lines_list.append(lines)
        for i in range(class_num):
            for j in range(len(train_list[i])):
                index = train_list[i][j]
                feature_data = None
                for k in range(len(feature_file_list)):
                    temp_string = lines_list[k][index]
                    temp_string = temp_string.strip('\n')
                    temp_string = temp_string.split(' ')
                    data = list()
                    for n in range(feature_dimension_list[k]):
                        data.append(float(temp_string[n]))
                    if k == 0:
                        feature_data = np.array(data)
                    else:
                        feature_data = np.append(feature_data, data)
                self.train.feature.append(feature_data)
                self.train.label.append(i)

            for j in range(len(test_list[i])):
                index = test_list[i][j]
                feature_data = None
                for k in range(len(feature_file_list)):
                    temp_string = lines_list[k][index]
                    temp_string = temp_string.strip('\n')
                    temp_string = temp_string.split(' ')
                    data = list()
                    for n in range(feature_dimension_list[k]):
                        data.append(float(temp_string[n]))
                    if k == 0:
                        feature_data = np.array(data)
                    else:
                        feature_data = np.append(feature_data, data)
                self.test.feature.append(feature_data)
                self.test.label.append(i)
        permutation = np.random.permutation(len(self.train.feature))
        shuffle_data = np.array(self.train.feature)[permutation]
        shuffle_label = np.array(self.train.label)[permutation]
        self.train.feature = shuffle_data
        self.train.label = shuffle_label

    def clear_data(self):
        self.train.feature = list()
        self.train.label = list()
        self.test.feature = list()
        self.test.label = list()
