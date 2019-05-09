#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : FeatureExtractor.py
#   Author      : LiuBo
#   Created date: 2019-05-08 19:00
#   Description :
#
# ================================================================

import tensorflow as tf
import os
from PIL import Image
import numpy
from tqdm import trange
from osgeo import gdal
from osgeo.gdalconst import *


class FeatureExtractor(object):
    """
    Extract deep features of image objects
    """
    def __init__(self, model, padding_size=200):
        """
        :param model: CNN model
        :param padding_size: image padding size
        """
        self.model = model
        self.padding_size = padding_size
        self.image_data = None

    def extract_features(self, window_set_file, result_file, restore_checkpoint, image_file, aggregation_function,
                         out_tag=False):
        """
        :param window_set_file: window set file of image objects
        :param result_file: features result file
        :param restore_checkpoint: checkpoint directory
        :param image_file: jpg image file
        :param aggregation_function: function of aggregation
        :param out_tag: is extract out features of image objects
        :return:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            # restore model
            self.model.restore(sess, tf.train.latest_checkpoint(restore_checkpoint))

            # get image data
            image = Image.open(image_file)
            image_data = numpy.array(image)
            self.image_data = image_data / 127.5 - 1
            with open(window_set_file) as read_f:
                with open(result_file, "w") as write_f:
                    # get number of image objects
                    line = read_f.readline()
                    line = line.strip('\n')
                    object_num = int(line)
                    for _ in trange(object_num):
                        # extract deep features
                        self.__get_object_features(sess, read_f, write_f, aggregation_function, out_tag)

    def __get_object_features(self, sess, read_f, write_f, aggregation_function, out_tag):
        """
        :param sess: session of tensorflow
        :param read_f: window set read file
        :param write_f: features result write file
        :param aggregation_function: aggregation function
        :param out_tag: is extract out features of image objects
        :return:
        """
        # number of grids of the image object
        line = read_f.readline()
        line = line.strip('\n')
        window_num = int(line)
        weight_list = list()
        feature_list = list()
        for i in range(window_num):
            # get the position and the weight of the grid
            line = read_f.readline()
            line = line.strip('\n')
            temp_strings = line.split(' ')
            y_begin = int(temp_strings[0]) + self.padding_size
            x_begin = int(temp_strings[1]) + self.padding_size
            y_size = int(temp_strings[2])
            x_size = int(temp_strings[3])
            weight = float(temp_strings[4])
            weight_list.append(weight)
            # get input data
            input_data = list()
            data = self.image_data[y_begin: y_begin + y_size, x_begin: x_begin + x_size, :]
            input_data.append(data)
            # get deep features
            deep_features = self.model.get_deep_features(sess, input_data)
            deep_features = numpy.array(deep_features[0])
            shape = deep_features.shape
            temp_list = list()
            # deep features aggregation
            for j in range(shape[2]):
                temp_array = deep_features[:, :, j]
                temp = aggregation_function(temp_array)
                temp_list.append(temp)
            temp_list = numpy.array(temp_list)
            feature_list.append(temp_list)
        feature_list = numpy.array(feature_list)
        weight_list = numpy.array(weight_list)
        if out_tag is True:
            weight_list = 1 - weight_list
        assert len(feature_list) == len(weight_list)
        for i in range(len(feature_list)):
            feature_list[i] *= weight_list[i]
        feature_list = numpy.sum(feature_list, axis=0)
        weight_sum = numpy.sum(weight_list)
        assert weight_sum > 0
        feature_list /= weight_sum
        # write result
        for i in range(feature_list.shape[0]):
            write_f.write("%f " % feature_list[i])
        write_f.write("\n")

    def extract_object_features_by_id(self, window_set_file, result_file, restore_checkpoint, image_file, object_id,
                                      coordinate_file="D:/DoLab/Research/Data/WV/10400_10400.tif"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            # restore model
            self.model.restore(sess, tf.train.latest_checkpoint(restore_checkpoint))

            # get image data
            image = Image.open(image_file)
            image_data = numpy.array(image)
            self.image_data = image_data / 127.5 - 1
            with open(window_set_file) as read_f:
                # get number of image objects
                line = read_f.readline()
                line = line.strip('\n')
                object_num = int(line)
                assert object_id < object_num
                for i in range(object_id):
                    line = read_f.readline()
                    line = line.strip('\n')
                    window_num = int(line)
                    for j in range(window_num):
                        read_f.readline()
                self.__extract_object_features_by_id(sess, read_f, result_file, coordinate_file)

    def __extract_object_features_by_id(self, sess, read_f, result_file, coordinate_file):
        line = read_f.readline()
        line = line.strip('\n')
        window_num = int(line)
        x_min = 1e9
        y_min = 1e9
        x_max = 0
        y_max = 0
        y_begin_list = list()
        x_begin_list = list()
        y_size_list = list()
        x_size_list = list()
        input_data = list()
        for j in range(window_num):
            line = read_f.readline()
            line = line.strip("\n")
            temp_strings = line.split(" ")
            y_begin = int(temp_strings[0]) + self.padding_size
            x_begin = int(temp_strings[1]) + self.padding_size
            y_size = int(temp_strings[2])
            x_size = int(temp_strings[3])
            y_begin_list.append(y_begin)
            x_begin_list.append(x_begin)
            y_size_list.append(y_size)
            x_size_list.append(x_size)
            if x_begin < x_min:
                x_min = x_begin
            if y_begin < y_min:
                y_min = y_begin
            if x_begin + x_size > x_max:
                x_max = x_begin + x_size
            if y_begin + y_size > y_max:
                y_max = y_begin + y_size
            data = self.image_data[y_begin: y_begin + y_size, x_begin: x_begin + x_size, :]
            input_data.append(data)
        # get deep features
        deep_features = self.model.get_deep_features(sess, input_data)
        deep_features = numpy.array(deep_features)
        y_num_of_grids = (y_max - y_min) // y_size_list[0]
        x_num_of_grids = (x_max - x_min) // x_size_list[0]
        size_of_deep_features = deep_features.shape[1]
        output_data = numpy.zeros([y_num_of_grids * size_of_deep_features, x_num_of_grids * size_of_deep_features,
                                   deep_features.shape[3]])
        output_data -= 1
        for i in range(len(deep_features)):
            x_begin = (x_begin_list[i] - x_min) // x_size_list[i] * size_of_deep_features
            y_begin = (y_begin_list[i] - y_min) // y_size_list[i] * size_of_deep_features
            x_size = size_of_deep_features
            y_size = size_of_deep_features
            output_data[y_begin: y_begin + y_size, x_begin: x_begin + x_size, :] = deep_features[i]
        gdal.AllRegister()
        driver = gdal.GetDriverByName("GTiff")
        dataset = gdal.Open(coordinate_file, GA_ReadOnly)
        output_dataset = driver.Create(result_file, x_num_of_grids * size_of_deep_features,
                                       y_num_of_grids * size_of_deep_features, deep_features.shape[3],
                                       gdal.GDT_Float32)
        adf_geo_transform = dataset.GetGeoTransform()
        result_adf_geo_transform = numpy.array(adf_geo_transform)
        result_adf_geo_transform[0] = adf_geo_transform[0] + x_min * adf_geo_transform[1] + y_min * adf_geo_transform[2]
        result_adf_geo_transform[3] = adf_geo_transform[3] + x_min * adf_geo_transform[4] + y_min * adf_geo_transform[5]
        output_dataset.SetGeoTransform(result_adf_geo_transform)
        output_dataset.SetProjection(dataset.GetProjectionRef())
        for i in range(deep_features.shape[3]):
            output_array = output_data[:, :, i]
            band = output_dataset.GetRasterBand(i + 1)
            band.WriteArray(output_array)
        del dataset
        del output_dataset
