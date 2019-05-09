#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : extract.py
#   Author      : LiuBo
#   Created date: 2019-05-09 09:52
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
from Utils.FeatureExtractor import FeatureExtractor
import numpy


if __name__ == "__main__":
    segmentation_scale = 90
    input_size = 40
    object_id = 31936
    model = DenseNet40(train_summary_dir="summary/train/" + str(input_size),
                       test_summary_dir="summary/test/" + str(input_size), training=False)
    original_file = "D:/DoLab/Research/Data/WV/WV10400.jpg"
    window_set_file = "WindowSet/WV/" + str(segmentation_scale) + "/WindowSet" + str(input_size) + "Percent.txt"
    result_file = "features/WV/" + str(segmentation_scale) + "/meanFeatures" + str(input_size) + ".txt"
    checkpoint = "checkpoint/" + str(input_size)

    deep_features_file = "features/WV/" + str(segmentation_scale) + "/" + \
                         str(object_id) + "_" + str(input_size) + ".tif"
    aggregation_function = numpy.mean
    extractor = FeatureExtractor(model)
    extractor.extract_object_features_by_id(window_set_file, deep_features_file, checkpoint, original_file, object_id)
    # extractor.extract_features(window_set_file, result_file, checkpoint, original_file, aggregation_function)
