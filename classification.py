#! /usr/bin/env python3
# coding=utf-8

# ================================================================
#
#   Editor      : PyCharm
#   File name   : classification.py
#   Author      : LiuBo
#   Created date: 2019-05-09 10:30
#   Description :
#
# ================================================================

from sklearn.ensemble import RandomForestClassifier
from data_process.RFData import RFData
import numpy as np
from tqdm import trange


def from_feature_get_class_label(rfc, feature_list, class_label_file, feature_dimension_list):
    lines_list = list()
    for k in range(len(feature_list)):
        with open(feature_list[k]) as f:
            lines = f.readlines()
            lines_list.append(lines)
    object_num = len(lines_list[0])
    with open(class_label_file, "w") as f:
        input_data = list()
        for i in trange(object_num):
            feature_data = None
            for k in range(len(feature_file_list)):
                temp_string = lines_list[k][i]
                temp_string = temp_string.strip('\n')
                temp_string = temp_string.split(' ')
                temp_data = list()
                for n in range(feature_dimension_list[k]):
                    temp_data.append(float(temp_string[n]))
                if k == 0:
                    feature_data = np.array(temp_data)
                else:
                    feature_data = np.append(feature_data, temp_data)
            input_data.append(feature_data)
        result = rfc.predict(input_data)
        for i in range(len(result)):
            f.write(str(result[i] + 1) + "\n")


def confusion_matrix(ground_truth, predict_result, confusion_matrix_file, class_num=6):
    result_matrix = np.zeros((class_num + 2, class_num + 2))
    for i in range(len(ground_truth)):
        result_matrix[predict_result[i]][ground_truth[i]] += 1
    sum1 = result_matrix.sum(axis=0)
    sum2 = result_matrix.sum(axis=1)
    all_samples = result_matrix.sum()
    result_matrix[class_num] = sum1
    result_matrix[:, class_num] = sum2
    accuracy1 = list()
    accuracy2 = list()

    correct_samples = 0
    for i in range(class_num):
        temp_accuracy = result_matrix[i][i] / sum1[i]
        accuracy1.append(temp_accuracy)
        temp_accuracy = result_matrix[i][i] / sum2[i]
        accuracy2.append(temp_accuracy)
        correct_samples += result_matrix[i][i]
    accuracy1.append(0)
    accuracy1.append(0)
    accuracy2.append(0)
    accuracy2.append(0)
    result_matrix[class_num + 1] = accuracy1
    result_matrix[:, class_num + 1] = accuracy2
    result_matrix[class_num][class_num] = all_samples
    result_matrix[class_num + 1][class_num + 1] = correct_samples / all_samples
    with open(confusion_matrix_file, "w") as f:
        for i in range(class_num + 2):
            for j in range(class_num + 2):
                f.write(str(result_matrix[i][j]) + "  ")
            f.write("\n")
    print(result_matrix)


data = RFData()

# segment_scale = [50, 70, 90, 110, 130, 150, 170, 190, 210]
# patch_size = [8, 16, 24, 32, 40]
segment_scale = [90]
patch_size = [40]
clf = RandomForestClassifier(random_state=0, oob_score=True, n_estimators=200, max_depth=12, min_samples_split=2)
for i in range(len(segment_scale)):
    for j in range(len(patch_size)):
        dimension_list = [32]
        print("Segment Scale:%d, Patch Size:%d" % (segment_scale[i], patch_size[j]))
        train_file = "Samples/WV/" + str(segment_scale[i]) + "/balance_train.txt"
        test_file = "Samples/WV/" + str(segment_scale[i]) + "/balance_test.txt"
        matrix_file = "classResult/WV/" + str(segment_scale[i]) + "/WithObjectFeature/OnlyIn" + \
                      str(patch_size[j]) + "ConfusionMatrix_balance.txt"
        label_file = "Results/WV/" + str(segment_scale[i]) + "/WithObjectFeature/OnlyIn" + \
                     str(patch_size[j]) + "Label_balance.txt"
        feature_file_list = list()
        feature_file_list.append("feature/WV/" + str(segment_scale[i]) + "/maxFeature" + str(patch_size[j]) + ".txt")
        # feature_file_list.append("feature/WV/Class6/" + str(segment_scale[i])
        # + "Scale/meanOutFeature" + str(patch_size[j]) + ".txt")
        # feature_file_list.append("feature/WV/" + str(segment_scale[i]) + "/ObjectFeature.txt")
        data.clear_data()
        data.get_data_from_sample_txt(train_file, test_file, feature_file_list, class_num=6,
                                      feature_dimension_list=dimension_list)
        clf.fit(data.train.feature, data.train.label)
        print("训练效果：%f" % clf.oob_score_)
        y_hat = clf.predict(data.train.feature)
        accuracy = np.equal(y_hat, data.train.label)
        accuracy = np.mean(accuracy.astype(float))
        print("训练精度为：%f" % accuracy)
        y_hat = clf.predict(data.test.feature)
        accuracy = np.equal(y_hat, data.test.label)
        accuracy = np.mean(accuracy.astype(float))
        print("测试精度为：%f" % accuracy)
        confusion_matrix(data.test.label, y_hat, matrix_file)
        from_feature_get_class_label(clf, feature_file_list, class_label_file=label_file,
                                     feature_dimension_list=dimension_list)






