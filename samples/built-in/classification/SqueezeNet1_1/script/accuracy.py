# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import time
import re
import argparse

import numpy as np

def cre_groundtruth_dict_fromtxt(gtfile_path):
    """
    :param filename: file contains the imagename and label number
    :return: dictionary key imagename, value is label number
    """
    img_gt_dict = {}
    count = 1
    with open(gtfile_path, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            imgName = temp[0].split(".")[0]
            imgLab = temp[1]
            img_gt_dict[imgName] = imgLab
            count+=1
    return img_gt_dict


def load_statistical_predict_result(filepath):
    """
    :param filepath: filepath to prediction file
    :return: probabilities, number of label
    """
    pre_dict = []
    with open(filepath, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(",")
            pre_dict.append(temp[0])
    return pre_dict

def extract_number(filename):
    match = re.search(r'output(\d+)\.txt$', filename)
    if(match):
        return int(match.group(1))
    else:
        raise ValueError(f"filename {filename} error")



def create_visualization_statistical_result(prediction_file_path,
                                            json_file,
                                            img_gt_dict, topn=5):
    """
    :param prediction_file_path:
    :param json_file:
    :param img_gt_dict:
    :param topn:
    :return:
    """
    writer = open(json_file, 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []
    files = os.listdir(prediction_file_path)
    count = 0
    count_hit = np.zeros(topn)
    for tfile_name in files:
        count += 1
        temp = tfile_name.split('.')[0]
        filepath = os.path.join(prediction_file_path, tfile_name)
        ret = load_statistical_predict_result(filepath)
        last_underscore_index = temp.rfind('_')  # 查找最后一个_的位置
        temp = temp[:last_underscore_index]  # 截取最后一个_前的部分
        gt = img_gt_dict[temp]
        for i in range(topn):
            if str(gt) == str(ret[i]):
                count_hit[i] += 1
                break
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(count)},
             {"key": "Number of classes", "value": str(len(ret))}])
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(topn):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(accuracy[i] * 100, 2)) + '%'})
        json.dump(table_dict, writer)
    writer.close()

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./out/result/txt/')
    parser.add_argument('--label', default='./carvana/train_masks')
    parser.add_argument('--result', default='./result.txt')
    args = parser.parse_args()

    inference_result_dir = args.output # path to output bin files       
    annotation_file_path = args.label # annotation files path, "val_label.txt"                
    json_file_name = args.result # metric result json file name

    if not os.path.exists(inference_result_dir):
        print("Inference result dir does not exist.")
    if not os.path.exists(annotation_file_path):
        print("Ground truth file does not exist.")

    img_label_dict = cre_groundtruth_dict_fromtxt(annotation_file_path)
    create_visualization_statistical_result(inference_result_dir,
                                            json_file_name,
                                            img_label_dict, topn=5)

    elapsed = (time.time() - start)
    print("Time used:", elapsed)