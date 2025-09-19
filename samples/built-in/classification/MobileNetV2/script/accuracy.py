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
import numpy as np
import json
import time
import re

def load_predict_result(filepath):
    pre_dict = []
    with open(filepath, 'r')as f:
        for line in f.readlines():
            temp = line.strip().split(",")
            pre_dict.append(temp[0])
    return pre_dict

def get_groundtruth_dict(gt_file_path):
    gt_dict = {}
    with open(gt_file_path, 'r')as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            imgName = parts[0].replace('val/', '')  # 直接去除 'val/' 前缀
            imgName = imgName.split(".")[0]          # 可选：同时去掉扩展名
            imgLab = parts[1]

            gt_dict[imgName] = imgLab
    return gt_dict

def create_accuracy_result_json(prediction_file_dir_path,
                                            json_file,
                                            img_gt_dict, topn=5):
    file = open(json_file, 'w')
    json_table_dict = {}
    json_table_dict["title"] = "Overall statistical evaluation"
    json_table_dict["value"] = []
    files = os.listdir(prediction_file_dir_path)
    count = 0
    hit_count = np.zeros(topn)
    for tfile_name in files:
        count += 1
        temp = tfile_name.split('.')[0]
        filepath = os.path.join(prediction_file_dir_path, tfile_name)
        ret = load_predict_result(filepath)
        last_underscore_index = temp.rfind('_')  # 查找最后一个_的位置
        temp = temp[:last_underscore_index]  # 截取最后一个_前的部分
        if temp in img_gt_dict:
            gt = img_gt_dict[temp]
        else:
            print(f"{temp} is not in img_gt_dict")
            continue
        for i in range(topn):
            if str(gt) == str(ret[i]):
                hit_count[i] += 1
                break
    if count == 0:
        print(f"{prediction_file_dir_path} has not prediction file")
        file.close()
        return
    accuracy = np.cumsum(hit_count) / count

    # 写入json文件
    json_table_dict["value"].extend(
        [{"Number of images" : str(count)}, {"Number of classes" : str(len(ret))}])
    for i in range(topn):
        json_table_dict["value"].append({"Top" + str(i + 1) + " accuracy" : str(round(accuracy[i] * 100, 2)) + '%'})
    json.dump(json_table_dict, file)
    file.close()

if __name__ == '__main__':

    prediction_result_dir = sys.argv[1] # prediction result txt files dir path
    if not os.path.exists(prediction_result_dir):
        print("prediction result dir does not exist.")   
        sys.exit(1)   
    groundtruth_file_path = sys.argv[2] # annotation files path, "val_label.txt" 
    if not os.path.exists(groundtruth_file_path):
        print("Ground truth file does not exist.")   
        sys.exit(1)            
    json_file_name = sys.argv[3] # metric result json file name

    img_label_dict = get_groundtruth_dict(groundtruth_file_path)
    create_accuracy_result_json(prediction_result_dir,
                                            json_file_name,
                                            img_label_dict, topn=5)