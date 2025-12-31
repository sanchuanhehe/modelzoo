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

import numpy as np
import string
import cv2
import os
import sys
import argparse
import json
import re
import math
from rapidfuzz.distance import Levenshtein

def read_file_to_dict_basic(file_path):
    result_dict = {}
    # 打开文件（指定编码，如utf-8，避免中文乱码）
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # line_num记录行号，便于调试
            # 去除行首/行尾的空白字符（换行、空格、制表符等）
            line_stripped = line.strip()
            if not line_stripped:  # 跳过空行
                continue
            # 按两个空格分割（split的第二个参数设为1，只分割一次，避免value含两个空格）
            parts = line_stripped.split("  ", 1)
            # 确保分割后有key和value
            if len(parts) == 2:
                key = parts[0].strip().split("/")[-1].split(".")[0]  # 再次去空格，避免key前后有多余空格
                value = parts[1].strip()
                if "zh_val_1_" in key:
                    result_dict[key] = value
            else:
                print(f"警告：第{line_num}行格式错误，无法分割为key-value：{line}")
    return result_dict

def get_pre(file_path):
    pres = os.listdir(file_path)
    num = 0
    result_dict = {}
    for pre in pres:
       # 打开文件（指定编码，如utf-8，避免中文乱码）
        with open(file_path + pre, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # line_num记录行号，便于调试
                # 去除行首/行尾的空白字符（换行、空格、制表符等）
                line_stripped = line.strip()
                key = pre.split(".")[0]
                result_dict[key] = line_stripped
    return result_dict

def process(gt, pred_label):
        preds = pred_label
        labels = gt
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for key in labels:
            print("key: " , key)
            
            pred = preds[key + "_0"].replace(" ", "")
            print("pred: ", pred)
            target = labels[key].replace(" ", "")   
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        correct_num += correct_num
        all_num += all_num
        norm_edit_dis += norm_edit_dis
        return {
            "acc": correct_num / (all_num ),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num ),
        }


# 调用示例
if __name__ == "__main__":
    file_path = "./datasets/paddleocr_rec_input/zh_val_labels.txt"  # 替换为你的文件路径
    gt = read_file_to_dict_basic(file_path)
    pre_file = "./out/result/txt1/"
    pre = get_pre(pre_file)
    result = process(gt, pre)
    print("精度结果: ", result["acc"])