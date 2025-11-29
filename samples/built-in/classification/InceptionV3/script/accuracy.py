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


# 获取标签值 
def get_label_dict(label_path):
    label_dict = {}
    count = 1
    with open(label_path, "r") as f:
        for line in f.readlines():
            key = line.strip().split(" ")
            label_dict[key[0].split(".")[0].split("/")[-1]] = key[1]
            count += 1
    return label_dict


# 获取预测结果
def infer_dict(filepath):
    pre_dict = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            key = line.strip().split(",")
            pre_dict.append(key[0])
    return pre_dict


def extract_number(filename):
    match = re.search(r"output(\d+)\.txt$", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"filename {filename} error")


def get_accuracy(infer_path, json_file, label_dict, topk=5):
    count = 0
    count_hit = np.zeros(topk)
    files = os.listdir(infer_path)
    for tfile_name in files:
        count += 1
        key = tfile_name.split(".")[0]
        filepath = os.path.join(infer_path, tfile_name)
        ret = infer_dict(filepath)
        last_underscore_index = key.rfind("_")  # 查找最后一个_的位置
        key = key[:last_underscore_index]  # 截取最后一个_前的部分
        gt = label_dict[key]
        for i in range(topk):
            if str(gt) == str(ret[i]):
                count_hit[i] += 1
                break
    pre_json = {}
    pre_json["title"] = "Overall statistical evaluation"
    pre_json["value"] = []
    if "value" not in pre_json.keys():
        print("the item value does not exist!")
    else:
        pre_json["value"].extend(
            [
                {"key": "Number of images", "value": str(count)},
                {"key": "Number of classes", "value": str(len(ret))},
            ]
        )
        if count == 0:
            accuracy = 0
        else:
            accuracy = np.cumsum(count_hit) / count
        for i in range(topk):
            pre_json["value"].append(
                {
                    "key": "Top" + str(i + 1) + " accuracy",
                    "value": str(round(accuracy[i] * 100, 2)) + "%",
                }
            )
        writer = open(json_file, "w")
        json.dump(pre_json, writer)
        writer.close()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./out/result/txt/")
    parser.add_argument("--label", default="./carvana/train_masks")
    parser.add_argument("--result", default="./result.txt")
    args = parser.parse_args()

    img_label_dict = get_label_dict(args.label)
    get_accuracy(args.output, args.result, img_label_dict, topk=5)

    elapsed = time.time() - start
    print("Time used:", elapsed)
