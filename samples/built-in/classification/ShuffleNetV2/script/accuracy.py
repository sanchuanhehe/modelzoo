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

import os, json
import time, re
import argparse
import numpy as np

def create_gt_dict_from_txt(gtfile):
    gt_dict = {}
    with open(gtfile, 'r')as file:
        for line in file.readlines():
            tmp = line.strip().split(" ")
            img_filename = tmp[0].split(".")[0]
            if '/' in img_filename:
                img_filename = img_filename.split("/")[1]
            img_lable = tmp[1]
            gt_dict[img_filename] = img_lable
    return gt_dict


def load_predict_result_filepath(filepath):
    pre_dict = []
    with open(filepath, 'r')as file:
        for line in file.readlines():
            tmp = line.strip().split(",")
            pre_dict.append(tmp[0])
    return pre_dict

def create_visualization_statistical_result(infer_result_path,
                                            json_file,
                                            gt_dict, topn=5):
    writer = open(json_file, 'w')
    statistical_dict = {}
    statistical_dict["value"] = []
    statistical_dict["title"] = "Overall statistical evaluation"
    files = os.listdir(infer_result_path)
    cnt = 0
    count_hit = np.zeros(topn)
    for filename in files:
        cnt += 1
        tmp = filename.split('.')[0]
        filepath = os.path.join(infer_result_path, filename)
        pre_dict = load_predict_result_filepath(filepath)
        last_index = tmp.rfind('_')  # 查找最后一个_的位置
        
        last_clip = tmp[:last_index]  # 截取最后一个_前的部分
        gt = gt_dict[last_clip]
        for i in range(topn):
            if str(gt) == str(pre_dict[i]):
                count_hit[i] += 1
                break
    if 'value' not in statistical_dict.keys():
        print("the item value does not exist!")
    else:
        statistical_dict["value"].extend(
            [{"key": "Number of images", "value": str(cnt)},
             {"key": "Number of classes", "value": str(len(pre_dict))}])
        if cnt == 0:
            acc = 0
        else:
            acc = np.cumsum(count_hit) / cnt
        for i in range(topn):
            statistical_dict["value"].append({"key": "Top" + str(i + 1) + " accuracy",
                                        "value": str(
                                            round(acc[i] * 100, 2)) + '%'})
        json.dump(statistical_dict, writer)
    writer.close()

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output', default='./out/result/txt/')
    arg_parser.add_argument('--label', default='val_label.txt')
    arg_parser.add_argument('--result', default='./out/accurary.txt')
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()

    infer_result_path = args.output # path to output bin files       
    annotation_file = args.label # annotation files path, "val_label.txt"                
    result_json_file = args.result # metric result json file name

    if not os.path.exists(infer_result_path):
        print("Inference result dir does not exist.")
    if not os.path.exists(annotation_file):
        print("Ground truth file does not exist.")

    img_label_dict = create_gt_dict_from_txt(annotation_file)
    create_visualization_statistical_result(infer_result_path,
                                            result_json_file,
                                            img_label_dict, topn=5)

if __name__ == '__main__':
    start_time = time.time()
    main()
    used_time = (time.time() - start_time)
    print("Time used:", used_time)