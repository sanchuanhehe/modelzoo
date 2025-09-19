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


import os, argparse
import numpy as np
from tqdm import tqdm
import json

def evaluate_accuracy(args):
    label_result = dict()
    with open(args.label_path, 'r') as file:
        for gt_info in file.readlines():
            image_name, label_id = gt_info.split(' ')
            label_result[os.path.splitext(image_name)[0].split("/")[-1]] = np.array(int(label_id))
    pred_result = dict()
    predict_filelist = os.listdir(args.input_dir)
    # 过滤
    predict_filelist = list(
        filter(lambda x:os.path.splitext(x)[1] == ".bin", predict_filelist))
    # 遍历每个文件
    for pred_name in predict_filelist:
        pred_file = os.path.join( args.input_dir, pred_name)
        pred_arr = np.argsort(-1 * np.fromfile(pred_file, dtype=args.dtype))
        # 保存每个预测文件的top1 和top5 结果
        pred_result[os.path.splitext(pred_name)[0][:-2]] = {
            "top1": pred_arr[0], "top5": pred_arr[:5]}
    # label 数量
    num_total = len(label_result)
    if len(pred_result) != num_total:
        raise ValueError(
            "Error : len(pred_result)({})  != num_total({})".format(
                len(pred_result), num_total
            ))
    acc1_num = 0
    acc5_num = 0
    # 统计准确率
    for file_name in tqdm(pred_result):
        gt_label = label_result.get(file_name)
        pred_acc1 = pred_result.get(file_name)["top1"]
        pred_acc5 = pred_result.get(file_name)["top5"]
        acc1_num += np.sum(pred_acc1 == gt_label)
        acc5_num += np.sum(pred_acc5 == gt_label.repeat(5))
    # 计算最终的top1 和top5 结果
    result_output = {
        "Top1 Acc": "{:.2f}%".format(acc1_num * 100 / num_total),
        "Top5 Acc": "{:.2f}%".format(acc5_num * 100 / num_total)
    }
    print(result_output)
    with open(args.save_path, 'w') as f:
        json.dump(result_output, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # 解析参数
    parser_args = argparse.ArgumentParser(description='Postprocess.')
    parser_args.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Post-processing results of model inference')
    parser_args.add_argument('-l', '--label_path', type=str, required=True,
                        help='Ground Truth Label')
    parser_args.add_argument('-s', '--save_path', type=str, default='./out/result_acc.json',
                        help='Saved results file of accuracy evaluation')
    parser_args.add_argument('-d', '--dtype', type=str, default='float32',
                        help='dtype for predict result')
    args = parser_args.parse_args()
    # 评估精度
    evaluate_accuracy(args)
