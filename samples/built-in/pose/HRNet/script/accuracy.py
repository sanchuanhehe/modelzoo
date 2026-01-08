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
import cv2
import torch
import sys
import argparse
import json
import os
from tqdm import tqdm

sys.path.append(r"./HigherHRNet-Human-Pose-Estimation")
sys.path.append(r"./HigherHRNet-Human-Pose-Estimation/lib")

from core.group import HeatmapParser
from lib.config import cfg
from lib.config import update_config
from lib.utils.transforms import get_multi_scale_size
from lib.dataset.COCODataset import CocoDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--input',
                        help='json file',
                        required=True,
                        type=str)

    parser.add_argument('--dir',
                        help='model output dir',
                        required=True,
                        type=str)

    args = parser.parse_args()
    update_config(cfg, args)

    return args

def get_affine_matrix(center, scale_w, scale_h, output_size):
    """计算仿射变换矩阵（原始图像→模型输入图像）"""
    dst_w, dst_h = output_size
    src_w = scale_w * 200.0  # 基于人体尺度的原始宽度
    src_h = scale_h * 200.0  # 基于人体尺度的原始高度

    # 缩放+平移矩阵（保证中心对齐）
    trans = np.array([
        [dst_w / src_w, 0, dst_w * (-center[0] / src_w + 0.5)],
        [0, dst_h / src_h, dst_h * (-center[1] / src_h + 0.5)]
    ], dtype=np.float32)
    return trans

def get_multi_stage_outputs(outputs, size_projected):
    heatmaps = []
    tags = []

    output = torch.nn.functional.interpolate(
        outputs[0],
        size=(outputs[-1].size(2), outputs[-1].size(3)),
        mode='bilinear',
        align_corners=False
    )
    offset_feat = 17
    heatmaps = output[:, :offset_feat] + outputs[-1][:, :offset_feat]
    heatmaps /= 2
    tags = output[:, offset_feat:]

    heatmaps = torch.nn.functional.interpolate(
        heatmaps,
        size=(size_projected[1], size_projected[0]),
        mode='bilinear',
        align_corners=False
    )

    tags = torch.nn.functional.interpolate(
        tags,
        size=(size_projected[1], size_projected[0]),
        mode='bilinear',
        align_corners=False
    )
    tags = torch.unsqueeze(tags, dim=4)
    return heatmaps, tags


def main():
    args = parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 3. 提取fileList字段（容错：若字段不存在返回空列表）
    file_list = json_data.get('fileList', [])
    if not isinstance(file_list, list):
        print(f"fileList字段不是列表类型")
        return

    all_results = []
    all_scores = []

    coco_dataset = CocoDataset(cfg.DATASET.ROOT, cfg.DATASET.TEST, cfg.DATASET.DATA_FORMAT)
    file_name_list = []

    for sub_list in tqdm(file_list):
        if not isinstance(sub_list, list) or len(sub_list) != 1:
            return
    
        file_path = sub_list[0][3:]
        image = cv2.imread(file_path)

        (w_resized, h_resized), center, (scale_w, scale_h) = get_multi_scale_size(image, cfg.DATASET.INPUT_SIZE, 1.0, 1)
        output_size = (w_resized, h_resized)

        # 仿射变换：缩放到计算出的尺寸（保证短边优先+64倍数）
        trans = get_affine_matrix(center, scale_w, scale_h, output_size)

        # 获取文件名（带扩展名）
        file_name_with_ext = os.path.basename(file_path)
        file_name_list.append(file_name_with_ext)
        file_name, _ = os.path.splitext(file_name_with_ext)
        outputs = []
        for i in range(2):
            output_file_path = f"{args.dir}/{file_name}_{i}.bin"
            inference_result = torch.tensor(np.fromfile(output_file_path, dtype=np.float32))
            inference_result = inference_result.reshape([1, 17 * (2 - i), 128 * (i + 1), 192 * (i + 1)])
            outputs.append(inference_result)

        heatmaps, tags = get_multi_stage_outputs(outputs, output_size)

        parser = HeatmapParser(cfg)
        # 4. 热图解析
        grouped_joints, scores = parser.parse(heatmaps, tags)

        # 输入图像→原始图像的逆变换矩阵
        inv_trans = cv2.invertAffineTransform(trans)

        final_results = []
        for person in grouped_joints[0]:
            joints_orig = np.zeros((person.shape[0], person.shape[1]), dtype=np.float32)
            for jid in range(person.shape[0]):
                x_input, y_input, score, tag = person[jid]
                # 输入图像坐标 → 原始图像坐标
                x_orig, y_orig = cv2.transform(
                    np.array([[[x_input, y_input]]]), inv_trans
                )[0][0]
                joints_orig[jid] = [x_orig, y_orig, score, tag]
            final_results.append(joints_orig)
        all_results.append(final_results)
        all_scores.append(scores)

    coco_dataset.ids = [coco_dataset.file_name_to_id[value] for value in file_name_list]
    coco_dataset.evaluate(cfg, all_results, all_scores, "final_output_dir")
    return

if __name__ == "__main__":
    main()