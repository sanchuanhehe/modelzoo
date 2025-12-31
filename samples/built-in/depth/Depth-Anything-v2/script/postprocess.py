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

import yaml
import json
import argparse
import re
import os
import fnmatch
import cv2

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
import torchvision
import torch.nn.functional as F

import re

def find_matching_key_regex(annotations, filename):
    for key in annotations.keys():
        # 检查键是否以 "images/" 开头并以文件名结尾
        if key.startswith("images/") and f"/{filename}" in key:
            return key
    return None

def postprocess(opt):
    print("postprocess: ")
    with open("./data/shapes.json", "r") as f:
        loaded_data = json.load(f)
        shapes_data = {k.split(".")[0]: tuple(v) for k, v in loaded_data.items()}
    
    with open('./data/DA-2K/DA-2K/annotations.json', 'r') as f:
        annotations = json.load(f)

    with open('./data/file_list.txt', 'r') as f:
        lines = f.readlines()

    points_num = 0
    acc_num = 0
    for i in tqdm(range(len(lines))):
        print('img_path: ', lines[i])
        out_path = Path(lines[i]).stem
        file_path = out_path
        out_filepath = f"{opt.output}/{file_path}_0.bin"
        print("out_filepath ", out_filepath)
        inference_result = torch.tensor(np.fromfile(out_filepath, dtype=np.float32))
        inference_result = inference_result.reshape(1,518,518)# x(bs, 255, H, W) 
        _, ny, nx = inference_result.shape  # x(bs, 255, H, W) → x(bs, anchors, H, W, box_attrs)
        print("inference_result ", inference_result.shape)
        h,w,h1,w1,top_pad, bottom_pad, left_pad, right_pad = shapes_data[out_path]
        # 裁剪数组
        cropped_array = inference_result[
            :,  # 所有通道
            top_pad:518 - bottom_pad,  # 高度方向
            left_pad:518 - right_pad   # 宽度方向
        ]
        #还原图像大小
        depth = F.interpolate(cropped_array[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        depth = depth.squeeze()
        
        depth = np.array(depth).astype(np.float32)
        
        matching_key = find_matching_key_regex(annotations, out_path)
        points = annotations[matching_key]
        for point in points:
            points_num = points_num + 1
            point_gt1 = point['point1']
            point_gt2 = point['point2']
            point_infer1 = depth[point_gt1[0], point_gt1[1]]
            point_infer2 = depth[point_gt2[0], point_gt2[1]]
            if(point_infer1 > point_infer2):
                acc_num = acc_num + 1
        
    if points_num > 0:
        print("accuary: " , acc_num/points_num)
        return acc_num/points_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 offline model inference.')
    parser.add_argument('--output',
                        default="./out/result/bin",
                        type=str)
    
    opt = parser.parse_args()

    postprocess(opt)
