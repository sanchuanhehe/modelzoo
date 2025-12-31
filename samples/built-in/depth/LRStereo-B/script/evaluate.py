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

def postprocess(depth):
    print("postprocess: ")
    min_val = 0
    max_val = 128
    color_map = cv2.COLORMAP_JET
    
    
    ny, nx = depth.shape  # x(bs, 255, H, W) → x(bs, anchors, H, W, box_attrs)
    print("depth ", depth.shape)
    invalid_mask = depth >= np.inf
    vis = ((depth - min_val) / ( max_val - min_val)).clip(0, 1) * 255

    vis = cv2.applyColorMap(cv2.convertScaleAbs(vis.clip(0, 255).astype(np.uint8), alpha=1), color_map)

    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis.astype(np.uint8)


def epe_metric(disp_pred, disp_gt , mask):
    e_abs = torch.abs(disp_gt - disp_pred)
    masked = torch.where(mask, e_abs, torch.zeros_like(e_abs))

    # 沿H,W维度求和，得到每张图的总误差
    sum = masked.sum(dim=[1,2])

    # 每张图的有效像素数量
    num_valid_pixels = mask.sum(dim=[1,2])

    # 每张图的平均误差
    epe_image = sum / num_valid_pixels
    epe_image = torch.where(num_valid_pixels > 0, epe_image, torch.zeros_like(epe_image))

    return epe_image

def d1_metric(disp_pred, disp_gt , mask):
    e = torch.abs(disp_gt - disp_pred)
    err_mask = (e > 3) & (e / torch.abs(disp_gt) > 0.05)
    err_mask = err_mask & mask

    # 每张图的错误像素
    num_errors = err_mask.sum(dim=[1,2])

    # 每张图的有效像素数量
    num_valid_pixels = mask.sum(dim=[1,2])
    d1_image = num_errors.float() / num_valid_pixels.float() * 100
    d1_image = torch.where(num_valid_pixels > 0, d1_image, torch.zeros_like(d1_image))

    return d1_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model inference.')
    parser.add_argument('--input',
                        default="./datasets/kitti15/training",
                        type=str)
    parser.add_argument('--output',
                        default="./out/result/bin",
                        type=str)
    opt = parser.parse_args()
    data_root = opt.input
    image_left_paths = sorted([os.path.join(data_root, "image_2", f) 
                                        for f in os.listdir(os.path.join(data_root, "image_2")) 
                                        if f.endswith("_10.png")])
    image_right_paths = sorted([os.path.join(data_root, "image_3", f) 
                                         for f in os.listdir(os.path.join(data_root, "image_3")) 
                                         if f.endswith("_10.png")])
    disp_gt_paths = sorted([os.path.join(data_root, "disp_noc_0", f) 
                                     for f in os.listdir(os.path.join(data_root, "disp_noc_0")) 
                                     if f.endswith("_10.png")])
    epe_list, d1_list = [], []
    for file in image_left_paths:
        file_name = Path(file).stem
        out_filepath = f"{opt.output}/{file_name}_0.bin"
        print("out_filepath ", out_filepath)
        inference_result = np.fromfile(out_filepath, dtype=np.float32)
        inference_result = inference_result.reshape(384,1248)# x(bs, 255, H, W) 
        pre_color = postprocess(inference_result)
        gt_filepath = f"{data_root}/disp_noc_0/{file_name}.png"
        print(" gt_filepath: " , gt_filepath)
        gt = cv2.imread(gt_filepath, -1) / 256.0
        h, w = gt.shape
        print(" h, w, " , h, w)
        pre_color = pre_color[0: h , 0: w]
        gt_color = postprocess(gt)
        gt = torch.from_numpy(np.array(gt, dtype = np.float32)).unsqueeze(-1)
        val = (gt > 0) & (gt < 1000)
        inference_result = inference_result[0: h , 0: w]
        disp = torch.from_numpy(inference_result).unsqueeze(-1)

        epe = epe_metric(disp, gt , val).mean().item()
        d1 = d1_metric(disp, gt , val).mean().item()
        epe_list.append(epe)
        d1_list.append(d1)
    
    epe_meam = np.mean(epe_list)
    dl_mean = np.mean(d1_list)
    print("accuray epe_meam: ", epe_meam, " dl_mean: " , dl_mean)





