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
import argparse
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_image(img_path, opt):
    """重写加载图像逻辑：中心剪裁+零填充"""
    print("img_path: " , img_path)
    img = cv2.imread(img_path)  # HWC, BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 关键：OpenCV自带的通道转换
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w, c = img.shape
    target_h = opt.img_size_h
    target_w = opt.img_size_w

    # -------------------------- 1. 中心剪裁 --------------------------
    # 计算剪裁区域（以中心为原点，取target_w x target_h，超出则取原图边界）
    crop_left = max(0, (w - target_w) // 2)
    crop_top = max(0, (h - target_h) // 2)
    crop_right = min(w, crop_left + target_w)
    crop_bottom = min(h, crop_top + target_h)

    # 执行剪裁
    img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
    crop_h, crop_w = img_cropped.shape[:2]
    print(crop_h, " ===============  " , crop_w)
    # -------------------------- 2. 零填充 --------------------------
    # 创建目标尺寸的零矩阵，将剪裁后的图像粘贴到中心
    #img_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_right = (target_w - crop_w)
    pad_bottom = (target_h - crop_h)

    pad_width = np.array([[0, pad_bottom], [0, pad_right], [0, 0]])
    img_padded = np.pad(img_cropped, pad_width)
    #img_padded[0:crop_h, 0: crop_w] = img_cropped
    img = np.expand_dims(img_padded, axis=0)
    img = img.transpose(0,3,1,2)
    file_name = Path(img_path).stem
    img.tofile( opt.output + file_name + ".bin")
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model inference.')
    parser.add_argument('--input',
                        default="./datasets/kitti15/training",
                        type=str)
    parser.add_argument('--output',
                        default="./data/pre/",
                        type=str)
    parser.add_argument('--img_size_w', type=int, default=1248, help='图像尺寸,w')
    parser.add_argument('--img_size_h', type=int, default=384, help='图像尺寸h')              
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
    for files in image_left_paths:
        print("files: ", files)
        file_name = Path(files).stem + ".png"
        img_left = load_image(os.path.join(data_root, "image_2", file_name), opt)
        img_right = load_image(os.path.join(data_root, "image_3", file_name), opt)
