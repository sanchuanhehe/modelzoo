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

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import os
import sys
import argparse
import json
import re
import math
from PIL import Image
import numpy as np
from tqdm import tqdm

def resize_to_rec_input(img: np.ndarray, imgH = 48, imgW=320) -> np.ndarray:
    """将裁剪后的图片调整为PaddleOCR-REC输入尺寸（保持长宽比，空白处补白）"""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
        
    # 计算缩放比例（优先满足高度，宽度自适应；若宽度超上限则按宽度缩放）
    scale = imgH / h
    new_w = int(w * scale)
    new_h = imgH
        
    if new_w > imgW:
        scale = imgW / w
        new_w = imgW
        new_h = int(h * scale)
        
    # 缩放图片
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    # 创建空白画布（白色背景，符合PaddleOCR默认输入）
    rec_img = np.ones((imgH, imgW, 3), dtype=np.uint8) * 255
        
    # 将缩放后的图片居中放置
    y_offset = (imgH - new_h) // 2
    x_offset = (imgW - new_w) // 2
    rec_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_img
        
    return rec_img

def pre_process(image_file, imgH = 48, imgW=320):
    print("preproces_img : " , image_file)
    img_pre = cv2.imread(image_file)
    img = resize_to_rec_input(img_pre)
    print("type: 0", type(img))
    if img is not None:
        h = img.shape[0]
        w = img.shape[1]
        print(f"图像尺寸：{w}x{h}")
    else:
        print(f"无法读取图像: {image_file}")
        # 设置默认值或跳过处理
        h, w = 0, 0
        return np.zeros(shape=[3,48,320])
    h = img.shape[0]
    w = img.shape[1]
    wh_ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        w_resized = imgW
    else:
        w_resized = int(math.ceil(imgH * ratio))
    after_resized_img = cv2.resize(img, (w_resized, imgH))
    after_resized_img = after_resized_img.astype("float32")

    after_resized_img = after_resized_img.transpose((2, 0, 1)) / 255
    after_resized_img -= 0.5
    after_resized_img /= 0.5
    zero_pad_img = np.zeros((3, imgH, imgW), dtype=np.float32)
    zero_pad_img[:, :, :w_resized] = after_resized_img
    valid_ratio = min(1.0, float(w_resized / imgW))
    zero_pad_img.tofile(image_file + ".bin")
    return zero_pad_img

def gen_input_bin(file):
    print("file: " ,file)
    inference_result = torch.tensor(np.fromfile("./datasets/quant_rec/" + file, dtype=np.float32))
    inference_result=inference_result.reshape(1,3,48,320).numpy()
    return inference_result

def extract_number(filename):
    # 匹配最后的下划线和数字部分
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return -1  # 如果没找到数字，返回-1

if __name__ == '__main__':
    batch = 12

    res_image = np.zeros(shape=[batch, 3,48,320])
    images = os.listdir("./datasets/paddleocr_rec_input/img/")
    num = 0
    if not os.path.isdir("./data/" + '/quant/'):
        os.makedirs(os.path.realpath("./data/"+ '/quant/'))
   
    for image_name in images:
        if image_name.startswith("zh_val_1"):
            single_quant_data = pre_process("./datasets/paddleocr_rec_input/img/" + image_name)
            res_image[num, :] = single_quant_data
            num +=1
            if num == batch:
                break;
    #img = res_image.astype(np.float32)
    img = np.array(res_image).astype(np.float32)  # 形状 [num, 3, 224, 224]
    img_2d = img.reshape(-1)
    np.savetxt('./data/quant/data_rec.txt', img_2d, fmt='%.6f', delimiter=' ')