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

def pre_process(image_file, imgH = 48, imgW=320):
    print("preproces_img : " , image_file)
    img = cv2.imread(image_file)
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
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")

    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
    padding_im[:, :, :resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    padding_im.tofile(image_file + ".bin")
    return padding_im

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
    batch = 8

    res_image = np.zeros(shape=[batch, 3,48,320])
    images = os.listdir("./datasets/paddleocr_rec_input/img/")
    num = 0
    if not os.path.isdir("./data/" + '/quant/'):
        os.makedirs(os.path.realpath("./data/"+ '/quant/'))
   
    for image_name in images:
        if filename.startswith("zh_val_1"):
            single_quant_data = pre_process("./datasets/paddleocr_rec_input/img/" + image_name)
            res_image[num, :] = single_quant_data
            num +=1
            if num == batch - 1:
                break;
    #img = res_image.astype(np.float32)
    img = np.array(res_image).astype(np.float32)  # 形状 [num, 3, 224, 224]
    img_2d = img.reshape(-1)
    np.savetxt('./data/quant/data_rec.txt', img_2d, fmt='%.6f', delimiter=' ')