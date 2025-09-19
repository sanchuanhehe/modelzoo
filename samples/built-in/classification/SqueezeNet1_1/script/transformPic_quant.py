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

import os
import sys
import argparse
import json

from PIL import Image
import numpy as np
from tqdm import tqdm

def gen_input_bin(args, file):
    # RGBA to RGB
    image = Image.open(os.path.join(input_path, file)).convert('RGB')
   
    # 预处理图片
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    res_img = val_transform(image)

    return res_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../../../datasets/ImageNet/quickstart')
    parser.add_argument('--output_path', default='./data')
    batch = 26

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_path = os.path.realpath(input_path)
    output_path = os.path.realpath(output_path)
    res_image = np.zeros(shape=[batch, 3, 224, 224])
    images = os.listdir(input_path)
    num = 0
    if not os.path.isdir(output_path + '/quant/'):
        os.makedirs(os.path.realpath(output_path+ '/quant/'))
    for image_name in images:
        single_quant_data = gen_input_bin(args, image_name)
        res_image[num, :] = single_quant_data
        num +=1
        if num == batch - 1:
            break;
    #img = res_image.astype(np.float32)
    img = np.array(res_image).astype(np.float32)  # 形状 [num, 3, 224, 224]
    img_2d = img.reshape(-1)
    np.savetxt('./data/quant/data.txt', img_2d, fmt='%.6f', delimiter=' ')