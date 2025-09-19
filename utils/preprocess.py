# Copyright 2021 Huawei Technologies Co., Ltd
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
import json
import multiprocessing

from PIL import Image
import numpy as np
from tqdm import tqdm

def center_crop(img, output_size):
    print("center crop size: " , output_size)
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def resize(img, size, mode, interpolation=Image.BILINEAR):
    print("resize: " , size, mode)
    if(mode == 1):
         print("resize mode 1 ")
         return img.resize((size, size))
    if(mode == 0):
        print("resize mode 0")
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

def select_dtype(value):
    if value == "uint8":
        return np.uint8    # 无符号 8 位
    elif value == "int8":
        return np.int8    # 有符号 8 位
    elif value == "int16":
        return np.int16    # 有符号 16 位
    elif value == "int32":
        return np.int32    # 有符号 32 位
    elif value == "float32":
        return np.float32    # 浮点数 32 位
    else:
        return np.int8    # 大整数

def transpose(img, mode):
    if mode == 0:
        return np.transpose(img, [2,0,1])
    elif mode == 1:
        return img
    else:
        return img

def gen_input_bin(args, file_batches):
    for file in file_batches:
        # RGBA to RGB
        image = Image.open(os.path.join(input_path, file)).convert('RGB')
        width, height = image.size
        print("width, height: " , width, height)
        if args.resize:
            image = resize(image, args.resize, args.resize_mode) # Resize
        if args.center_crop:
            image = center_crop(image, args.center_crop) # CenterCrop
        image_type = select_dtype(args.type)
        print("type: " , image_type)
        img = np.array(image, dtype=image_type)
        if args.mean:
            print("mean: " , args.mean)
            img = img / args.mean # mean
        img = transpose(img, args.transpose)
        print("img: shape " , img.shape)
        bin_data = np.expand_dims(img, axis=0)
        bin_data = bin_data.flatten()
        bin_data.tofile(os.path.join(output_path+ '/img/', file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../../../datasets/ImageNet/val')
    parser.add_argument('--output_path', default='./data')
    parser.add_argument('--resize', type=int)
    parser.add_argument('--resize_mode', type=int, default=0)
    parser.add_argument('--center_crop', type=int)
    parser.add_argument('--mean', type=int)
    parser.add_argument('--type', default='int8')
    parser.add_argument('--transpose', type=int, default=0)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_path = os.path.realpath(input_path)
    output_path = os.path.realpath(output_path)
    if not os.path.isdir(output_path + '/img/'):
        os.makedirs(os.path.realpath(output_path+ '/img/'))
    gen_input_bin(args, os.listdir(input_path))
    # 获取当前目录下的所有文件和文件夹
    file_list = os.listdir(output_path + '/img/')
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join(output_path , 'file_list.txt'), 'w', encoding='utf-8') as f:
        for item in sorted_files:
            f.write(f"img/{item}\n")
