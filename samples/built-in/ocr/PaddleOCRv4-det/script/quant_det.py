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
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import cv2
import numpy as np
import math
from PIL import Image

def preproces_img(image_file):
        print("preproces_img: ", image_file)
        img = cv2.imread(image_file)
        if img is not None:
            h = img.shape[0]
            w = img.shape[1]
            print(f"图像尺寸：{w}x{h}")
        else:
            print(f"无法读取图像: {image_file}")
            # 设置默认值或跳过处理
            h, w = 0, 0
        src_h, src_w, _ = img.shape
        print(" src_h, src_w, ", src_h, src_w, sum([src_h, src_w]))
        if sum([src_h, src_w]) < 64:
            print("do padding")
            img = image_padding(img)
        img, [ratio_h, ratio_w] = resize_image_type0(img)
        scale = np.float32(1.0 / 255.0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")
        if isinstance(img, Image.Image):
            img = np.array(img)
        img = (img.astype("float32") * scale - mean) / std
        img = img.transpose((2, 0, 1))
        img.tofile("./data/quant/det/" + Path(image_file).stem + ".bin")
        return img


def image_padding(im, value=0):
    h, w, c = im.shape
    im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
    im_pad[:h, :w, :] = im
    return im_pad

def resize_image_type0(img):
    print("resize_image_type0")
    limit_side_len = 960
    h, w, c = img.shape
    print(h, w, c, limit_side_len)
    if max(h, w) > limit_side_len:
        if h > w:
            resize_ratio = float(limit_side_len) / h
        else:
            resize_ratio = float(limit_side_len) / w
    else:
        resize_ratio = 1.0
    resize_height = int(h * resize_ratio)
    resize_width = int(w * resize_ratio)

    #将图像尺寸调整为32的倍数，并确保最小尺寸为32
    resize_height = max(int(round(resize_height / 32) * 32), 32)
    resize_width = max(int(round(resize_width / 32) * 32), 32)

    if int(resize_width) <= 0 or int(resize_height) <= 0:
        return None, (None, None)
    img = cv2.resize(img, (int(resize_width), int(resize_height)))
    ratio_h = resize_height / float(h)
    ratio_w = resize_width / float(w)
    img = cv2.copyMakeBorder(
            img,
            0,
            960 - resize_height,
            0,
            960 - resize_width,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    print("resize_height2 - img_h: ", 960 - resize_height)
    print("resize_image_type0", [ratio_h, ratio_w])
    return img, [ratio_h, ratio_w]

def gen_input_bin(file):
    inference_result = torch.tensor(np.fromfile("./data/imgevaldet_quant/" + file, dtype=np.float32))
    inference_result=inference_result.reshape(1,3,960,960).numpy()

    return inference_result

if __name__ == '__main__':
    batch = 13
    res_image = np.zeros(shape=[batch, 3, 960, 960])
    images = os.listdir("./PaddleOCR2Pytorch/doc/imgs/")
    quant_img = []
    for image_name in images:
        name = Path(image_name).stem
        if name.isdigit():
            quant_img.append(image_name)
    #sorted_files = sorted(images, key = extract_number)
    num = 0
    if not os.path.isdir("./data/" + '/quant/'):
        os.makedirs(os.path.realpath("./data/"+ '/quant/'))
        os.makedirs(os.path.realpath("./data"+ '/quant' + '/det'))
    for image_name in quant_img:
        single_quant_data = preproces_img("./PaddleOCR2Pytorch/doc/imgs/" + image_name)
        res_image[num, :] = single_quant_data
        num +=1
        if num == batch - 1:
            break
    #img = res_image.astype(np.float32)
    img = np.array(res_image).astype(np.float32)  # 形状 [num, 3, 224, 224]
    img_2d = img.reshape(-1)
    np.savetxt('./data/quant/data_det.txt', img_2d, fmt='%.6f', delimiter=' ')