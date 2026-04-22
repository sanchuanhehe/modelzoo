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
import matplotlib as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

extra_path = os.path.abspath("./Depth-Anything-V2")
sys.path.append(extra_path)

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

def image2tensor(raw_image, image_path, shape_kv, input_size=518):
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='upper_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
            image_path = image_path,
            shape_kv = shape_kv
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    h, w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to('cpu')
    img = np.array(image).astype(np.float32)
    return img, (h, w)

def main():
    input_path = './data/DA-2K/images/'
    output_path = os.path.realpath('./data')
    if not os.path.isdir(output_path + '/img/'):
        os.makedirs(os.path.realpath(output_path+ '/img/'))
    input_class = os.listdir(input_path)
    shapes_kv = {}
    for files in input_class:
        print("files: ", files)
        if files.replace(" ", "") == ".DS_Store":
            continue
        image_paths = os.listdir(input_path + '/' + files)
        for image_path in image_paths:
            file_path = input_path + '/' + files + '/' + image_path
            print("file_path: " , file_path)
            raw_image = cv2.imread(file_path)
            img, (h, w) = image2tensor(raw_image, image_path, shapes_kv, 518)
            print("save_path: ", output_path + '/img/' + image_path.split(".")[0] + ".bin")
            img.tofile(output_path + '/img/' + image_path.split(".")[0] + ".bin")
    with open("./data/shapes.json", "w") as f:
        json.dump(shapes_kv, f, indent=4)

if __name__ == "__main__":
    print("file_path: 123949917_fd08c80d60_b.jpg")
    output_path = os.path.realpath('./data')
    if not os.path.isdir(output_path + '/img/'):
        os.makedirs(os.path.realpath(output_path+ '/img/'))
    image_path = "123949917_fd08c80d60_b.jpg"
    raw_image = cv2.imread("./data/DA-2K/images/object/123949917_fd08c80d60_b.jpg")
    shapes_kv = {}
    img, (h, w) = image2tensor(raw_image, image_path, shapes_kv, 518)
    print("save_path: ", output_path + '/img/' + image_path.split(".")[0] + ".bin")
    img.tofile(output_path + '/img/' + image_path.split(".")[0] + ".bin")
    #main()
