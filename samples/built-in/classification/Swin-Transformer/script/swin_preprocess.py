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

import sys
import os
import argparse

from tqdm import tqdm
from torchvision import transforms
import numpy as np
from PIL import Image


# 模型图片预处理,使用transforms将图片大小等做转化
def get_transform(args):
    val_transform = transforms.Compose(
        [
            transforms.Resize(
                size=256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return val_transform


def preprocess(args):
    Transform = get_transform(args)
    val_path = os.path.join(args.data_path, "val")
    save_path = args.bin_path + "/img"
    
    if args.data_path.endswith(".JPEG"):
        input_image = Image.open(args.data_path).convert("RGB")
        input_tensor = Transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, args.data_path.split("/")[-1].split(".")[0] + ".bin"))
        return
    val_files = os.listdir(val_path)
    for img_path in tqdm(val_files):
        input_image = Image.open(os.path.join(val_path, img_path)).convert("RGB")
        input_tensor = Transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(save_path, img_path.split(".")[0] + ".bin"))


def parse_ags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../../../../datasets/ImageNet/")
    parser.add_argument("--bin_path", default="./data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_ags()
    preprocess(args)
    img_dir = os.path.join(args.bin_path, "img")
    img_dir = os.path.join(img_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    file_list = os.listdir(args.bin_path + "/img/")
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join(args.bin_path, "file_list.txt"), "w", encoding="utf-8") as f:
        for item in sorted_files:
            f.write(f"img/{item}\n")
