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
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image
from tqdm import tqdm

from timm.data import create_transform
def get_transform():
    # 创建与 timm 模型兼容的变换
    val_transform = create_transform(
        input_size=(3, 288, 288),
        is_training=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation='bicubic',
        crop_pct=1
    )
    return val_transform

def efficient_preprocess(dataset_path, data_save_path):
    val_transform = get_transform()
    
    if dataset_path.endswith(".JPEG"):
        input_image = Image.open(dataset_path).convert('RGB')
        print(" image_path: " , dataset_path )
        input_tensor = val_transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(data_save_path, dataset_path.split('/')[-1].split('.')[0] + ".bin"))
        return
    val_path = os.path.join(dataset_path, 'val')
    val_files = os.listdir(val_path)
    i = 0
    for img_path in tqdm(val_files):
        input_image = Image.open(os.path.join(val_path, img_path)).convert('RGB')
        print(" image_path: " , img_path )
        input_tensor = val_transform(input_image)
        img = np.array(input_tensor).astype(np.float32)
        img.tofile(os.path.join(data_save_path, img_path.split('.')[0] + ".bin"))
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='EfficientNetV2 preprocess')
    parser.add_argument('--dataset_path', type=str, default = '../../../../datasets/ImageNet/',  help='dataset path')
    parser.add_argument('--data_save_path', type=str, default = './data/img/', help='bin file save path')
    args = parser.parse_args()
    if not os.path.isdir(args.data_save_path):
        os.makedirs(os.path.realpath(args.data_save_path))
    efficient_preprocess(args.dataset_path, args.data_save_path)

    file_list = os.listdir(args.data_save_path)
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join("./data/" , 'file_list.txt'), 'w', encoding='utf-8') as f:
        for item in sorted_files:
            f.write(f"img/{item}\n")