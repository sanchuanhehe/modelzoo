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
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm


def preprocess_image(img, target_size=(224, 224)):
    """预处理图像：RGB转BGR、调整尺寸、归一化到[0, 1]"""
    # 调整尺寸
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    # 归一化到[0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    # 转换为NCHW格式 (1, 3, H, W)
    img_nchw = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0)
    return np.ascontiguousarray(img_nchw)


def image_to_bin(image_path, output_bin_path, target_size=(224, 224)):
    """将单张图像转换为预处理后的bin文件"""
    try:
        img = cv2.imread(image_path)
    except Exception as e:
        print(f"警告: 无法读取图像 {image_path}: {e}")
        return False

    # 预处理
    try:
        processed = preprocess_image(img, target_size)
        processed.tofile(output_bin_path)
        return True
    except Exception as e:
        print(f"警告: 处理图像 {image_path} 失败: {e}")
        return False


def process_images(file_list_path, output_dir, bin_list_file, target_size=(224, 224), max_count=5000):
    """批量处理图像并生成bin文件列表"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取文件列表
    if not os.path.exists(file_list_path):
        print(f"错误: 文件列表 {file_list_path} 不存在")
        return

    with open(file_list_path, 'r') as f:
        try:
            data = json.load(f)
            image_files = [item[0] for item in data.get('fileList', [])]
        except json.JSONDecodeError as e:
            print(f"错误: 解析JSON文件失败: {e}")
            return

    # 限制处理数量
    process_files = image_files[:max_count]

    # 处理图像
    with open(bin_list_file, 'w') as f_list:
        for filename in tqdm(process_files, desc="预处理进度"):
            if not os.path.exists(filename):
                print(f"警告: 文件不存在，跳过: {filename}")
                continue

            # 生成输出路径
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(output_dir, f"{base_name}.bin")

            # 转换为bin文件
            if image_to_bin(filename, output_path, target_size):
                f_list.write(f"{os.path.abspath(output_path)}\n")


def main():
    parser = argparse.ArgumentParser(description='CrowdCount图像预处理脚本')
    parser.add_argument('--file_list', default='../data/file_list.json',
                        help='包含图像路径的JSON文件路径')
    parser.add_argument('--output_dir', default='../data/preprocess/bin',
                        help='预处理后bin文件输出目录')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='模型输入尺寸，格式为 宽 高，默认224 224')
    parser.add_argument('--max_count', type=int, default=5000,
                        help='最大处理图像数量，默认5000张')

    args = parser.parse_args()

    # 生成bin文件列表路径
    bin_list_file = os.path.join(os.path.dirname(args.output_dir), 'file_list.txt')

    # 执行预处理
    process_images(
        file_list_path=args.file_list,
        output_dir=args.output_dir,
        bin_list_file=bin_list_file,
        target_size=tuple(args.target_size),
        max_count=args.max_count
    )
    print(f"预处理完成，结果保存至: {args.output_dir}")
    print(f"文件列表保存至: {bin_list_file}")


if __name__ == "__main__":
    main()
