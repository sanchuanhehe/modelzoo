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


def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    """保持比例缩放图片并填充至目标尺寸（与Ultralytics实现对齐）"""
    shape = img.shape[:2]  # 原始尺寸 (h, w)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # 计算缩放比例
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    # 中心对齐填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (new_w, new_h)
    dw, dh = (target_size[1] - new_unpad[0]) / 2, (target_size[0] - new_unpad[1]) / 2  # 填充量

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 填充边界
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    # 格式转换: BGR->RGB, 归一化, 转置为CHW格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, axis=0)  # 添加batch维度
    return img


def image_to_bin(image_path, output_bin_path, target_size=(640, 640)):
    """将图片转换为bin文件（符合YOLOv8输入要求）"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return False

    # 预处理
    image = letterbox(image, target_size)

    # 保存为bin文件
    try:
        image.tofile(output_bin_path)
        return True
    except Exception as e:
        print(f"保存bin文件失败 {output_bin_path}: {e}")
        return False


def process_images(file_list_path, output_dir, bin_list_file, target_size=(640, 640), max_count=5000):
    """处理图片并生成bin文件列表"""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(file_list_path):
        print(f"错误: 文件列表 {file_list_path} 不存在")
        return

    with open(file_list_path, 'r') as f:
        try:
            data = json.load(f)
            image_files = [item[0] for item in data.get('fileList', [])]
        except json.JSONDecodeError as e:
            print(f"解析JSON文件失败: {e}")
            return

    process_files = image_files[:max_count]

    with open(bin_list_file, 'w') as f_list:
        for filename in tqdm(process_files, desc="处理进度"):
            if not os.path.exists(filename):
                print(f"文件不存在，跳过: {filename}")
                continue

            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_path = os.path.join(output_dir, f"{base_name}.bin")

            if image_to_bin(filename, output_path, target_size):
                abs_output_path = os.path.abspath(output_path)
                f_list.write(f"{abs_output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8s-seg图像预处理脚本')
    parser.add_argument('--file_list', default='../data/file_list.json',
                        help='包含图片路径的JSON文件路径')
    parser.add_argument('--output_dir', default='../data/preprocess/bin',
                        help='输出bin文件路径')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                        help='目标尺寸，格式为 宽 高')
    parser.add_argument('--max_count', type=int, default=5000,
                        help='最大处理图片数量')

    args = parser.parse_args()
    process_images(
        file_list_path=args.file_list,
        output_dir=args.output_dir,
        bin_list_file=os.path.dirname(args.output_dir) + '/file_list.txt',
        target_size=tuple(args.target_size),
        max_count=args.max_count
    )
    print(f"预处理完成，bin文件已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
