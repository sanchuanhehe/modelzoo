# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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
import math
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
from shutil import rmtree


def letterbox(img, target_size=(640, 640)):
    """保持比例缩放图片并填充至目标尺寸（与C++端LetterBox一致）"""
    shape = img.shape[:2]  # (h, w)

    # 计算缩放比例，不放大
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    r = min(r, 1.0)

    # 计算未填充的尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (target_size[1] - new_unpad[0]) / 2.0
    dh = (target_size[0] - new_unpad[1]) / 2.0

    # 缩放
    if (shape[1], shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 填充
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return img


def preprocess_image(image_path, target_size=(640, 640)):
    """预处理单张图片，与C++ Yolov9sPreprocess保持一致

    流程: 读取BGR -> 预缩放(长边=target) -> letterbox -> BGR2RGB -> /255.0 -> HWC->CHW
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return None

    h0, w0 = image.shape[:2]

    # 预缩放：等比缩放比取宽/高方向较小值，确保缩放后不超出目标框（与C++端一致）
    pre_r = min(target_size[0] / h0, target_size[1] / w0)
    if pre_r != 1.0:
        new_w = min(int(math.ceil(w0 * pre_r)), target_size[1])
        new_h = min(int(math.ceil(h0 * pre_r)), target_size[0])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox填充
    image = letterbox(image, target_size)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 归一化到 [0, 1] float32（YOLOv9s不使用AIPP，需软件归一化）
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW
    image = image.transpose(2, 0, 1)

    # 确保连续内存布局
    image = np.ascontiguousarray(image)

    # 添加batch维度 (1 x C x H x W)
    image = np.expand_dims(image, axis=0)

    return image


def image_to_bin(image_path, output_bin_path, target_size=(640, 640)):
    """将图片转换为bin文件

    Args:
        image_path: 输入图片路径
        output_bin_path: 输出bin文件路径
        target_size: 目标尺寸，默认为 (640, 640)

    Returns:
        bool: 处理是否成功
    """
    # 预处理图片
    processed_image = preprocess_image(image_path, target_size)
    if processed_image is None:
        return False

    # 保存为bin文件
    try:
        processed_image.tofile(output_bin_path)
        return True
    except Exception as e:
        print(f"保存bin文件失败 {output_bin_path}: {e}")
        return False


def process_images(file_list_path, output_dir, target_size=(640, 640), max_count=5000):
    """处理file_list.json中的图片，转换为bin文件并保存到输出文件夹

    Args:
        file_list_path: 包含图片路径的JSON文件
        output_dir: 输出目录
        target_size: 目标尺寸，默认为 (640, 640)
        max_count: 最大处理数量
    """
    # 如果目标目录已存在，先删除以清空旧数据
    if os.path.exists(output_dir):
        print(f"清空已存在的目录: {output_dir}")
        rmtree(output_dir)

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取file_list.json
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

    # 限制处理数量
    process_files = image_files[:max_count]

    # 遍历并处理图片，带进度条
    success_count = 0
    for filename in tqdm(process_files, desc="YOLOv9s预处理进度"):
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"文件不存在，跳过: {filename}")
            continue

        # 生成输出文件名（保持原名，替换扩展名为.bin）
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(output_dir, f"{base_name}.bin")

        # 转换图片为bin文件
        if image_to_bin(filename, output_path, target_size):
            success_count += 1

    print(f"处理完成，成功处理 {success_count}/{len(process_files)} 张图片")


def main():
    parser = argparse.ArgumentParser(description='YOLOv9s 数据预处理：将图片转换为float32 bin文件')

    parser.add_argument('--file_list', default='../data/file_list.json',
                        help='包含图片路径的JSON文件路径，默认../data/file_list.json')
    parser.add_argument('--output_dir', default='../out/preprocess/bin',
                        help='输出bin文件文件夹路径 (默认: ../out/preprocess/bin)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                        help='目标尺寸，格式为 高 宽，默认640 640')
    parser.add_argument('--max_count', type=int, default=5000,
                        help='最大处理图片数量，默认5000张')

    args = parser.parse_args()

    # 处理图片
    process_images(
        file_list_path=args.file_list,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        max_count=args.max_count
    )
    print(f"YOLOv9s预处理完成，bin文件已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
