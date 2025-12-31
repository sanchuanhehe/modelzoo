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
import json
import numpy as np
import cv2
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='PFLD图像预处理脚本（OpenCV直接读取版）')
    parser.add_argument('--file_list', default='../data/file_list.json',
                        help='包含图像路径的JSON文件路径')
    parser.add_argument('--output_dir', default='../data/preprocess/bin',
                        help='预处理后bin文件输出目录')
    parser.add_argument('--target_size', type=int, nargs=2, default=[112, 112],
                        help='模型输入尺寸（宽 高），默认112 112（PFLD常用输入尺寸）')
    parser.add_argument('--max_count', type=int, default=50000,
                        help='最大处理图像数量，默认50000张')
    args = parser.parse_args()

    # 生成bin文件列表路径（与输出目录同级）
    bin_list_file = os.path.join(os.path.dirname(args.output_dir), 'file_list.txt')

    # 读取JSON文件中的图像路径
    if not os.path.exists(args.file_list):
        print(f"错误: 文件列表 {args.file_list} 不存在")
        return

    with open(args.file_list, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            image_files = [item[0] for item in data.get('fileList', [])]
            if not image_files:
                print("警告: JSON文件中未找到有效图像路径")
                return
        except json.JSONDecodeError as e:
            print(f"错误: 解析JSON文件失败: {e}")
            return

    # 限制处理数量
    process_files = image_files[:args.max_count]
    print(f"共读取到 {len(image_files)} 张图像，将处理前 {len(process_files)} 张")

    # 准备输出目录（强制清空旧目录）
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理每张图像
    failed_files = []  # 记录处理失败的文件
    with open(bin_list_file, 'w', encoding='utf-8') as f_list:
        for img_path in tqdm(process_files, desc="预处理进度"):
            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"\n警告: 图像文件不存在 - {img_path}")
                failed_files.append(img_path)
                continue

            # 1. OpenCV读取图像（默认BGR通道，HWC格式，uint8[0,255]）
            img = cv2.imread(img_path)
            if img is None:
                print(f"\n警告: 图像读取失败 - {img_path}")
                failed_files.append(img_path)
                continue

            # 2. 调整尺寸到模型输入大小
            target_w, target_h = args.target_size
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # 3. 归一化到 [0.0, 1.0]（float32）
            img_norm = img_resized.astype(np.float32) / 255.0

            # 4. 通道顺序转换：HWC → CHW → NCHW（批量维度为1，适配模型输入格式）
            # HWC (h, w, 3) → CHW (3, h, w) → NCHW (1, 3, h, w)
            img_chw = np.transpose(img_norm, (2, 0, 1))  # HWC→CHW
            img_nchw = np.expand_dims(img_chw, axis=0)  # CHW→NCHW

            # 5. 保存为bin文件（float32格式，按内存顺序直接写入）
            img_stem = Path(img_path).stem
            output_bin_path = os.path.join(args.output_dir, f"{img_stem}.bin")
            img_nchw.tofile(output_bin_path)

            # 6. 写入文件列表（绝对路径，方便后续加载）
            f_list.write(f"{os.path.abspath(output_bin_path)}\n")

    # 输出处理统计信息
    print(f"\n预处理完成！")
    print(f"成功处理: {len(process_files) - len(failed_files)} 张")
    print(f"失败处理: {len(failed_files)} 张")
    if failed_files:
        print(f"失败文件列表已保存至: failed_files.txt")
        with open('failed_files.txt', 'w', encoding='utf-8') as f:
            for path in failed_files:
                f.write(f"{path}\n")
    print(f"预处理结果保存至: {args.output_dir}")
    print(f"文件列表保存至: {bin_list_file}")


if __name__ == "__main__":
    main()
