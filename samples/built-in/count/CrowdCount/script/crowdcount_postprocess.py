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
import numpy as np
import cv2
import argparse
from PIL import Image


def postprocess(preds, original_h=None, original_w=None, threshold=0.01):
    """后处理：从模型输出计算人数并生成可视化密度图"""
    # 处理模型输出（密度图）
    density_map = preds.squeeze()  # 去除多余维度
    density_map[density_map < threshold] = 0  # 过滤低置信度值

    # 计算人数（密度图求和）
    count = int(np.sum(density_map))

    # 如果提供了原始尺寸且与当前尺寸不符，则进行缩放
    if original_h and original_w and (density_map.shape[0] != original_h or density_map.shape[1] != original_w):
        density_map = cv2.resize(density_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    # 密度图可视化处理（归一化到0-255）
    if np.max(density_map) > 0:
        density_map_norm = (density_map / np.max(density_map) * 255).astype(np.uint8)
    else:
        density_map_norm = np.zeros_like(density_map, dtype=np.uint8)
    density_map_colored = cv2.applyColorMap(density_map_norm, cv2.COLORMAP_JET)

    return count, density_map_colored


def run_postprocess(args):
    """执行后处理并保存结果（计数txt和密度图）"""
    # 创建输出目录结构
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'density_maps'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'txt'), exist_ok=True)

    # 获取所有推理结果文件
    bin_files = [f for f in os.listdir(args.bin_dir) if f.endswith('_output_0.bin')]
    if not bin_files:
        print(f"错误: 在 {args.bin_dir} 中未找到任何输出bin文件")
        return

    # 处理每个推理结果
    total = len(bin_files)
    target_h, target_w = args.target_size

    for idx, bin_file in enumerate(bin_files):
        # 提取图像名称
        img_name = bin_file.split("_output_0.bin")[0]

        # 获取原始图像尺寸（可选）
        original_h, original_w = None, None
        if args.img_dir:  # 仅当提供图像目录时才尝试读取原始尺寸
            img_path = os.path.join(args.img_dir, f"{img_name}.jpg")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    original_w, original_h = img.size
                except Exception as e:
                    print(f"警告: 无法读取图像 {img_path}: {e}，将使用原始输出尺寸")
            else:
                print(f"警告: 原始图像 {img_path} 不存在，将使用原始输出尺寸")

        # 加载推理结果
        bin_path = os.path.join(args.bin_dir, bin_file)
        try:
            preds = np.fromfile(bin_path, dtype=np.float32)
            preds = preds.reshape(target_h, target_w)
        except Exception as e:
            print(f"警告: 处理文件 {bin_file} 失败: {e}，跳过")
            continue

        # 后处理
        count, density_map = postprocess(
            preds,
            original_h=original_h,
            original_w=original_w,
            threshold=args.threshold
        )

        # 保存密度图
        density_map_path = os.path.join(args.output_dir, 'density_maps', f"{img_name}_density.jpg")
        try:
            cv2.imwrite(density_map_path, density_map)
        except Exception as e:
            print(f"警告: 保存密度图 {density_map_path} 失败: {e}，跳过")
            continue

        # 保存计数结果到单独的txt文件
        txt_path = os.path.join(args.output_dir, 'txt', f"{img_name}.txt")
        try:
            with open(txt_path, 'w') as f:
                f.write(str(count))
        except Exception as e:
            print(f"警告: 保存计数结果 {txt_path} 失败: {e}，跳过")
            continue

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {bin_file} - 人数: {count}")

    print(f"所有后处理结果已保存至: {args.output_dir}")
    print(f"计数txt文件已保存至: {os.path.join(args.output_dir, 'txt')}")


def main():
    parser = argparse.ArgumentParser(description='CrowdCount后处理脚本')
    parser.add_argument('--bin_dir', default='../out/result_pc/bin',
                        help='存放推理输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/CrowDensity/Crowdata/img',
                        help='原始图像目录（可选，不提供则不进行缩放）')
    parser.add_argument('--output_dir', default='../out/result_pc',
                        help='输出结果目录（包含密度图和txt计数）')
    parser.add_argument('--target_size', type=int, nargs=2, default=[56, 56],
                        help='模型输出尺寸，格式为 高 宽，224x224输入对应56x56输出')
    parser.add_argument('--threshold', type=float, default=0.06,
                        help='密度图阈值过滤，默认0.06')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.bin_dir):
        print(f"错误: 推理结果目录 {args.bin_dir} 不存在")
        return
    run_postprocess(args)


if __name__ == "__main__":
    main()
