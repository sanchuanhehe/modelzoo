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


def postprocess(preds, original_h, original_w, target_h, target_w):
    """后处理：将模型输出转换为原始尺寸的图像"""
    # 处理模型输出
    img_np = preds[0].clip(-1, 1)  # 裁剪到[-1, 1]范围
    img_np = (img_np + 1) / 2  # 映射到[0, 1]

    # 转换为HWC格式并从RGB转BGR
    img_np = img_np.transpose(1, 2, 0)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 转换为uint8格式
    img_np = (img_np * 255.0).round().astype(np.uint8)

    # 调整回原始图像尺寸
    if (img_np.shape[1], img_np.shape[0]) != (original_w, original_h):
        img_np = cv2.resize(img_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    return img_np


def run_postprocess(args):
    """执行后处理并保存结果图像"""
    os.makedirs(args.output_dir, exist_ok=True)

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
        img_path = os.path.join(args.img_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            print(f"警告: 原始图像 {img_path} 不存在，跳过")
            continue

        # 获取原始图像尺寸
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}，跳过")
            continue
        original_h, original_w = image.shape[:2]

        # 加载推理结果
        bin_path = os.path.join(args.bin_dir, bin_file)
        try:
            preds = np.fromfile(bin_path, dtype=np.float32)
            # 恢复为模型输出形状 (1, 3, H, W)
            preds = preds.reshape(1, 3, target_h, target_w)
        except Exception as e:
            print(f"警告: 处理文件 {bin_file} 失败: {e}，跳过")
            continue

        # 后处理
        result_img = postprocess(
            preds,
            original_h=original_h,
            original_w=original_w,
            target_h=target_h,
            target_w=target_w
        )

        # 保存结果图像
        output_img_path = os.path.join(args.output_dir, f"{img_name}_restored.jpg")
        try:
            cv2.imwrite(output_img_path, result_img)
        except Exception as e:
            print(f"警告: 保存结果图像 {output_img_path} 失败: {e}，跳过")
            continue

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {bin_file}")

    print(f"所有后处理结果已保存至: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='CodeFormer后处理脚本')
    parser.add_argument('--bin_dir', default='../out/result_pc/bin',
                        help='存放推理输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/celeba-512/celeba-512-300/hq/',
                        help='原始图像目录')
    parser.add_argument('--output_dir', default='../out/result_pc/img',
                        help='输出结果图像目录')
    parser.add_argument('--target_size', type=int, nargs=2, default=[512, 512],
                        help='模型输入尺寸，格式为 高 宽，默认512 512')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.bin_dir):
        print(f"错误: 推理结果目录 {args.bin_dir} 不存在")
        return
    if not os.path.isdir(args.img_dir):
        print(f"错误: 原始图像目录 {args.img_dir} 不存在")
        return

    run_postprocess(args)


if __name__ == "__main__":
    main()