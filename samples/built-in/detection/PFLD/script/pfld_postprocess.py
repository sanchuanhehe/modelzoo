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
import argparse
from tqdm import tqdm


def postprocess(bin_file, target_size=(112, 112)):
    """解码bin文件并转换为 landmarks 坐标"""
    inference_result = np.fromfile(bin_file, dtype=np.float32)
    # 重塑为 (1, L, 2) 格式并缩放至目标尺寸
    landmarks = np.array(inference_result).reshape(1, -1, 2) * target_size
    return landmarks.squeeze()  # 去除多余维度，返回 (L, 2) 格式


def run_postprocess(args):
    """执行后处理并保存结果到txt文件"""
    # 创建输出目录
    os.makedirs(args.output_txt_dir, exist_ok=True)

    # 获取所有bin文件
    bin_files = [f for f in os.listdir(args.bin_dir) if f.endswith('_output_0.bin')]
    if not bin_files:
        print(f"错误: 在 {args.bin_dir} 中未找到任何输出bin文件")
        return

    # 处理每个bin文件
    total = len(bin_files)
    for idx, bin_file in enumerate(tqdm(bin_files, desc="后处理进度")):
        # 提取图像名称
        img_name = bin_file.split("_output_0.bin")[0]

        # 加载并处理bin文件
        bin_path = os.path.join(args.bin_dir, bin_file)
        try:
            landmarks = postprocess(bin_path)
        except Exception as e:
            print(f"警告: 处理文件 {bin_file} 失败: {e}，跳过")
            continue

        # 保存为txt文件（每行一个坐标对，格式：x y）
        txt_path = os.path.join(args.output_txt_dir, f"{img_name}.txt")
        try:
            with open(txt_path, 'w') as f:
                for (x, y) in landmarks:
                    f.write(f"{x:.6f} {y:.6f}\n")
        except Exception as e:
            print(f"警告: 保存结果 {txt_path} 失败: {e}，跳过")
            continue

    print(f"所有后处理结果已保存至: {args.output_txt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='PFLD后处理脚本（仅解码bin文件）')
    parser.add_argument('--bin_dir', default="../out/result_pc/bin",
                        help='存放推理输出bin文件的目录')
    parser.add_argument('--output_txt_dir', default="../out/result_pc/txt",
                        help='输出txt文件的保存目录')
    parser.add_argument('--target_size', type=int, nargs=2, default=[112, 112],
                        help='坐标缩放尺寸，默认[112, 112]')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_postprocess(args)
