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
import numpy as np
from tqdm import tqdm


def bin_to_txt(bin_path, txt_path):
    """将bin文件转换为txt文件（保存512维特征）"""
    try:
        features = np.fromfile(bin_path, dtype=np.float32)
        if len(features) != 512:
            print(f"警告: {bin_path} 特征维度不正确 ({len(features)})，跳过")
            return False

        with open(txt_path, 'w') as f:
            f.write(' '.join(map(str, features)))
        return True
    except Exception as e:
        print(f"警告: 处理 {bin_path} 失败: {e}，跳过")
        return False


def run_postprocess(args):
    """执行后处理"""
    os.makedirs(args.output_txt_dir, exist_ok=True)

    # 获取所有推理结果文件
    bin_files = [f for f in os.listdir(args.bin_dir) if f.endswith('_output.bin')]
    if not bin_files:
        print(f"错误: 在 {args.bin_dir} 中未找到任何输出bin文件")
        return

    # 处理每个bin文件
    total = len(bin_files)
    for idx, bin_file in enumerate(tqdm(bin_files, desc="后处理进度")):
        base_name = os.path.splitext(bin_file)[0].replace('_output', '')
        bin_path = os.path.join(args.bin_dir, bin_file)
        txt_path = os.path.join(args.output_txt_dir, f"{base_name}.txt")

        bin_to_txt(bin_path, txt_path)

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%)")

    print(f"后处理完成，结果保存至: {args.output_txt_dir}")


def main():
    parser = argparse.ArgumentParser(description='FaceNet后处理脚本')
    parser.add_argument('--bin_dir', default='../out/result/bin',
                        help='存放推理输出bin文件的目录')
    parser.add_argument('--output_txt_dir', default='../out/result/txt',
                        help='特征向量txt文件输出目录')
    args = parser.parse_args()

    if not os.path.isdir(args.bin_dir):
        print(f"错误: 推理结果目录 {args.bin_dir} 不存在")
        return

    run_postprocess(args)


if __name__ == "__main__":
    main()
