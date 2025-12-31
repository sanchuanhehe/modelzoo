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
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization


def tensor_to_bin(tensor, output_bin_path):
    """将预处理后的tensor（对应eval_lfw.py中的xb）保存为bin文件"""
    try:
        # 直接保存预处理后的tensor（与eval_lfw.py中xb格式完全一致）
        tensor.numpy().tofile(output_bin_path)
        return True
    except Exception as e:
        print(f"警告: 生成bin文件失败 {output_bin_path}: {e}")
        return False


def process_images(file_list_path, bin_dir, max_count=50000):
    """批量处理流程：使用已有裁剪图 → 预处理 → 生成bin（无推理）"""
    # 强制创建输出目录
    os.makedirs(bin_dir, exist_ok=True)
    print(f"预处理bin文件输出至: {bin_dir}")

    # 读取文件列表（直接包含裁剪后图像路径）
    if not os.path.exists(file_list_path):
        print(f"错误: 文件列表 {file_list_path} 不存在")
        return

    with open(file_list_path, 'r') as f:
        try:
            data = json.load(f)
            # 直接获取裁剪后图像路径列表
            cropped_paths = [item[0] for item in data.get('fileList', []) if item[0].strip()]
        except json.JSONDecodeError as e:
            print(f"错误: 解析JSON文件失败: {e}")
            return

    # 限制处理数量
    process_files = cropped_paths[:max_count]
    if not process_files:
        print("错误: 没有找到有效的裁剪图像文件路径")
        return
    print(f"找到 {len(process_files)} 个裁剪图像文件，开始处理...")

    # 验证裁剪图像是否存在
    valid_cropped_paths = []
    skipped_count = 0
    for crop_path in process_files:
        if not os.path.exists(crop_path):
            print(f"\n警告: 裁剪文件不存在，跳过: {crop_path}")
            skipped_count += 1
            continue
        valid_cropped_paths.append(crop_path)

    if not valid_cropped_paths:
        print("错误: 没有找到有效的裁剪人脸图像，无法生成bin文件")
        return
    print(f"\n有效裁剪人脸图像数: {len(valid_cropped_paths)}")

    # 预处理裁剪图像（保持与原代码完全一致的预处理逻辑）
    print("\n开始预处理裁剪图像并生成bin文件...")

    # 仅预处理生成bin，不进行ResNet推理
    bin_list = []
    with torch.no_grad():
        for crop_path in tqdm(valid_cropped_paths, desc="预处理生成bin进度"):
            try:
                # 读取图像并转换为RGB
                with Image.open(crop_path).convert('RGB') as img:
                    # 手动复现原transforms.Compose的处理逻辑
                    # 1. 转换为float32 numpy数组
                    img_np = np.array(img, dtype=np.float32)
                    # 2. 转换为Tensor (会自动归一化到[0,1])
                    img_tensor = transforms.ToTensor()(img_np)
                    # 3. 应用固定图像标准化
                    img_tensor = fixed_image_standardization(img_tensor)

                    # 生成bin文件路径
                    base_name = os.path.splitext(os.path.basename(crop_path))[0]
                    bin_path = os.path.join(bin_dir, f"{base_name}.bin")

                    if tensor_to_bin(img_tensor, bin_path):
                        bin_list.append(os.path.abspath(bin_path))
            except Exception as e:
                print(f"\n警告: 处理 {crop_path} 失败: {str(e)}，跳过")
                skipped_count += 1
                continue

    # 保存bin文件列表（保持与原代码一致的保存位置）
    bin_list_path = os.path.join(os.path.dirname(bin_dir), 'bin_file_list.txt')
    with open(bin_list_path, 'w') as f:
        for path in bin_list:
            f.write(f"{path}\n")

    # 输出统计信息（保持与原代码一致的格式）
    print("\n" + "=" * 50)
    print(f"预处理完成！")
    print(f"总处理文件数: {len(process_files)}")
    print(f"有效裁剪人脸数: {len(valid_cropped_paths)}")
    print(f"成功生成bin文件数: {len(bin_list)}")
    print(f"跳过文件数: {skipped_count}")
    print(f"bin文件列表保存至: {bin_list_path}")
    print(f"bin文件保存目录: {bin_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='FaceNet图像预处理脚本（使用已有裁剪图生成bin）')
    parser.add_argument('--file_list', default='../data/file_list.json',
                        help='包含裁剪后图像路径的JSON文件路径')
    parser.add_argument('--output_dir', default='../data/preprocess/bin',
                        help='预处理后bin文件输出目录')
    parser.add_argument('--max_count', type=int, default=50000,
                        help='最大处理图像数量，默认50000张')
    args = parser.parse_args()

    process_images(
        file_list_path=args.file_list,
        bin_dir=args.output_dir,
        max_count=args.max_count
    )


if __name__ == "__main__":
    main()
