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
from PIL import Image
import torch
from facenet_pytorch import MTCNN


def get_image_paths(raw_dir):
    """递归查找原始LFW数据集目录下的所有图片"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_paths = []
    if not os.path.exists(raw_dir):
        print(f"错误: 原始数据集目录 {raw_dir} 不存在")
        return image_paths

    # 递归遍历目录
    for root, dirs, files in os.walk(raw_dir):
        files.sort()  # 保证处理顺序一致
        for file_name in files:
            # 过滤非图片文件
            if not file_name.lower().endswith(image_extensions):
                continue
            full_path = os.path.join(root, file_name)
            image_paths.append(full_path)

    print(f"递归查找完成，共发现 {len(image_paths)} 张图片")
    return image_paths


def process_cropping(raw_dir, cropped_dir):
    os.makedirs(cropped_dir, exist_ok=True)
    print(f"裁剪后人脸保存至: {cropped_dir}")

    # 获取原始图片路径列表
    process_files = get_image_paths(raw_dir)
    if not process_files:
        print("错误: 没有找到有效的图像文件")
        return

    print(f"将处理 {len(process_files)} 个图像文件...")

    # 初始化MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'在该设备上运行: {device}')
    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )

    # 执行裁剪
    cropped_paths = []
    skipped_count = 0
    for img_path in tqdm(process_files, desc="MTCNN裁剪进度"):
        if not os.path.exists(img_path):
            print(f"\n警告: 文件不存在，跳过: {img_path}")
            skipped_count += 1
            continue

        # 生成裁剪后文件名（保留原始相对路径结构）
        rel_path = os.path.relpath(img_path, raw_dir)
        rel_dir = os.path.dirname(rel_path)
        os.makedirs(os.path.join(cropped_dir, rel_dir), exist_ok=True)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_path = os.path.join(cropped_dir, rel_dir, f"{base_name}.jpg")

        try:
            with Image.open(img_path).convert('RGB') as img:
                cropped_tensor = mtcnn(img, save_path=crop_path)

                if cropped_tensor is None:
                    print(f"\n警告: 未检测到人脸，跳过: {img_path}")
                    if os.path.exists(crop_path) and os.path.getsize(crop_path) == 0:
                        os.remove(crop_path)
                    skipped_count += 1
                    continue

                cropped_paths.append(crop_path)

        except Exception as e:
            print(f"\n警告: 处理 {img_path} 失败: {str(e)}，跳过")
            if os.path.exists(crop_path):
                os.remove(crop_path)
            skipped_count += 1
            continue

    # 清理资源
    del mtcnn
    torch.cuda.empty_cache()

    # 输出统计信息
    print("\n" + "=" * 50)
    print(f"人脸裁剪完成！")
    print(f"总处理文件数: {len(process_files)}")
    print(f"成功裁剪人脸数: {len(cropped_paths)}")
    print(f"跳过文件数: {skipped_count}")
    print(f"裁剪人脸保存目录: {cropped_dir}")
    print("=" * 50)

    return cropped_paths


def main():
    parser = argparse.ArgumentParser(description='FaceNet数据集准备脚本（仅人脸裁剪）')
    parser.add_argument('--raw_dir', default='../../../../../datasets/LFW/lfw',
                        help='原始LFW数据集根目录（包含子文件夹）')
    parser.add_argument('--cropped_dir', default='../data/preprocess/lfw_cropped',
                        help='裁剪后人脸保存目录')
    args = parser.parse_args()

    process_cropping(
        raw_dir=args.raw_dir,
        cropped_dir=args.cropped_dir
    )


if __name__ == "__main__":
    main()