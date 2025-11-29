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
import math
import random
import numpy as np
import argparse
from glob import glob
from basicsr.utils import imfrombytes, imwrite
from basicsr.data.gaussian_kernels import random_mixed_kernels  # 复用原仓库的混合核函数


def set_random_seed(seed):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)


def generate_lq_image(img_gt, opt, seed):
    """生成低质量图像"""
    set_random_seed(seed)
    img_lq = np.clip(img_gt.copy(), 0, 1)
    h, w = img_gt.shape[:2]

    # 1. 多阶段高斯模糊
    if random.random() < opt['multi_stage_gaussian_prob']:
        # 2-3次轻度高斯模糊叠加
        for _ in range(random.randint(2, 3)):
            sigma = random.uniform(opt['multi_stage_sigma_range'][0], opt['multi_stage_sigma_range'][1])
            kernel_size = int(6 * sigma + 1)  # 核尺寸与sigma匹配（3σ原则）
            if kernel_size % 2 == 0:
                kernel_size += 1  # 确保奇数核
            img_lq = cv2.GaussianBlur(img_lq, (kernel_size, kernel_size), sigmaX=sigma)
            img_lq = np.clip(img_lq, 0, 1)

    # 2. 定向高斯模糊
    if random.random() < opt['directional_gaussian_prob']:
        angle = random.uniform(0, math.pi)  # 模糊方向角度
        sigma = random.uniform(opt['directional_sigma_range'][0], opt['directional_sigma_range'][1])
        # 创建定向高斯核（椭圆核，沿角度方向拉伸）
        kernel = cv2.getGaussianKernel(ksize=opt['directional_kernel_size'], sigma=sigma)
        kernel = np.outer(kernel, kernel)  # 2D高斯核
        # 旋转核以实现定向模糊
        M = cv2.getRotationMatrix2D((kernel.shape[1]//2, kernel.shape[0]//2), angle*180/math.pi, 1)
        rotated_kernel = cv2.warpAffine(kernel, M, kernel.shape[1::-1], flags=cv2.INTER_LINEAR)
        rotated_kernel /= rotated_kernel.sum()  # 归一化
        img_lq = cv2.filter2D(img_lq, -1, rotated_kernel)
        img_lq = np.clip(img_lq, 0, 1)

    # 3. 增强运动模糊（与高斯模糊互补）
    if random.random() < opt['motion_blur_prob']:
        kernel_size = random.choice(opt['motion_kernel_sizes'])
        angle = random.uniform(0, 360)
        max_length = kernel_size // 2
        length = random.randint(3, max_length)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        angle_rad = math.radians(angle)

        for i in range(length):
            x = int(center + i * math.cos(angle_rad))
            y = int(center + i * math.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] += 1

        kernel = cv2.GaussianBlur(kernel, (3, 3), sigmaX=0.5)
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        img_lq = cv2.filter2D(img_lq, -1, kernel)
        img_lq = np.clip(img_lq, 0, 1)

    # 4. 强化基础高斯模糊（核心退化）
    if random.random() < opt['gaussian_blur_prob']:
        kernel = random_mixed_kernels(
            kernel_list=opt['kernel_list'],
            kernel_prob=opt['kernel_prob'],
            kernel_size=random.choice(opt['gaussian_kernel_sizes']),
            sigma_x_range=opt['gaussian_sigma_range'],
            sigma_y_range=opt['gaussian_sigma_range'],
            rotation_range=[-math.pi/6, math.pi/6],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_lq, -1, kernel)
        img_lq = np.clip(img_lq, 0, 1)

    # 5. 高斯金字塔下采样（新增：模拟真实成像的分辨率损失）
    if random.random() < opt['gaussian_pyramid_prob']:
        levels = random.randint(1, 2)  # 1-2层金字塔
        temp = img_lq.copy()
        for _ in range(levels):
            # 先高斯模糊再下采样（符合真实成像流程）
            temp = cv2.GaussianBlur(temp, (5, 5), sigmaX=1.0)
            temp = cv2.resize(temp, (temp.shape[1]//2, temp.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        # 恢复原始尺寸
        img_lq = cv2.resize(temp, (w, h), interpolation=cv2.INTER_LINEAR)
        img_lq = np.clip(img_lq, 0, 1)

    # 6. 基础下采样退化（与高斯金字塔互补）
    if random.random() < opt['downsample_prob']:
        scale = random.uniform(opt['downsample_range'][0], opt['downsample_range'][1])
        new_size = (int(w // scale), int(h // scale))
        new_size = (max(64, new_size[0]), max(64, new_size[1]))
        img_lq = cv2.resize(img_lq, new_size, interpolation=cv2.INTER_LINEAR)
        img_lq = np.clip(img_lq, 0, 1)

    # 7. 噪声
    if random.random() < opt['noise_prob']:
        noise_sigma = random.uniform(opt['noise_range'][0] / 255., opt['noise_range'][1] / 255.)
        gaussian_noise = np.float32(np.random.randn(*img_lq.shape)) * noise_sigma
        img_lq = np.clip(img_lq + gaussian_noise, 0, 1)

    # 8. JPEG压缩
    if random.random() < opt['jpeg_prob']:
        quality = random.randint(opt['jpeg_range'][0], opt['jpeg_range'][1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img_8bit = (img_lq * 255.).astype(np.uint8)
        _, encimg = cv2.imencode('.jpg', img_8bit, encode_param)
        img_lq = np.float32(cv2.imdecode(encimg, cv2.IMREAD_COLOR)) / 255.
        img_lq = np.clip(img_lq, 0, 1)

    # 恢复原始尺寸（确保最终尺寸正确）
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
    img_lq = np.clip(img_lq, 0, 1)
    return img_lq


def main():
    parser = argparse.ArgumentParser(description='强化高斯模糊的低质量图像生成工具（易修复）')
    parser.add_argument('--gt_dir', type=str, default="../../../../../datasets/celeba-512/celeba-512-300/hq", help='高清图像文件夹路径')
    parser.add_argument('--lq_dir', type=str, default="../../../../../datasets/celeba-512/celeba-512-300/lq", help='低质量图像保存路径')
    parser.add_argument('--num_repeats', type=int, default=1, help='每张图生成的低质量版本数量')
    parser.add_argument('--seed', type=int, default=42, help='基础随机种子')
    parser.add_argument('--img_size', type=int, default=None, help='图像尺寸（None表示保持原图尺寸）')
    args = parser.parse_args()

    os.makedirs(args.lq_dir, exist_ok=True)
    set_random_seed(args.seed)

    # 退化参数：
    opt = {
        # 多阶段高斯模糊（多次轻度模糊叠加）
        'multi_stage_gaussian_prob': 1,  # 高概率应用
        'multi_stage_sigma_range': [0.8, 1.5],  # 单次模糊强度适中

        # 定向高斯模糊（沿特定方向平滑）
        'directional_gaussian_prob': 1,
        'directional_kernel_size': 15,  # 固定核尺寸
        'directional_sigma_range': [3.0, 5.0],  # 模糊强度

        # 高斯金字塔下采样（模拟真实成像）
        'gaussian_pyramid_prob': 1,

        # 运动模糊（与高斯模糊互补）
        'motion_blur_prob': 1,
        'motion_kernel_sizes': [9, 11, 13],

        # 基础高斯模糊
        'gaussian_blur_prob': 1,
        'kernel_list': ['iso', 'aniso'],  # 增加各向异性核多样性
        'kernel_prob': [0.7, 0.3],  # 主要用各向同性
        'gaussian_kernel_sizes': [13, 15, 17, 19],  # 更多核尺寸选择
        'gaussian_sigma_range': [1.5, 3.5],  # 更大的sigma范围

        # 基础下采样
        'downsample_prob': 0.6,
        'downsample_range': [2.0, 3.0],

        # 噪声
        'noise_prob': 0.2,
        'noise_range': [2, 5],

        # JPEG压缩
        'jpeg_prob': 0.4,
        'jpeg_range': [70, 90],
    }

    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    gt_paths = []
    for ext in img_extensions:
        gt_paths.extend(glob(os.path.join(args.gt_dir, ext)))
    gt_paths = sorted(gt_paths)

    if not gt_paths:
        print(f'错误：在 {args.gt_dir} 中未找到任何图像文件')
        return

    total = len(gt_paths) * args.num_repeats
    for i, gt_path in enumerate(gt_paths):
        img_name = os.path.basename(gt_path)
        basename, ext = os.path.splitext(img_name)

        try:
            with open(gt_path, 'rb') as f:
                img_bytes = f.read()
            img_gt = imfrombytes(img_bytes, float32=True)

            if img_gt is None or img_gt.size == 0:
                print(f'跳过无效图像 {gt_path}')
                continue

            if args.img_size is not None:
                img_gt = cv2.resize(
                    img_gt,
                    (args.img_size, args.img_size),
                    interpolation=cv2.INTER_AREA
                )
        except Exception as e:
            print(f'跳过损坏的图像 {gt_path}：{str(e)}')
            continue

        for r in range(args.num_repeats):
            current_seed = args.seed + i * args.num_repeats + r
            img_lq = generate_lq_image(img_gt, opt, current_seed)

            save_name = f'{basename}{ext}'
            save_path = os.path.join(args.lq_dir, save_name)

            img_lq_8bit = (np.clip(img_lq, 0, 1) * 255).astype(np.uint8)
            imwrite(img_lq_8bit, save_path)

            if (i * args.num_repeats + r + 1) % 10 == 0:
                print(f'进度：{i * args.num_repeats + r + 1}/{total}')

    print(f'所有低质量图像已保存至 {args.lq_dir}')


if __name__ == '__main__':
    main()