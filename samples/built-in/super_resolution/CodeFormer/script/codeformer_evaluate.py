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
import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from lpips import LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    """加载图像并转换为RGB格式"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    return img.astype(np.float32) / 255.0  # 归一化到[0,1]


def calculate_lpips(gt_img, pred_img, lpips_model, transform, device):
    """计算LPIPS指标（使用预初始化的模型和转换函数）"""
    gt_tensor = transform(gt_img).unsqueeze(0).to(device)  # 直接使用传入的device
    pred_tensor = transform(pred_img).unsqueeze(0).to(device)

    with torch.no_grad():
        lpips_score = lpips_model(gt_tensor, pred_tensor).item()
    return lpips_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='../../../../../datasets/celeba-512/celeba-512-300/hq',
                        help='GT图像目录（512x512）')
    parser.add_argument('--restored_dir', type=str, default='../out/result/img',
                        help='CodeFormer修复结果目录')
    parser.add_argument('--output_file', type=str, default='../out/result/metrics_results.txt',
                        help='评估结果保存文件')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 仅初始化一次LPIPS模型和图像转换函数
    lpips_model = LPIPS(net='alex').to(device)  # 模型只加载一次
    transform = transforms.Compose([  # 转换函数只定义一次
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 获取所有图像路径
    gt_paths = glob.glob(os.path.join(args.gt_dir, '*.[jpJP][pnPN]*[gG]'))
    if not gt_paths:
        print("错误: 未找到GT图像")
        return

    # 初始化指标累加器
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    # 计算指标
    for gt_path in tqdm(gt_paths, desc='计算评估指标'):
        img_name = os.path.basename(gt_path)
        img_name = os.path.splitext(img_name)[0]
        restored_path = os.path.join(args.restored_dir, f"{img_name}_restored.jpg")

        # 检查修复结果是否存在
        if not os.path.exists(restored_path):
            print(f"警告: 未找到修复结果 {img_name}，已跳过")
            continue

        # 加载图像
        gt_img = load_image(gt_path)
        restored_img = load_image(restored_path)

        # 确保图像尺寸一致
        if gt_img.shape != restored_img.shape:
            print(f"警告: {img_name} 尺寸不匹配，已跳过")
            continue

        # 计算PSNR
        psnr_score = psnr(
            (gt_img * 255).astype(np.uint8),
            (restored_img * 255).astype(np.uint8),
            data_range=255
        )

        # 计算SSIM
        ssim_score = ssim(
            (gt_img * 255).astype(np.uint8)[..., 0],
            (restored_img * 255).astype(np.uint8)[..., 0],
            data_range=255
        )

        # 计算LPIPS（使用预初始化的模型和转换函数）
        lpips_score = calculate_lpips(gt_img, restored_img, lpips_model, transform, device)

        # 累加指标
        total_psnr += psnr_score
        total_ssim += ssim_score
        total_lpips += lpips_score
        count += 1

    # 计算平均值
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count

        # 输出结果
        result_str = (f"评估结果 (共 {count} 张图像):\n"
                      f"平均PSNR: {avg_psnr:.4f} dB\n"
                      f"平均SSIM: {avg_ssim:.4f}\n"
                      f"平均LPIPS: {avg_lpips:.4f}")
        print(result_str)

        # 保存结果到文件
        with open(args.output_file, 'w') as f:
            f.write(result_str)
        print(f"结果已保存至 {args.output_file}")
    else:
        print("错误: 没有有效的图像对用于计算指标")


if __name__ == '__main__':
    main()
