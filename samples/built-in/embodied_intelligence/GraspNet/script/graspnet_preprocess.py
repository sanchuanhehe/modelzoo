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
import sys
import numpy as np
import scipy.io as scio
import logging
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from PIL import Image
import open3d

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_DIR, 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image

def sample_point_cloud(points, colors, target_num):
    """
    对点云进行采样，确保返回target_num个点

    Args:
        points: 原始点云数据 (N, 3)
        colors: 对应颜色数据 (N, 3)
        target_num: 目标采样点数
        
    Returns:
        sampled_points: 采样后的点云
        sampled_colors: 采样后的颜色
    """
    num_points = len(points)

    if num_points == 0:
        raise ValueError("输入点云数据为空")

    if num_points >= target_num:
        # 点足够多，随机不放回采样
        indices = np.random.choice(num_points, target_num, replace=False)
    else:
        # 点不足，先保留所有点，再补充采样
        original_indices = np.arange(num_points)
        additional_num = target_num - num_points
        # 随机有放回采样补充
        additional_indices = np.random.choice(num_points, additional_num, replace=True)
        indices = np.concatenate([original_indices, additional_indices], axis=0)

    return points[indices], colors[indices]

def data_preprocess(args):
    # 读取相机参数
    camera_para = scio.loadmat(os.path.join(args.data_dir, 'meta.mat'))
    intrinsic_matrix = camera_para['intrinsic_matrix']
    depth_scale_factor = camera_para['factor_depth']

    # 生成点云数据
    camera = CameraInfo(
        width=1280.0, 
        height=720.0, 
        fx=intrinsic_matrix[0][0],  # 内参矩阵fx
        fy=intrinsic_matrix[1][1],  # 内参矩阵fy
        cx=intrinsic_matrix[0][2],  # 内参矩阵cx
        cy=intrinsic_matrix[1][2],  # 内参矩阵cy
        scale=depth_scale_factor     # 深度缩放因子
    )
    # 读取图像数据
    depth_data = np.array(Image.open(os.path.join(args.data_dir, 'depth.png')))
    color_data = np.array(Image.open(os.path.join(args.data_dir, 'color.png')), dtype=np.float32)
    color_data = color_data / 255.0
    # 读取图像有效区域
    valid_region = np.array(Image.open(os.path.join(args.data_dir, 'workspace_mask.png')))
    cloud_data = create_point_cloud_from_depth_image(depth_data, camera, organized=True)

    # 获取有效点云数据
    mask = (valid_region & (depth_data > 0))
    cloud_valid = cloud_data[mask]
    color_valid = color_data[mask]

    # 对点云数据进行采样
    cloud_sampled, color_sampled = sample_point_cloud(cloud_valid, color_valid, args.num_points)

    # 数据转换
    end_points = dict()
    # 添加batch维度: (N, 3) -> (1, N, 3)
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    if torch_npu.npu.is_available():
        device = torch.device("npu")
        logging.info(f"使用NPU设备")
    else:
        device = torch.device("cpu")
        logging.info("使用CPU设备")
    # 将数据移动到对应设备
    cloud_sampled_tensor = cloud_sampled_tensor.to(device)
    end_points['point_clouds'] = cloud_sampled_tensor  # PyTorch Tensor
    end_points['cloud_colors'] = color_sampled # NumPy数组

    points_cloud = open3d.geometry.PointCloud()
    # 设置点云坐标后续作为模型入参
    points_cloud.points = open3d.utility.Vector3dVector(cloud_valid.astype(np.float32))
    # 设置点云颜色用于结果可视化
    points_cloud.colors = open3d.utility.Vector3dVector(color_valid.astype(np.float32))

    return end_points, points_cloud
