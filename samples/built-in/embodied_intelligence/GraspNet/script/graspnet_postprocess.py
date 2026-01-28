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
import torch
import torch_npu
import logging
import open3d
from graspnetAPI import GraspGroup

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_DIR, 'utils'))

from collision_detector import ModelFreeCollisionDetector

def grasp_collision_detect(grasp_group, points_cloud, args):
    """
    抓取碰撞检测
    
    Args:
        grasp_group: 待检测的抓取姿态组
        points_cloud: 点云数据，用于碰撞检测
        args: 命令行参数，包含voxel_size和collision_thresh等配置
    
    Returns:
        过滤后的无碰撞抓取姿态组
    """
    grasp_detector = ModelFreeCollisionDetector(points_cloud, voxel_size=args.voxel_size)
    # 执行碰撞检测，返回一个布尔掩码（True表示有碰撞）
    mask = grasp_detector.detect(grasp_group, approach_dist=0.05, collision_thresh=args.collision_thresh)
    return grasp_group[~mask]

def result_visualization(grasp_group, points_cloud):
    """
    对抓取姿态进行排序和选择，并使用Open3D可视化
    
    Args:
        grasp_group: 抓取姿态组
        points_cloud: 原始点云数据
    """
    # 应用非极大值抑制（NMS），去除空间上过于相似的抓取姿态
    grasp_group.nms()
     # 根据抓取分数进行降序排序（分数越高表示抓取质量越好）
    grasp_group.sort_by_score()
    # 选择前50个最佳抓取姿态用于可视化
    grasp_group = grasp_group[:50]
    logging.info("top six grasp posture info:")
    logging.info(grasp_group[:6])
    # 将抓取姿态组转换为Open3D几何对象列表，便于可视化
    grasp_postures = grasp_group.to_open3d_geometry_list()
    # 使用Open3D可视化点云和抓取姿态
    open3d.visualization.draw_geometries([points_cloud, *grasp_postures])

def data_postprocess(grasp_group_array, points_cloud, args):
    """
    碰撞检测和可视化，完成抓取姿态的最终处理
    
    Args:
        grasp_group_array: 原始抓取姿态数组（通常是模型预测的原始输出）
        points_cloud: Open3D点云对象
        args: 命令行参数配置
    """
    grasp_group = GraspGroup(grasp_group_array)
     # 如果碰撞阈值大于0，执行碰撞检测
    if args.collision_thresh > 0:
        # 碰撞检测需要点云数据为numpy数组格式
        grasp_group = grasp_collision_detect(grasp_group, np.array(points_cloud.points), args)
    # 对处理后的抓取姿态进行可视化
    result_visualization(grasp_group, points_cloud)
