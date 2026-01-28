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
import open3d as o3d
import struct
import logging

from graspnetAPI import GraspGroup

def read_saved_points_cloud(file_path):
    with open(file_path, 'rb') as f:
        # 读取所有数据
        all_data = f.read()

    data_size = len(all_data)
    if data_size < 8:
        logging.info(f"invalid points cloud data: {file_path}")

    # 解析前8字节作为点数量（int64）
    num_points = struct.unpack('l', all_data[:8])[0]
    
    # 解析剩余数据作为点坐标（float64数组）
    points_data = all_data[8:]
    points = np.frombuffer(points_data, dtype=np.float64)
    
    # 重塑为点云数组
    point_cloud = points.reshape(num_points, 3)
    
    return point_cloud

if __name__=='__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    grasp_group_array = np.loadtxt('../out/txt/color_sorted_grasp_group.txt', dtype=np.float32)
    points = read_saved_points_cloud('../out/bin/cloud_points.bin')
    colors = read_saved_points_cloud('../out/bin/cloud_color.bin')
    logging.info(f"grasp_group_array is a numpy array of type {grasp_group_array.dtype} and shape {grasp_group_array.shape}")
    logging.info(f"points is a numpy array of type {points.dtype} and shape {points.shape}")
    logging.info(f"colors is a numpy array of type {colors.dtype} and shape {colors.shape}")

    points_cloud = o3d.geometry.PointCloud()
    points_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    points_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

    grasp_group = GraspGroup(grasp_group_array)
    logging.info("top six grasp posture info:")
    logging.info(grasp_group[:6])
    grippers = grasp_group.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([points_cloud, *grippers])