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
import argparse
import time
import torch
import torch_npu
import logging
from ais_bench.infer.interface import InferSession
from torch_npu.contrib import transfer_to_npu

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_DIR, 'models'))

from graspnet_preprocess import data_preprocess
from graspnet_postprocess import data_postprocess

def offline_inference(end_points):
    t0 = time.perf_counter()
    # 模型加载
    session = InferSession(device_id=0, model_path=args.model_path)
    t1 = time.perf_counter()
    logging.info(f"InferSession time: {t1 - t0} s")

    om_input = [end_points['point_clouds'].cpu().numpy()]
    # 模型推理
    model_output = session.infer(feeds=om_input)
    # 获取输出信息
    output_info = session.get_outputs()
    # 模型输出结果会被战平为1维数据，需要reshape处理
    for i, output_tensor in enumerate(output_info):
        logging.info(f"output info: name={output_tensor.name}, shape={output_tensor.shape}")
        if isinstance(model_output[i], list):
            model_output[i] = np.array(model_output[i])
        model_output[i] = model_output[i].reshape(output_tensor.shape)
    t2 = time.perf_counter()
    logging.info(f"Offline inference time: {t2 - t1} s")
    return model_output

def infer_and_pred(end_points):
    with torch.no_grad():
        # 离线推理
        grasp_group = offline_inference(end_points)
    grasp_group_array = grasp_group[0]
    # 通过object_id是否为-1判断数据是否有效
    mask = (grasp_group_array[:, 16] == -1)
    valid_grasp_group = grasp_group_array[mask]
    return valid_grasp_group

def runn_inference(args):
    # 数据预处理
    t0 = time.perf_counter()
    end_points, points_cloud = data_preprocess(args)
    t1 = time.perf_counter()
    logging.info(f"Data preprocess time: {t1 - t0} s")
    # 离线推理
    grasp_group_array = infer_and_pred(end_points)
    t2 = time.perf_counter()
    # 预测结果后处理
    data_postprocess(grasp_group_array, points_cloud, args)
    t3 = time.perf_counter()
    logging.info(f"Data postprocess time: {t3 - t2} s")

if __name__=='__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info("Start GraspNet offline model inference:")
    parser = argparse.ArgumentParser(description='GraspNet offline model inference.')
    parser.add_argument('--data_dir', type=str, default='doc/example_data', help='Input data path')
    parser.add_argument('--model_path', type=str, default='graspnet_linux_aarch64.om', help='Offline model path')
    parser.add_argument('--num_points', type=int, default=20000, help='Number of points')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Used for collision detection')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Used for points cloud down sample caculation')
    args = parser.parse_args()
    # 离线推理
    runn_inference(args)