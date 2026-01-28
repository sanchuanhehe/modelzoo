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
import argparse
import logging
import torch
import torch_npu
import onnx
from magiconnx import OnnxGraph
from torch_npu.contrib import transfer_to_npu

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_DIR, 'models'))

from graspnet import GraspNet

def get_network(args):
    # 初始化模型
    network = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
    logging.info(f"Device: {device}")
    network.to(device)
    # 加载预训练权重参数
    weight = torch.load(args.weight_path)
    network.load_state_dict(weight['model_state_dict'])
    start_epoch = weight['epoch']
    logging.info("Weight loading path %s "%(args.weight_path))
    # 设置模型为eval mode
    network.eval()
    return network

def export_saved_model(net, args):
    # 输入节点名
    input_names = ['point_clouds']
    # 输出节点名
    output_names = ['grasp_preds']

    end_points = dict()
    device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
    logging.info(f"Device: {device}")
    end_points['point_clouds'] = torch.randn(1, args.num_points, 3).to(device)
    dummy_input = {'end_points': end_points}
    # 导出onnx模型
    torch.onnx.export(
        net,
        dummy_input,
        args.onnx_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )
    logging.info("Export onnx success!")

def model_keep_default_domain():
    # 解决different domain version问题
    model = OnnxGraph(args.onnx_file)

    model.keep_default_domain()
    model.save(args.onnx_file)

if __name__=='__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logging.info("Start export onnx:")
    parser = argparse.ArgumentParser(description='GraspNet offline model export.')
    parser.add_argument('--weight_path', type=str, default='checkpoint-rs.tar', help='Model checkpoint path')
    parser.add_argument('--onnx_file', type=str, default='graspnet.onnx', help='Output onnx file name')
    parser.add_argument('--num_points', type=int, default=20000, help='Points number')
    args = parser.parse_args()

    net = get_network(args)
    export_saved_model(net, args)

    model_keep_default_domain()
