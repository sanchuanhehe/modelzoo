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

import torch
import argparse
from facenet_pytorch import InceptionResnetV1


def export_facenet_to_onnx(args):
    """
    将 FaceNet (InceptionResnetV1) 模型导出为 ONNX 格式
    优化点：1. 去掉推理时标准化（依赖预处理已完成）；2. 纯静态维度
    """
    # 1. 初始化设备（CPU 导出，避免设备兼容问题）
    device = torch.device('cpu')
    print(f"使用设备: {device} 导出 ONNX")

    # 2. 加载模型（与推理代码完全一致的配置）
    print("加载 InceptionResnetV1 模型（vggface2 预训练权重）...")
    model = InceptionResnetV1(
        classify=False,  # 输出 512 维特征向量
        pretrained='vggface2'  # 与推理代码一致的预训练权重
    ).to(device)
    model.eval()  # 评估模式（禁用训练层）

    # 3. 构造静态虚拟输入（与推理代码输入格式完全对齐）
    # 静态维度：(batch_size=1, 3通道, 160x160) → 与推理代码输入完全一致
    # 预处理已完成标准化，此处直接用 float32 张量（无需额外处理）
    dummy_input = torch.randn(1, 3, 160, 160, device=device, dtype=torch.float32)

    # 4. 导出 ONNX 配置（纯静态，无动态维度）
    input_names = ["input"]  # 输入节点名称
    output_names = ["output"]  # 输出节点名称（512维特征向量）

    print(f"开始导出静态 ONNX 到: {args.output_onnx_path}")
    torch.onnx.export(
        model=model,
        args=(dummy_input,),  # 静态输入（batch_size=1）
        f=args.output_onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset_version,  # 推荐 11-13，兼容主流部署框架
        do_constant_folding=True,  # 折叠常量节点，优化模型
        verbose=args.verbose,
        export_params=True,  # 导出模型权重（必需）
        # 禁用动态维度（纯静态）
        dynamic_axes=None,
    )

    # 5. 输出结果信息
    print(f"\nONNX 模型导出完成！路径: {args.output_onnx_path}")
    print(f"模型静态配置：")
    print(f"  - 输入形状: (1, 3, 160, 160) （batch_size=1，不可修改）")
    print(f"  - 输出形状: (1, 512) （512维特征向量）")
    print(f"  - 算子集版本: opset-{args.opset_version}")


def main():
    parser = argparse.ArgumentParser(description='FaceNet 模型导出为 ONNX 脚本（纯静态维度）')
    parser.add_argument('--output_onnx_path', default='../model/facenet_vggface2_static.onnx',
                        help='ONNX 模型输出路径（默认：../model/facenet_vggface2_static.onnx）')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX 算子集版本（推荐11-13，兼容大部分框架）')
    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细导出日志')
    args = parser.parse_args()

    # 执行导出
    export_facenet_to_onnx(args)


if __name__ == "__main__":
    main()
