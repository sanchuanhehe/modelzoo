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
import numpy as np
import onnxruntime as ort
import argparse


def load_onnx_model(model_path):
    """加载ONNX模型"""
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    input_info = session.get_inputs()[0]
    print(f"模型输入名称: {input_info.name}")
    print(f"模型输入形状: {input_info.shape}")

    output_info = session.get_outputs()[0]
    print(f"模型输出名称: {output_info.name}")
    print(f"模型输出形状: {output_info.shape}")

    return session, input_info.name, output_info.name


def run_inference(args):
    """执行推理并保存结果"""
    # 创建输出目录
    os.makedirs(args.infer_bin_dir, exist_ok=True)

    # 加载文件列表
    if not os.path.exists(args.file_list_path):
        print(f"错误: 文件列表 {args.file_list_path} 不存在")
        exit(1)

    with open(args.file_list_path, "r") as f:
        # 读取所有bin文件路径（相对路径转绝对路径）
        bin_relative_paths = [line.strip() for line in f if line.strip()]
        bin_file_paths = []
        for rel_path in bin_relative_paths:
            # 处理相对路径，转为绝对路径
            abs_path = os.path.abspath(os.path.join(os.path.dirname(args.file_list_path), rel_path))
            if os.path.exists(abs_path):
                bin_file_paths.append(abs_path)
            else:
                print(f"[警告] 文件列表中指定的文件不存在: {rel_path}，跳过")

    if not bin_file_paths:
        print("错误: 未找到有效的预处理bin文件")
        exit(1)

    # 加载模型
    if not os.path.exists(args.onnx_model_path):
        print(f"错误: ONNX模型文件 {args.onnx_model_path} 不存在")
        exit(1)

    session, input_name, output_name = load_onnx_model(args.onnx_model_path)

    # 解析输入尺寸 (h, w)
    input_h, input_w = args.input_size

    # 遍历所有预处理文件进行推理
    total = len(bin_file_paths)
    for idx, bin_load_path in enumerate(bin_file_paths):
        # 提取文件名（用于生成输出文件名）
        file_name = os.path.basename(bin_load_path)
        base_name = os.path.splitext(file_name)[0]

        # 打印进度信息
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"推理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {file_name}")

        # 加载二进制数据并恢复为张量
        try:
            input_tensor = np.fromfile(bin_load_path, dtype=np.float32)
            input_tensor = input_tensor.reshape(1, 3, input_h, input_w)  # 恢复为模型输入形状
        except Exception as e:
            print(f"[警告] 处理文件 {bin_load_path} 失败: {e}，跳过")
            continue

        # 执行推理
        try:
            preds = session.run([output_name], {input_name: input_tensor})
        except Exception as e:
            print(f"[警告] 推理文件 {bin_load_path} 失败: {e}，跳过")
            continue

        # 保存推理结果为二进制文件
        infer_bin_name = f"{base_name}_output_0.bin"
        infer_bin_path = os.path.join(args.infer_bin_dir, infer_bin_name)

        try:
            preds[0].tofile(infer_bin_path)  # preds[0]为模型输出张量
        except Exception as e:
            print(f"[警告] 保存推理结果 {infer_bin_path} 失败: {e}，跳过")
            continue

    print(f"推理完成，结果保存至: {args.infer_bin_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型推理脚本')

    parser.add_argument('--preprocess_bin_dir', default='../data/preprocess/bin', help='预处理后二进制文件目录')
    parser.add_argument('--infer_bin_dir', default='../out/result_pc/bin', help='推理结果二进制文件输出目录')
    parser.add_argument('--file_list_path', default='../data/preprocess/file_list.txt', help='预处理生成的文件列表路径')
    parser.add_argument('--onnx_model_path', default='../model/yolo11s-pose.onnx', help='ONNX模型文件路径')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640], help='模型输入尺寸，格式为 高 宽，默认640 640')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.preprocess_bin_dir):
        print(f"错误: 预处理文件目录 {args.preprocess_bin_dir} 不存在")
        exit(1)

    # 执行推理
    run_inference(args)


if __name__ == "__main__":
    main()
