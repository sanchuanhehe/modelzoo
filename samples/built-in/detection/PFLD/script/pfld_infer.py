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

datasets_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "PFLD-pytorch",
    "models"
)

# 将datasets.py所在路径加入搜索路径
if datasets_dir not in sys.path:
    sys.path.append(datasets_dir)

from pfld import PFLDInference


def dynamic_import_dependencies(infer_engine):
    """根据选择的推理引擎动态导入对应的库"""
    if infer_engine == "onnx":
        try:
            import onnxruntime as ort
            return {"ort": ort}
        except ImportError:
            print("错误: 选择ONNX推理引擎，但未安装onnxruntime！")
            print("请执行: pip install onnxruntime onnxruntime-gpu（如需GPU支持）")
            sys.exit(1)
    elif infer_engine == "pytorch":
        # 检查PyTorch是否可用
        try:
            import torch
            return {"torch": torch}
        except ImportError:
            print("错误: 选择PyTorch推理引擎，但未安装PyTorch！")
            print("请执行: pip install torch torchvision")
            sys.exit(1)
    else:
        print(f"错误: 不支持的推理引擎 {infer_engine}，仅支持 'onnx' 或 'pytorch'")
        sys.exit(1)


def load_onnx_model(model_path, deps):
    """加载ONNX模型并获取输入输出信息"""
    ort = deps["ort"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # 优先使用GPU推理，若无则使用CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_available_providers() else [
        'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, so, providers=providers)

    input_info = session.get_inputs()
    print(f"ONNX模型输入名称: {[x.name for x in input_info]}")
    print(f"ONNX模型输入形状: {[x.shape for x in input_info]}")

    output_info = session.get_outputs()
    print(f"ONNX模型输出名称: {[x.name for x in output_info]}")
    print(f"ONNX模型输出形状: {[x.shape for x in output_info]}")

    return session, [x.name for x in input_info], [x.name for x in output_info]


def load_pytorch_model(model_path, deps):
    """加载PyTorch模型"""
    torch = deps["torch"]
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    pfld_backbone = PFLDInference()
    # 兼容不同的checkpoint存储格式（有的存在state_dict里，有的直接存）
    if 'state_dict' in checkpoint:
        pfld_backbone.load_state_dict(checkpoint['state_dict'])
    elif 'pfld_backbone' in checkpoint:
        pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    else:
        pfld_backbone.load_state_dict(checkpoint)
    pfld_backbone.eval()  # 设置为评估模式
    print("PFLD模型加载完成")
    return pfld_backbone


def run_inference(args):
    """统一执行推理逻辑，根据选择的引擎适配不同处理"""
    # 动态导入依赖
    deps = dynamic_import_dependencies(args.infer_engine)

    # 创建输出目录
    os.makedirs(args.infer_bin_dir, exist_ok=True)

    # 加载文件列表
    if not os.path.exists(args.file_list_path):
        print(f"错误: 文件列表 {args.file_list_path} 不存在")
        sys.exit(1)

    with open(args.file_list_path, "r") as f:
        bin_relative_paths = [line.strip() for line in f if line.strip()]
        bin_file_paths = []
        for rel_path in bin_relative_paths:
            abs_path = os.path.abspath(rel_path)
            if os.path.exists(abs_path):
                bin_file_paths.append(abs_path)
            else:
                print(f"警告: 文件不存在，跳过: {rel_path}")

    if not bin_file_paths:
        print("错误: 未找到有效的预处理bin文件")
        sys.exit(1)

    # 加载对应引擎的模型
    input_h, input_w = args.input_size
    if args.infer_engine == "onnx":
        # 校验ONNX模型路径
        if not os.path.exists(args.onnx_model_path):
            print(f"错误: ONNX模型文件 {args.onnx_model_path} 不存在")
            sys.exit(1)
        model, input_names, output_names = load_onnx_model(args.onnx_model_path, deps)
    else:  # pytorch
        # 校验PyTorch模型路径
        if not os.path.exists(args.pytorch_model_path):
            print(f"错误: PyTorch模型文件 {args.pytorch_model_path} 不存在")
            sys.exit(1)
        model = load_pytorch_model(args.pytorch_model_path, deps)

    # 执行批量推理
    total = len(bin_file_paths)
    for idx, bin_path in enumerate(bin_file_paths):
        file_name = os.path.basename(bin_path)
        base_name = os.path.splitext(file_name)[0]

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"推理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {file_name}")

        # 加载预处理数据
        try:
            input_tensor = np.fromfile(bin_path, dtype=np.float32)
            input_tensor = input_tensor.reshape(1, 3, input_h, input_w)
        except Exception as e:
            print(f"警告: 处理文件 {bin_path} 失败: {e}，跳过")
            continue

        # 执行推理
        try:
            if args.infer_engine == "onnx":
                preds = model.run(output_names, {input_names[0]: input_tensor})
                result = preds[1]  # 取第2个输出
            else:
                # PyTorch推理
                torch = deps["torch"]
                input_tensor_pt = torch.from_numpy(input_tensor)
                with torch.no_grad():  # 关闭梯度计算
                    model_outputs = model(input_tensor_pt)
                    # 取第2个输出作为核心结果
                    result = model_outputs[1].numpy()
        except Exception as e:
            print(f"警告: 推理文件 {bin_path} 失败: {e}，跳过")
            continue

        # 保存推理结果
        infer_bin_name = f"{base_name}_output_0.bin"
        infer_bin_path = os.path.join(args.infer_bin_dir, infer_bin_name)
        try:
            result.tofile(infer_bin_path)
        except Exception as e:
            print(f"警告: 保存推理结果 {infer_bin_path} 失败: {e}，跳过")
            continue

    print(f"推理完成，结果保存至: {args.infer_bin_dir}")


def main():
    parser = argparse.ArgumentParser(description='PFLD 统一推理脚本（支持ONNX/PyTorch）')

    # 公共参数
    parser.add_argument('--infer_engine', default='pytorch', choices=['onnx', 'pytorch'],
                        help='推理引擎选择：onnx 或 pytorch（必填）')
    parser.add_argument('--preprocess_bin_dir', default='../data/preprocess/bin',
                        help='预处理后二进制文件目录')
    parser.add_argument('--file_list_path', default='../data/preprocess/file_list.txt',
                        help='预处理生成的文件列表路径')
    parser.add_argument('--infer_bin_dir', default='../out/result_pc/bin',
                        help='推理结果二进制文件输出目录')
    parser.add_argument('--input_size', type=int, nargs=2, default=[112, 112],
                        help='模型输入尺寸，格式为 高 宽，默认112 112')

    # ONNX专属参数
    parser.add_argument('--onnx_model_path', default="../model/pfld-sim.onnx",
                        help='ONNX模型文件路径（仅infer_engine=onnx时有效）')

    # PyTorch专属参数
    parser.add_argument('--pytorch_model_path', default="../PFLD-pytorch/checkpoint/snapshot/checkpoint.pth.tar",
                        help='PyTorch模型文件路径（仅infer_engine=pytorch时有效）')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.isdir(args.preprocess_bin_dir):
        print(f"错误: 预处理文件目录 {args.preprocess_bin_dir} 不存在")
        sys.exit(1)

    run_inference(args)


if __name__ == "__main__":
    main()