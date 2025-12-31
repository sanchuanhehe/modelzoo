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
import time
import sys
import numpy as np
import argparse


# 根据推理引擎动态导入依赖
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
    elif infer_engine == "tf":
        try:
            import tensorflow as tf
            from keras import backend as K
            # 导入MSCNN模型（保持原TF脚本的路径配置）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, '../CrowdCount/src')
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)
            from model import MSCNN
            # 配置GPU（与原TF脚本一致）
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
            return {"tf": tf, "K": K, "MSCNN": MSCNN}
        except ImportError as e:
            print(f"错误: 选择TF推理引擎，但依赖导入失败！")
            print(f"错误详情: {e}")
            print("请执行: pip install tensorflow keras")
            sys.exit(1)
    else:
        print(f"错误: 不支持的推理引擎 {infer_engine}，仅支持 'onnx' 或 'tf'")
        sys.exit(1)


# ONNX模型加载函数
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

    output_info = session.get_outputs()[0]
    print(f"ONNX模型输出名称: {output_info.name}")
    print(f"ONNX模型输出形状: {output_info.shape}")

    return session, [x.name for x in input_info], output_info.name


# TF模型加载函数
def load_tf_model(model_path, input_size, deps):
    """加载TensorFlow模型（基于原测试脚本的模型结构）"""
    MSCNN = deps["MSCNN"]
    # 构建模型结构（与crowdcount_test.py中的MSCNN一致）
    model = MSCNN((input_size[0], input_size[1], 3))
    # 加载权重文件
    if os.path.exists(model_path):
        model.load_weights(model_path, by_name=True)
        print(f"成功加载TF模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"TF模型权重文件不存在: {model_path}")

    # 打印TF模型输入输出维度信息
    # 获取输入信息（支持多输入场景）
    inputs = model.inputs
    print(f"\nTF模型输入数量: {len(inputs)}")
    for i, input_tensor in enumerate(inputs):
        print(f"TF模型输入{i + 1}名称: {input_tensor.name}")
        print(f"TF模型输入{i + 1}形状: {input_tensor.shape.as_list()}")  # 转为列表更易读

    # 获取输出信息（支持多输出场景）
    outputs = model.outputs
    print(f"\nTF模型输出数量: {len(outputs)}")
    for i, output_tensor in enumerate(outputs):
        print(f"TF模型输出{i + 1}名称: {output_tensor.name}")
        print(f"TF模型输出{i + 1}形状: {output_tensor.shape.as_list()}")

    return model


def run_inference(args):
    """统一执行推理逻辑，根据选择的引擎适配不同处理"""
    # 动态导入依赖
    deps = dynamic_import_dependencies(args.infer_engine)

    # 创建输出目录
    os.makedirs(args.infer_bin_dir, exist_ok=True)

    # 加载文件列表（复用原逻辑）
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
        model, input_names, output_name = load_onnx_model(args.onnx_model_path, deps)
    else:  # tf
        # 校验TF模型路径
        if not os.path.exists(args.tf_model_path):
            print(f"错误: TF模型权重文件 {args.tf_model_path} 不存在")
            sys.exit(1)
        model = load_tf_model(args.tf_model_path, args.input_size, deps)

    # 执行批量推理
    total = len(bin_file_paths)
    for idx, bin_path in enumerate(bin_file_paths):
        file_name = os.path.basename(bin_path)
        base_name = os.path.splitext(file_name)[0]

        # 打印进度（复用原逻辑）
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"推理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {file_name}")

        # 加载预处理数据（统一读取逻辑）
        try:
            input_tensor = np.fromfile(bin_path, dtype=np.float32)
            input_tensor = input_tensor.reshape(1, 3, input_h, input_w)
            # 根据引擎适配通道顺序
            if args.infer_engine == "tf":
                # TF: NHWC (1, H, W, 3)    ONNX: NCHW (1, 3, H, W)
                input_tensor = input_tensor.transpose(0, 2, 3, 1)
        except Exception as e:
            print(f"警告: 处理文件 {bin_path} 失败: {e}，跳过")
            continue

        # 执行推理（适配不同引擎）
        try:
            start = time.perf_counter()  # 开始计时
            if args.infer_engine == "onnx":
                preds = model.run([output_name], {input_names[0]: input_tensor})
                result = preds[0]  # ONNX输出是列表，取第一个元素
            else:
                # TF推理
                result = model.predict(input_tensor)
            print(f"模型预测耗时: {time.perf_counter() - start:.6f} 秒")
        except Exception as e:
            print(f"警告: 推理文件 {bin_path} 失败: {e}，跳过")
            continue

        # 保存推理结果（统一输出格式）
        infer_bin_name = f"{base_name}_output_0.bin"
        infer_bin_path = os.path.join(args.infer_bin_dir, infer_bin_name)
        try:
            result.tofile(infer_bin_path)
        except Exception as e:
            print(f"警告: 保存推理结果 {infer_bin_path} 失败: {e}，跳过")
            continue

    print(f"推理完成，结果保存至: {args.infer_bin_dir}")


def main():
    parser = argparse.ArgumentParser(description='CrowdCount 统一推理脚本（支持ONNX/TF）')

    # 公共参数
    parser.add_argument('--infer_engine', default='tf', choices=['onnx', 'tf'],
                        help='推理引擎选择：onnx 或 tf（必填）')
    parser.add_argument('--preprocess_bin_dir', default='../data/preprocess/bin',
                        help='预处理后二进制文件目录')
    parser.add_argument('--file_list_path', default='../data/preprocess/file_list.txt',
                        help='预处理生成的文件列表路径')
    parser.add_argument('--infer_bin_dir', default='../out/result_pc/bin',
                        help='推理结果二进制文件输出目录')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                        help='模型输入尺寸，格式为 高 宽，默认224 224')

    # ONNX专属参数
    parser.add_argument('--onnx_model_path', default='../model/mscnn_model.onnx',
                        help='ONNX模型文件路径（仅infer_engine=onnx时有效）')

    # TF专属参数
    parser.add_argument('--tf_model_path', default='../model/mscnn_model_weights.h5',
                        help='TF模型权重文件路径（.h5格式，仅infer_engine=tf时有效）')

    args = parser.parse_args()

    # 验证输入目录（公共校验）
    if not os.path.isdir(args.preprocess_bin_dir):
        print(f"错误: 预处理文件目录 {args.preprocess_bin_dir} 不存在")
        sys.exit(1)

    run_inference(args)


if __name__ == "__main__":
    main()
