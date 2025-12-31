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

import argparse
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
import tf2onnx
import onnx
from onnx import shape_inference
import os
import sys

# 配置TensorFlow会话（适配TF1.x兼容模式）
config = tf.compat.v1.ConfigProto()  # 修复 deprecated 警告
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)  # 修复 deprecated 警告
K.set_session(sess)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造model.py所在目录的绝对路径
model_dir = os.path.join(current_dir, '../CrowdCount/src')
# 将该目录添加到系统路径
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
from model import MSCNN


def convert_h5_to_onnx(h5_model_path, onnx_model_path):
    # 加载原始NHWC格式的MSCNN模型
    base_model = MSCNN(input_shape=(224, 224, 3))
    if not os.path.exists(h5_model_path):
        raise FileNotFoundError(f"未找到模型文件: {h5_model_path}")
    base_model.load_weights(h5_model_path, by_name=True)
    print(f"成功加载权重文件: {h5_model_path}")

    # 1. 定义NCHW格式的输入（batch, channel, height, width）
    nchw_input = Input(shape=(3, 224, 224), dtype=tf.float32, name='input')  # NCHW输入

    # 2. 添加手动transpose层：将NCHW转为NHWC（供Keras模型使用）
    # 转换逻辑：(N, C, H, W) -> (N, H, W, C)，对应维度索引转换[0, 2, 3, 1]
    nhwc_input = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 3, 1]))(nchw_input)

    # 3. 将转换后的NHWC输入传入原始模型
    base_model_output = base_model(nhwc_input)

    # 将输出从NHWC转换为NCHW
    nchw_output = Lambda(
        lambda x: tf.transpose(x, perm=[0, 3, 1, 2]),  # (N,H,W,C) -> (N,C,H,W)
        name='output'
    )(base_model_output)

    # 4. 构建新模型：输入为NCHW，输出为转换后的NCHW
    new_model = Model(inputs=nchw_input, outputs=nchw_output)

    # 5. 定义NCHW格式的输入签名（与新模型输入一致）
    input_signature = [tf.TensorSpec((1, 3, 224, 224), tf.float32, name='input')]

    # 6. 转换为ONNX（适配tf2onnx 1.9.3，移除不支持的参数）
    onnx_model, _ = tf2onnx.convert.from_keras(
        new_model,
        input_signature=input_signature,
        opset=11,
        output_path=onnx_model_path
    )

    # 7. 手动修正输入维度（固定batchsize=1）
    # TensorFlow 1.x 中 tf.TensorSpec 的静态维度定义在转换为 ONNX 时存在兼容性问题。
    # 需要通过 显式约束输入形状 并结合 onnx 库手动修正维度来解决。
    # 获取输入节点
    input_node = onnx_model.graph.input[0]
    # 输入维度顺序：NCHW -> (1, 3, 224, 224)，修正前先确认维度索引！
    input_node.type.tensor_type.shape.dim[0].dim_value = 1  # batch
    input_node.type.tensor_type.shape.dim[1].dim_value = 3  # channel
    input_node.type.tensor_type.shape.dim[2].dim_value = 224  # height
    input_node.type.tensor_type.shape.dim[3].dim_value = 224  # width

    # 8. 形状推理（让ONNX根据输入静态形状推导所有层维度）
    inferred_model = shape_inference.infer_shapes(onnx_model)

    # 9. 修正输出维度（固定batchsize=1并确认NCHW格式）
    output_node = inferred_model.graph.output[0]
    output_node.type.tensor_type.shape.dim[0].dim_value = 1  # batch
    output_node.type.tensor_type.shape.dim[1].dim_value = 1  # channel
    output_node.type.tensor_type.shape.dim[2].dim_value = 56  # height
    output_node.type.tensor_type.shape.dim[3].dim_value = 56  # width

    # 保存修正后的模型
    onnx.save(inferred_model, onnx_model_path)
    print(f"输入形状：{input_node.type.tensor_type.shape}")
    print(f"输出形状：{output_node.type.tensor_type.shape}")
    print(f"成功转换模型并固定输入输出batchsize=1，存储到: {onnx_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='将MSCNN的h5模型转换为ONNX模型（NCHW输入格式，固定batchsize=1）')
    parser.add_argument('--input_h5', default='../model/mscnn_model_weights.h5', help='原始h5模型的文件路径')
    parser.add_argument('--output_onnx', default='../model/mscnn_model.onnx', help='输出的onnx模型的文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_h5_to_onnx(args.input_h5, args.output_onnx)
