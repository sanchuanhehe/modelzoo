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

import onnx
import argparse
from onnx import helper, TensorProto


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="精简CodeFormer的ONNX模型（固定输入常量并保留指定输出）")
    parser.add_argument("--input", required=True, default='../model/codeformer.onnx',
                        help="输入的ONNX模型文件路径（如：/path/to/codeformer.onnx）")
    parser.add_argument("--output", required=True, default='../model/codeformer_fixed.onnx',
                        help="输出的精简后ONNX模型文件路径（如：/path/to/codeformer_fixed.onnx）")
    args = parser.parse_args()

    # 加载模型
    try:
        model = onnx.load(args.input)
    except Exception as e:
        raise FileNotFoundError(f"无法加载输入模型 {args.input}：{str(e)}")
    graph = model.graph

    # ====================== 精简输入节点 ======================
    # 1. 删除原有的"onnx::Cast_1"输入节点
    new_inputs = [inp for inp in graph.input if inp.name != "onnx::Cast_1"]
    del graph.input[:]  # 清空原有输入
    graph.input.extend(new_inputs)  # 添加过滤后的输入

    # 2. 创建常量节点替代"onnx::Cast_1"输入
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["onnx::Cast_1"],
        value=helper.make_tensor(
            name="const_value",
            data_type=TensorProto.DOUBLE,  # 匹配原输入类型
            dims=[],  # 标量维度
            vals=[0.5]  # 固定值（可根据需求修改）
        )
    )
    graph.node.insert(0, const_node)  # 插入常量节点到计算图最前面

    # ====================== 精简输出节点 ======================
    # 仅保留名称为"2936"的输出，删除其他输出
    target_output_name = "2936"
    new_outputs = [out for out in graph.output if out.name == target_output_name]

    if not new_outputs:
        # 提示可用输出节点名称，方便调试
        available_outputs = [out.name for out in graph.output]
        raise ValueError(
            f"未找到名称为 {target_output_name} 的输出节点\n"
            f"模型中可用的输出节点：{available_outputs}"
        )

    del graph.output[:]  # 清空原有输出
    graph.output.extend(new_outputs)  # 添加保留的输出节点

    # 保存精简后的模型
    try:
        onnx.save(model, args.output)
    except Exception as e:
        raise IOError(f"无法保存输出模型 {args.output}：{str(e)}")

    # 输出结果信息
    print(f"模型精简完成：")
    print(f"  保留输入节点: {[inp.name for inp in graph.input]}")
    print(f"  保留输出节点: {[out.name for out in graph.output]}")
    print(f"  已保存至: {args.output}")


if __name__ == "__main__":
    main()
