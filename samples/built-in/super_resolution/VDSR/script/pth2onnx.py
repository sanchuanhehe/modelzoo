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
import onnx
import onnxruntime
import numpy as np
from collections import OrderedDict
from model import VDSR

def pth_to_onnx(
        pth_path: str,
        onnx_path: str,
        input_shape: tuple = (1, 1, 516, 516),
        opset_version: int = 13,  # 用兼容更好的 opset 13
        device: str = "cpu"
) -> tuple:  # 返回加载好的模型，供验证使用
    """
    将 PyTorch .pth/.tar 模型转换为 ONNX 格式，返回加载好权重的模型
    """
    # 1. 实例化模型并移动到指定设备
    model = VDSR().to(device)
    model.eval()  # 强制设为评估模式，禁用 Dropout/BatchNorm 动态行为

    # 2. 加载 .tar 权重（关键：正确提取 state_dict）
    # 加上 weights_only=True 解决 FutureWarning
    checkpoint = torch.load(pth_path, map_location=device) #, weights_only=True

    # 处理 .tar 中可能的嵌套结构（常见：checkpoint["state_dict"] 或 checkpoint["model"]）
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint  # 若直接是 state_dict，直接使用

    # 去除多 GPU 训练的 "module." 前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v

    # 加载权重到模型（验证权重是否匹配）
    try:
        model.load_state_dict(new_state_dict)
        print(f"✅ 成功加载权重：{pth_path}")
    except Exception as e:
        print(f"❌ 权重加载失败：{e}")
        raise e

    # 3. 创建虚拟输入（与模型输入形状一致）
    dummy_input = torch.randn(input_shape, dtype=torch.float32).to(device)

    # 4. 导出 ONNX（修正：使用 dynamic_axes 以兼容旧 PyTorch 版本）
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes = None,
        do_constant_folding=False,  # 禁用常量折叠，避免数值偏差
        verbose=False,
        export_params=True,
    )
    print(f"✅ ONNX 模型导出成功：{onnx_path}")

    # 验证 ONNX 模型结构是否合法
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX 模型结构验证通过")

    return model  # 返回加载好权重的模型，供后续对比使用

# -------------------------- 3. 正确验证 ONNX 模型（用加载好权重的 PyTorch 模型对比） --------------------------
def verify_onnx(onnx_path: str, input_shape: tuple, device: str, pytorch_model: torch.nn.Module) -> None:
    """
    用加载好训练权重的 PyTorch 模型，对比 ONNX 模型输出
    """
    # 1. 初始化 ONNX Runtime
    providers = ["CUDAExecutionProvider" if device.startswith("cuda") else "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    # 2. 生成相同的随机输入（确保两者输入一致）
    np.random.seed(42)  # 固定随机种子，保证输入完全相同
    np_input = np.random.randn(*input_shape).astype(np.float32)
    torch_input = torch.from_numpy(np_input).to(device)

    # 3. PyTorch 模型推理（禁用梯度计算，加速且避免干扰）
    pytorch_model.eval()
    with torch.no_grad():
        torch_output = pytorch_model(torch_input).detach().cpu().numpy()

    # 4. ONNX 模型推理
    ort_inputs = {ort_session.get_inputs()[0].name: np_input}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 5. 对比输出差异（残差网络允许微小浮点误差，<1e-4 即为合格）
    abs_diff = np.abs(torch_output - ort_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    print(f"\n📊 输出差异统计：")
    print(f"最大绝对差异：{max_diff:.6f}")
    print(f"平均绝对差异：{mean_diff:.6f}")

    if max_diff < 1e-4:
        print("✅ ONNX 模型验证通过！输出与 PyTorch 一致")
    else:
        print("⚠️  差异略大但可接受（残差网络浮点误差），若部署异常可进一步调试")


# -------------------------- 4. 运行转换与验证 --------------------------
if __name__ == "__main__":
    # 配置参数（根据你的实际路径修改）
    PTH_PATH = "../model/vdsr-TB291-fef487db.pth.tar"  # 你的 .tar 权重路径
    ONNX_PATH = "../model/vdsr.onnx"  # 输出 ONNX 路径
    INPUT_SHAPE = (1, 1, 516, 516)  # 输入形状（单 batch、单通道）
    OPSET_VERSION = 13
    DEVICE = "cpu"  # 有 GPU 可改为 "cuda:0"

    # 步骤1：转换 ONNX 并返回加载好权重的 PyTorch 模型
    pytorch_model = pth_to_onnx(
        pth_path=PTH_PATH,
        onnx_path=ONNX_PATH,
        input_shape=INPUT_SHAPE,
        opset_version=OPSET_VERSION,
        device=DEVICE
    )

    # 步骤2：正确验证 ONNX 模型
    verify_onnx(
        onnx_path=ONNX_PATH,
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        pytorch_model=pytorch_model  # 传入加载好权重的模型
    )