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
from model import SiameseNetwork

# 定义设备：优先用 CPU（因为是 CPU 版 PyTorch）
device = torch.device("cpu")


net = SiameseNetwork().to(device)

net.load_state_dict(torch.load("../model/siamese_model_weights.pt", map_location=device))
net.eval()

batch_size = 1
channels = 1  # 模型输入通道数=1（灰度图）
height = 100  # 模型设计的输入高度
width = 100  # 模型设计的输入宽度

# 创建严格匹配的示例输入（NCHW格式：batch, channel, height, width）
dummy_img1 = torch.randn(batch_size, channels, height, width).to(device)
dummy_img2 = torch.randn(batch_size, channels, height, width).to(device)

onnx_file_path = "../model/siamese_network.onnx"
torch.onnx.export(
    model=net,
    args=(dummy_img1, dummy_img2),
    f=onnx_file_path,
    input_names=["img1", "img2"],
    output_names=["output1", "output2"],
    dynamic_axes={
        "img1": {0: "batch_size"},
        "img2": {0: "batch_size"},
        "output1": {0: "batch_size"},
        "output2": {0: "batch_size"}
    },
    opset_version=13,
    do_constant_folding=False,
    export_params=True
)
print(f"\n ONNX 模型导出成功！路径：{onnx_file_path}")