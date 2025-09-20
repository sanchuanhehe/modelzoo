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

import sys

import torch
import timm
import torch
import torch.onnx
import torchvision.models as models



def convert(pth_path, onnx_file):
    # 修改后
    model = timm.create_model('efficientnetv2_rw_t', pretrained=False)
    # 然后手动加载权重
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))

    dummy_input = torch.randn(1, 3, 288, 288)

    torch.onnx.export(model,
                dummy_input,
                onnx_file,
                input_names=['image'],
                output_names=['output'],
                dynamic_axes=None
    )


if __name__ == "__main__":
    pth_path = sys.argv[1]
    onnx_file = sys.argv[2]
    convert(pth_path, onnx_file)