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

import numpy as np
import os
import torch
import torch.onnx

from depth_anything_v2.dpt import DepthAnythingV2

def main():
    encoder = "vits"
    encoder_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }
    deploy_shape = 1, 3, 518, 518

    rand_input = torch.randn(deploy_shape)
    model = DepthAnythingV2(**encoder_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))

    model = model.eval()

    model(rand_input)

    # Export the PyTorch model to ONNX format
    torch.onnx.export(model,
                      torch.randn(deploy_shape).to('cpu'),
                      '../model/depth_anything_v2_vits.onnx',
                      opset_version=12, input_names=["input"],
                      output_names=["output"],
                      do_constant_folding=True)

    print("Model exported to ../model/depth_anything_v2_vits.onnx")

if __name__ == "__main__":
   main()