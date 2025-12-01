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
import torch.onnx
import os
from Se_resnet50.senet.se_resnet import se_resnet50

classes_num = 1000 # 图片集类别数量

def pth_convert_onnx(pth_path, onnx_file):
    model = se_resnet50(num_classes=classes_num)
    checkpoint = torch.load(pth_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    input_names = ["input_0"]
    output_names = ["output_0"]
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )
    print(f"Model successfully converted to ONNX: {onnx_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_seresnet50.py <pth_file_path> <onnx_output_path>")
        sys.exit(1)
    pth_path = sys.argv[1]
    onnx_file = sys.argv[2]
    pth_convert_onnx(pth_path, onnx_file)
