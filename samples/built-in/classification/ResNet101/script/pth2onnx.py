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
import torchvision.models as models


def convert(pthfile, onnx_file):
    model = models.resnet101()
    resnet101 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(resnet101)
    print(model)

    input_names = ["input_0"]
    output_names = ["output_0"]
    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = None
    torch.onnx.export(
        model, 
        dummy_input,
        onnx_file,
        input_names=input_names, 
        output_names=output_names, 
        dynamic_axes=dynamic_axes,
        opset_version=11)


if __name__ == "__main__":
    pth_path = sys.argv[1]
    onnx_file = sys.argv[2]
    convert(pth_path, onnx_file)