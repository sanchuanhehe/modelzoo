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
import torchvision.models as models

def export_onnx(shufflenet_model, onnx_file):
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(shufflenet_model, dummy_input, onnx_file,
        input_names=['input_0'],
        output_names=['output_0'],
        dynamic_axes=None,
        verbose=False,
        opset_version=11
    )

def pytorch2onnx(pth_file, onnx_file):
    shufflenet_model = models.shufflenet_v2_x1_0()     
    shufflenet_model.load_state_dict(torch.load(pth_file, map_location=None))
    shufflenet_model.eval()
    export_onnx(shufflenet_model, onnx_file)

if __name__=="__main__":
    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]
    pytorch2onnx(pth_file, onnx_file)