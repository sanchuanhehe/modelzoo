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
from unet import *

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    local_model = torch.load(input_file, map_location="cpu")
    model.load_state_dict(local_model)
    print('load_state_dict sucess')
    model.eval()

    input_names = ["actual_input_1"]
    print('input_names: ', input_names)
    output_names = ["output1"]
    print('output_names: ', output_names)

    torch.onnx.export(model,
                      torch.randn(1, 3, 572, 572),
                      output_file,
                      input_names = input_names,
                      dynamic_axes = None,
                      output_names = output_names,
                      opset_version=11)
