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
from collections import OrderedDict


def pth_convert_onnx(pth_path, onnx_file, soc_version):
    inception_v3_model = models.inception_v3(pretrained=False)
    inception_v3 = torch.load(pth_path, map_location="cpu")
    if "state_dict" in inception_v3:
        state_dict = inception_v3["state_dict"]
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            state_dict_new[name] = v
        inception_v3_model.load_state_dict(state_dict_new)
    else:
        inception_v3_model.load_state_dict(inception_v3)

    input_names = ["input_0"]
    output_names = ["output_0"]
    if soc_version == "OPTG":
        dummy_input = torch.randn(1, 3, 304, 304)
    else:
        dummy_input = torch.randn(1, 3, 299, 299)
    dynamic_axes = None
    torch.onnx.export(
        inception_v3_model,
        dummy_input,
        onnx_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )


if __name__ == "__main__":
    pth_path = sys.argv[1]
    onnx_file = sys.argv[2]
    soc_version = sys.argv[3]
    pth_convert_onnx(pth_path, onnx_file, soc_version)
