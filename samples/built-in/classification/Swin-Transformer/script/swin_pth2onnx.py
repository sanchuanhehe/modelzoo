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

from collections import OrderedDict
import sys
import torch

sys.path.append("./Swin-Transformer")
from main import parse_option
from models import build_model


def get_model_dict(checkpoint, AttrName):
    model_dict = OrderedDict()
    for key, value in checkpoint[AttrName].items():
        name = key
        if "module." in key:
            name = key.replace("module.", "")
        model_dict[name] = value
    return model_dict


def get_onnx_model(config):
    onnx_model = build_model(config)
    resume = torch.load(config.MODEL.RESUME, map_location="cpu")
    onnx_model.load_state_dict(get_model_dict(resume, "model"))
    onnx_model.cpu()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        onnx_model,
        dummy_input,
        "./model/swin.onnx",
        input_names=["image"],
        output_names=["class"],
        opset_version=11,
        verbose=True,
    )
    print("model saved successful")


if __name__ == "__main__":
    _, opts = parse_option()
    get_onnx_model(opts)
