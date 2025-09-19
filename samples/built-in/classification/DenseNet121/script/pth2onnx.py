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
import re, sys, torch, torchvision


def export_model(densenet_model, onnx_file):
    # export onnx model  
    input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        densenet_model,
        input,
        onnx_file,
        input_names=['input_0'],
        output_names=['output_0'],
        dynamic_axes=None,
        verbose=False,  # 是否开启打印网络结构
        opset_version=11
    )

def pytorch2onnx(pth_file, onnx_file):
    # conver pytorch model to onnx model
    densenet_model = torchvision.models.densenet121()
    # fix model
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
    state_dict = torch.load(pth_file, map_location='cpu')      
    for key in list(state_dict.keys()):      
        result = pattern.match(key)
        if result:
            new_key = result.group(1) + result.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    densenet_model.load_state_dict(state_dict)
    densenet_model.eval()
    export_model(densenet_model, onnx_file)

if __name__=="__main__":
    pth_file = sys.argv[1]
    onnx_file = sys.argv[2]
    pytorch2onnx(pth_file, onnx_file)