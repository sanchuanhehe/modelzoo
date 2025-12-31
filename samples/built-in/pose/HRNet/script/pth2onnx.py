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
import argparse
import torch

sys.path.append(r"./HigherHRNet-Human-Pose-Estimation")
sys.path.append(r"./HigherHRNet-Human-Pose-Estimation/lib")

from lib.config import update_config
from lib.models import pose_higher_hrnet
from lib.config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
                
    parser.add_argument('--cfg', required=True, type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--hw',
                        help='输入整数数组（用空格分隔，如：--hw 512 512)',
                        nargs=2,         # 接收2个元素
                        type=int)        # 元素类型为整数

    parser.add_argument('--input', required=True, type=str)

    parser.add_argument('--output', required=True, type=str)

    args = parser.parse_args()
    update_config(cfg, args)

    return args

def pth2onnx():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    model = pose_higher_hrnet.get_pose_net(cfg, False)
    model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, args.hw[0], args.hw[1])
    torch.onnx.export(model, dummy_input, args.output, input_names = ["image"], dynamic_axes = None, output_names = ["class"], verbose=True, opset_version=11)

if __name__ == "__main__":
    pth2onnx()
