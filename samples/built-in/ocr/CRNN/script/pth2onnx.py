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

import os
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from box import Box
import CRNN_Chinese_Characters_Rec.lib.models.crnn as crnn


HEIGHT = 32
WIDTH_ORIGIN = 280
WIDTH = 160
NUM_HIDDEN = 256
NUM_CLASSES = 6735


def convert():
    config_dict = {
        'MODEL': {'NAME': 'crnn', 'IMAGE_SIZE': {'OW': WIDTH_ORIGIN, 'H': HEIGHT, 'W': WIDTH}, 'NUM_CLASSES': NUM_CLASSES, 'NUM_HIDDEN': NUM_HIDDEN}}
    checkpoint_filepath = "../model/mixed_second_finetune_acc_97P7.pth"
    device = torch.device('cpu')
    model = crnn.get_crnn(Box(config_dict)).to(device)
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    torch.onnx.export(model, torch.randn(1, 1, 32, 160), '../model/crnn.onnx', input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    convert()
