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
import os
import shutil
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

datasets_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "PFLD-pytorch",
    "dataset"
)

# 将datasets.py所在路径加入搜索路径
if datasets_dir not in sys.path:
    sys.path.append(datasets_dir)

# 直接导入
from datasets import WLFWDatasets


def gen_input_bin():
    file_list = '../data/wflw_test_cropped/list.txt'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = WLFWDatasets(file_list, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    i = 0
    batch = 20
    outDir = '../data/quant'
    res_image = np.zeros(shape=[batch, 3, 112, 112])
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.mkdir(outDir)
    for path, img, landmark, attribute, euler_angle in dataloader:
        res_image[i, :] = img
        if i == batch - 1:
            break
        i = i + 1
    img = np.array(res_image).astype(np.float32)
    img_2d = img.reshape(-1)
    np.savetxt('../data/quant/quant.txt', img_2d, fmt='%.6f', delimiter=' ')
    print(f"save quant data to ../data/quant/quant.txt")


if __name__ == '__main__':
    gen_input_bin()
