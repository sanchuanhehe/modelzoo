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

import numpy as np
import os
import shutil

import torch
from torchvision import transforms
from dataset.datasets import WLFWDatasets
from pathlib import Path
from torch.utils.data import DataLoader

def main():
    file_list = './data/test_data/list.txt'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = WLFWDatasets(file_list, transform)
    dataloader = DataLoader(dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)
    i = 1
    outDir = '../data/img'
    print(outDir)
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.mkdir(outDir)
    for path, img, landmark, attribute, euler_angle in dataloader:
        print("i ", i)
        i = i+1
        img_path = Path(path[0]).stem
        img.numpy().astype(np.float32).tofile("{}/{}.bin".format(outDir, img_path))

if __name__ == "__main__":
    main()
    file_list = os.listdir("../data/img/")
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join("../data" , 'file_list.txt'), 'w', encoding='utf-8') as f:
        for item in sorted_files:
            f.write(f"img/{item}\n")