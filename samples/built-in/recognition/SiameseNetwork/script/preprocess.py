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
import itertools
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

datasets_dir = r"../datasets/faces"
bin_dir = Path(r"../data/preprocess/bin")
bin_dir.mkdir(parents=True, exist_ok=True)
txt_dir = Path(r"../data")
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

def get_bin(img_path):
    img = Image.open(img_path).convert("L")
    img = transform(img)
    img_np = img.numpy().astype(np.float32).flatten()
    return img_np

img_list = []
for root, dirs, files in os.walk(os.path.join(datasets_dir, "testing")):
    for file in files:
        img_list.append(os.path.join(root, file).replace("\\", "/"))
img_pair = list(itertools.combinations(img_list, 2))


img_pair_mapping_txt = "pair_index,img0_original_path,img1_original_path,img0_bin_filename,img1_bin_filename,label\n"
for pair_index, pair in tqdm(enumerate(img_pair), total=len(img_pair)):
    img0_original_path = pair[0]
    img1_original_path = pair[1]
    label = 0 if img0_original_path.split("/")[-2] == img1_original_path.split("/")[-2] else 1
    img0_bin_filename = f"{img0_original_path.split('/')[-2]}_{img0_original_path.split('/')[-1].split('.')[0]}.bin"
    img1_bin_filename = f"{img1_original_path.split('/')[-2]}_{img1_original_path.split('/')[-1].split('.')[0]}.bin"
    get_bin(img0_original_path).tofile(bin_dir / img0_bin_filename)
    get_bin(img1_original_path).tofile(bin_dir / img1_bin_filename)

    img_pair_mapping_txt += f"{pair_index},{img0_original_path},{img1_original_path},preprocess/bin/{img0_bin_filename},preprocess/bin/{img1_bin_filename},{label}\n"

with open(f"{txt_dir}/file_list.txt", "w") as f:
    f.write(img_pair_mapping_txt)
