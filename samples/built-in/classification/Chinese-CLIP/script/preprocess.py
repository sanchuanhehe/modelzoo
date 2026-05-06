
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

import torch
import sys
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append('./Chinese-CLIP')
from cn_clip.clip.utils import image_transform
from cn_clip.eval.cvinw_zeroshot_templates import (
    cifar100_templates,
)
import numpy as np
from cn_clip.eval.data import get_zeroshot_dataset, _preprocess_text
import os
from tqdm import tqdm
from cn_clip.clip import tokenize
from cn_clip.eval.zeroshot_evaluation import parse_args


def get_txt_pre(classnames, templates, context_length):
    i= 0
    for cn_name in tqdm(classnames):
        print(" cn_name: " , cn_name, "i: ", i)
        # 将关键字填入模板中，然后加密
        template_texts = [_preprocess_text(template(cn_name)) for template in templates]
        tokenize_texts = tokenize(template_texts, context_length=context_length).to('cpu')
        j = 0
        for template in templates:
            one_text = np.array(tokenize_texts[j].to(torch.float32).unsqueeze(0))
            one_text.tofile(os.path.join("./data/text/", str(i) +"_"+ str(j) + ".bin"))
            j = j + 1
        i = i + 1
        if i == 1:
            return

def get_image_pre(dataloader, index):
    total_targets = []
    i = 0
    for images, target in tqdm(dataloader):
        images = images.to('cpu')
        img = np.array(images).astype(np.float32)
        img.tofile(os.path.join("./data/img/", str(i) + ".bin"))
        target = target.to('cpu')
        tar = np.array(target).astype(np.int8)
        tar.tofile(os.path.join("./data/target/", str(i) + ".bin"))
        total_targets.append(target)
        i = i + 1
        if i == 1:
            return

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir("./data/img/"):
        os.makedirs(os.path.realpath("./data/img/"))
    if not os.path.isdir("./data/target/"):
        os.makedirs(os.path.realpath("./data/target/"))
    if not os.path.isdir("./data/text/"):
        os.makedirs(os.path.realpath("./data/text/"))
    f = open(args.label_file, "r", encoding="utf8")
    classnames = [line.strip() for line in f.readlines()]
    data = {}
    data[args.dataset] = get_zeroshot_dataset(
        args, image_transform(224)
    )
    get_txt_pre(classnames, cifar100_templates, 512)
    get_image_pre(data[args.dataset].dataloader, 1)
    
