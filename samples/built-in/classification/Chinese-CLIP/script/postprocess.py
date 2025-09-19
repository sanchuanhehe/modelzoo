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
from tqdm import tqdm
import sys

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np

sys.path.append('./Chinese-CLIP')
from cn_clip.clip.utils import image_transform
from cn_clip.eval.data import get_zeroshot_dataset
from cn_clip.eval.zeroshot_evaluation import parse_args, accuracy

def del_txt(classnames):
    i = 0
    zeroshot_weights = []
    for classname in tqdm(classnames):
        j = 0
        text_infer = []
        for j in range(11):
            out_filepath = "./out/result/txt/"+ str(i) + "_" + str(j) + "_0.bin"
            print("out_filepath ", out_filepath)
            inference_result = torch.tensor(np.fromfile(out_filepath, dtype=np.float32))
            inference_result = inference_result.reshape((1,512))# x(bs, 255, H, W) 
            #print("inference_result: " , inference_result)
            text_infer.append(inference_result)
        i = i + 1
        combined_tensors = torch.cat(text_infer, dim=0)
        #print("text_infer: " , combined_tensors.shape)
        combined_tensors /= combined_tensors.norm(dim=-1, keepdim=True)
        combined_tensor = combined_tensors.mean(dim=0)
        combined_tensor /= combined_tensor.norm()
        zeroshot_weights.append(combined_tensor)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to('cpu')
    return zeroshot_weights

def load_zeroshot_weights():
    """
    从二进制文件加载 zeroshot_weights
    
    参数:
    file_path: 二进制文件路径
    original_shape: 原始张量的形状
    
    返回:
    恢复的 PyTorch 张量
    """
    original_shape = (512, 100)  # 例如 (512, 1000)

    # 使用 NumPy 从文件读取数据
    data = np.fromfile(os.path.join("./data/textout/", "zero.bin"), dtype=np.float32)
    
    # 重塑为原始形状
    data = data.reshape(original_shape)
    
    # 转换为 PyTorch 张量
    tensor_data = torch.from_numpy(data)
    
    return tensor_data

def del_image(classifier, dataloader):
    total_logits = []
    total_targets = []
    i = 0
    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader):
            out_filepath = "./out/result/jpg/"+ str(i) + "_0.bin"
            print("out_filepath ", out_filepath)
            inference_result = torch.tensor(np.fromfile(out_filepath, dtype=np.float32))
            image_features = inference_result.reshape((1,512))# x(bs, 255, H, W) 
            image_features /= image_features.norm(dim=-1, keepdim=True)
            model_logits = (100.0 * image_features @ classifier).softmax(dim=-1)
            total_logits.append(model_logits)
            acc1, acc5 = accuracy(model_logits, target, topk=(1, 1))
            print("run acc1: ", acc1)
            top1 += acc1
            i = i+1
            n += images.size(0)
            
    top1 = top1 / n
    print(" accurary: " , top1)
    return top1


if __name__ == "__main__":
    args = parse_args()
    f = open(args.label_file, "r", encoding="utf8")
    classnames = [line.strip() for line in f.readlines()]
    data = {}
    data[args.dataset] = get_zeroshot_dataset(
        args, image_transform(224)
    )

    # Make inference and evaluation
    print('Using classifier')
    text_out = del_txt(classnames)
    del_image(text_out, data[args.dataset].dataloader)
