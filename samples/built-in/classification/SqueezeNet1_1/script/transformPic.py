import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import sys
import argparse
import json

from PIL import Image
import numpy as np
from tqdm import tqdm

def gen_input_bin(args, file_batches):
    for file in file_batches:
        # RGBA to RGB
        image = Image.open(os.path.join(input_path, file)).convert('RGB')
   
        # 验证预处理
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        res_img = val_transform(image)

        img = res_img.float().numpy()
        bin_data = np.expand_dims(img, axis=0)
        print('shape: ' , bin_data.shape)
        bin_data = bin_data.flatten()
        
        bin_data.tofile(os.path.join(output_path+ '/img/', file.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../../../datasets/ImageNet/val')
    parser.add_argument('--output_path', default='./data')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    input_path = os.path.realpath(input_path)
    output_path = os.path.realpath(output_path)
    if not os.path.isdir(output_path + '/img/'):
        os.makedirs(os.path.realpath(output_path+ '/img/'))
    gen_input_bin(args, os.listdir(input_path))
    # 获取当前目录下的所有文件和文件夹
    file_list = os.listdir(output_path + '/img/')
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join(output_path , 'file_list.txt'), 'w', encoding='utf-8') as f:
        for item in sorted_files:
            f.write(f"img/{item}\n")