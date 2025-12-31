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

from torchvision import transforms
import os
import argparse
import json
import cv2
import numpy as np

def get_affine_matrix(center, scale_w, scale_h, output_size):
    """计算仿射变换矩阵（原始图像→模型输入图像）"""
    dst_w, dst_h = output_size
    src_w = scale_w * 200.0  # 基于人体尺度的原始宽度
    src_h = scale_h * 200.0  # 基于人体尺度的原始高度

    # 缩放+平移矩阵（保证中心对齐）
    trans = np.array([
        [dst_w / src_w, 0, dst_w * (-center[0] / src_w + 0.5)],
        [0, dst_h / src_h, dst_h * (-center[1] / src_h + 0.5)]
    ], dtype=np.float32)
    return trans

def get_scale_size(image):
    """
    原始项目核心缩放尺寸计算逻辑
    返回：缩放后尺寸(w_resized, h_resized)、图像中心(center)、缩放因子(scale_w, scale_h)
    """
    h, w = image.shape[:2]
    # 图像中心（四舍五入）
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)], dtype=np.float32)

    # 计算最小尺度下的基准尺寸（短边），并确保是64的倍数
    min_input_size = 512

    if w < h:  # 短边为宽
        # 当前尺度下的短边（宽）尺寸
        w_resized = min_input_size
        # 当前尺度下的长边（高）尺寸（按比例+64倍数约束）
        h_resized = int((min_input_size / w * h + 63) // 64 * 64)
        # 人体尺度基准（COCO 200像素）
        scale_w = w / 200.0
        scale_h = (h_resized / w_resized) * (w / 200.0)
    else:  # 短边为高
        # 当前尺度下的短边（高）尺寸
        h_resized = min_input_size
        # 当前尺度下的长边（宽）尺寸（按比例+64倍数约束）
        w_resized = int((min_input_size / h * w + 63) // 64 * 64)
        # 人体尺度基准（COCO 200像素）
        scale_h = h / 200.0
        scale_w = (w_resized / h_resized) * (h / 200.0)

    return (w_resized, h_resized), center, (scale_w, scale_h)


def gen_input_list(args, file_list):
    file_path_list= []
    for file in file_list:
        # RGBA to RGB
        img = cv2.imread(os.path.join(args.input_path, file))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_size, _, _ = get_scale_size(img)
        if output_size == (768, 512):
            # out文件夹相对于图片相对地址
            file_path_list.append(["../" + os.path.join(args.input_path, file)])
    
    # 构建JSON数据结构
    json_data = {"fileList": file_path_list}

    if not os.path.exists("data"):
        os.makedirs("data")

    # 将数据写入JSON文件
    with open("./data/file_list.json", "w") as f:
        json.dump(json_data, f, indent=4)

    print("JSON文件生成完成！共包含{}个文件路径。".format(len(file_path_list)))

    img = cv2.imread(file_path_list[0][0][3:]) # 去掉../
    (w_resized, h_resized), center, (scale_w, scale_h) = get_scale_size(img)
    trans = get_affine_matrix(center, scale_w, scale_h, (w_resized, h_resized))
    image_resized = cv2.warpAffine(
        img, trans, (w_resized, h_resized),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
        
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    res_img = val_transform(image_resized)

    img = res_img.float().numpy()

    img.tofile('./data/quant.bin')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../../../datasets/coco2017/val2017')
    parser.add_argument('--output_path', default='./data')

    args = parser.parse_args()

    gen_input_list(args, sorted(os.listdir(args.input_path)))