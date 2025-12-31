# Copyimg_right (c) ModelZoo. 2025-2025. All img_rights reserved.
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

import cv2
import yaml
import json
import argparse
from tqdm import tqdm
import os
import numpy as np

def main(opt):
    shapes_kv = {}
    input_path = os.path.realpath(opt.data_path + '/val2017/')
    for img_path in tqdm(os.listdir(input_path)):
        # Read image
        paths_name = img_path[:-4]  # 'test1
        
        im0 = cv2.imread(os.path.join(input_path, img_path))
        h0, w0, _ = im0.shape
        shapes_kv[paths_name] = (h0, w0)
        
        r = 640 / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            print(" scoll ")
            im0 = cv2.resize(im0, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        
        im = letterbox(im0, scaleup=False)  # padded resize
        
        im = np.array(im)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        
        img = np.expand_dims(im, axis=0)
        print(paths_name)
        if opt.data_type == "float32":
            print("float32")
            img = img.astype(np.float32)
            img /= 255.0
        img = np.ascontiguousarray(img)  # contiguous
        img.tofile("{}/{}.bin".format(opt.prep_data, paths_name))
    # 写入文件
    with open("shapes.json", "w") as f:
        json.dump(shapes_kv, f, indent=4)  # indent 美化格式


def letterbox(img, target_size=[640, 640], scaleup=True):
    """保持比例缩放图片并填充至目标尺寸"""
    shape = img.shape[:2]

    # 计算缩放比例
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    if not scaleup:
        print("scaleup")
        r = min(r, 1.0)
    print("r:: " , r)
    # 计算填充量（居中填充）
    unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w, h = (target_size[1] - unpad[0]) / 2, (target_size[0] - unpad[1]) / 2
    print("w, h :: " , w, h )
    # 缩放和填充
    if shape[::-1] != unpad:
        img = cv2.resize(img, unpad, interpolation=cv2.INTER_LINEAR)
    img_top, img_bottom = int(round(h - 0.1)), int(round(h + 0.1))
    img_left, img_right = int(round(w - 0.1)), int(round(w + 0.1))
    img = cv2.copyMakeBorder(img, img_top, img_bottom, img_left, img_right,
                             cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model inference')
    parser.add_argument('--data_path', type=str, default="./datasets/coco/images/")
    parser.add_argument('--prep_data', type=str, default='./data/prep_data_aipp')
    parser.add_argument('--data_type', type=str, default="float32", choices=["uint8", "float32"])
    parser.add_argument('--output_path', default="./data/", type=str)
    parser.add_argument('--img_size', nargs='+', type=int, default=640)

    opt = parser.parse_args()

    if not os.path.isdir(opt.prep_data):
        os.makedirs(os.path.realpath(opt.prep_data))
    
    main(opt)

    file_list = os.listdir(opt.prep_data)
    sorted_files = sorted(file_list)
    # 将文件列表保存到文本文件
    with open(os.path.join(opt.output_path , 'file_list.txt'), 'w', encoding='utf-8') as f:
        for item in sorted_files:
            f.write(f"prep_data_aipp/{item}\n")
