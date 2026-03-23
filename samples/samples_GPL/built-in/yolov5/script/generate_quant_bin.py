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
import argparse
import os
import numpy as np


def main(opt):
    if not os.path.exists(opt.data_path):
        print("image not exist!")
        return

    # Read image
    paths_name = opt.data_path.split("/")[-1][:-4]  # 'test1

    im0 = cv2.imread(opt.data_path)
    im = letterbox(im0, scaleup=False)  # padded resize
    im = np.array(im)
    if opt.img_format == "NCHW":
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(im, axis=0)
    if opt.data_type == "float32":
        img = img.astype(np.float32)
        img /= 255.0
    img = np.ascontiguousarray(img)  # contiguous
    img.tofile("./data/{}.bin".format(paths_name))


def letterbox(img, target_size=[640, 640], scaleup=True):
    """保持比例缩放图片并填充至目标尺寸"""
    shape = img.shape[:2]

    # 计算缩放比例
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 计算填充量（居中填充）
    unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w, h = (target_size[1] - unpad[0]) / 2, (target_size[0] - unpad[1]) / 2

    # 缩放和填充
    if shape[::-1] != unpad:
        img = cv2.resize(img, unpad, interpolation=cv2.INTER_LINEAR)
    img_top, img_bottom = int(round(h - 0.1)), int(round(h + 0.1))
    img_left, img_right = int(round(w - 0.1)), int(round(w + 0.1))
    img = cv2.copyMakeBorder(
        img,
        img_top,
        img_bottom,
        img_left,
        img_right,
        cv2.BORDER_CONSTANT,
        value=[114, 114, 114],
    )
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model infer")
    parser.add_argument(
        "--data_path", type=str, default="./datasets/coco/val2017/000000000139.jpg"
    )
    parser.add_argument(
        "--data_type", type=str, default="float32", choices=["uint8", "float32"]
    )
    parser.add_argument("--output_path", default="./data/", type=str)
    parser.add_argument("--img_format", type=str, default="NCHW")
    opt = parser.parse_args()

    main(opt)
