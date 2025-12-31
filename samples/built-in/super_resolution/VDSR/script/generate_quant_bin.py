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

import cv2
import numpy as np
import imgproc


def pad_image(image, pad):
    pad_top, pad_bottom, pad_left, pad_right = pad
    # 进行填充
    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

    return padded_image

def main() -> None:
    preprocess_img_size = (516, 516)
    lr_image_path = "../datasets/Set5/X2/LR/butterfly.png"
    r = 2  # 放大倍数
    lr_img_after_pad_h, lr_img_after_pad_w = preprocess_img_size[0] / r, preprocess_img_size[1] / r

    lr_image_origin = cv2.imread(lr_image_path)
    lr_origin_img_h, lr_origin_img_w = lr_image_origin.shape[:2]
    pad_top = (lr_img_after_pad_h - lr_origin_img_h) // 2
    pad_bottom = lr_img_after_pad_h - lr_origin_img_h - pad_top
    pad_left = (lr_img_after_pad_w - lr_origin_img_w) // 2
    pad_right = lr_img_after_pad_w - lr_origin_img_w - pad_left
    pad = [pad_top, pad_bottom, pad_left, pad_right]
    pad = [int(_) for _ in pad]

    lr_image_after_pad = pad_image(lr_image_origin, pad, )
    lr_image_after_pad = lr_image_after_pad.astype(np.float32) / 255.0
    lr_image_scale_up = imgproc.imresize(lr_image_after_pad, r)  # 上采样图片r倍
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image_scale_up, use_y_channel=False)
    lr_y_image = cv2.split(lr_ycbcr_image)[0]
    lr_y_tensor = np.expand_dims(np.expand_dims(lr_y_image, axis=0), axis=0).astype(np.float32)
    lr_y_tensor.tofile(f"../data/preprocess/bin/butterfly_2.bin")


if __name__ == "__main__":
    main()