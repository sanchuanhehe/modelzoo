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
import cv2
import os

img_list = ['23498078_3447636521.jpg', '31093296_629571590.jpg', '20890750_160601314.jpg', '30811296_2216675829.jpg',
            '21029625_1100953816.jpg', '21259312_2793608221.jpg', '29791875_2415756793.jpg', '31189562_2750289280.jpg',
            '29099562_353962762.jpg', '29550703_1059658537.jpg', '24530875_1147736728.jpg', '21303734_2984392279.jpg',
            '29696125_3807376125.jpg', '26862703_1555783028.jpg']

WIDTH = 32
HEIGHT_ORIGIN = 280
HEIGHT = 160
NORMAL_VALUE = 255
MEAN = 0.588
STD = 0.193


def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height = image.shape[0]
    scale_factor = WIDTH / height
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    scale_factor = HEIGHT / HEIGHT_ORIGIN
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=1, interpolation=cv2.INTER_CUBIC)

    image = image.astype(np.float32)
    image = (image / NORMAL_VALUE - MEAN) / STD

    height, width = image.shape
    return np.reshape(image, (1, 1, height, width))


def main():
    img_dir = '../datasets/Chinese'
    for img_path in img_list:
        file_path = os.path.join(img_dir, img_path)
        input_data = preprocess(file_path)
        input_data.tofile(os.path.join(
            "../data/preprocess/bin",
            img_path.replace(".jpg", ".bin")))


if __name__ == '__main__':
    main()
