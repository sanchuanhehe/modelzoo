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

import cv2
import numpy as np 
from numpy.typing import NDArray


def draw_mask(image: NDArray, masks: NDArray, image_path=''):
    for i, mask in enumerate(masks):
        weight = mask.astype(np.float32)[..., None] * 0.5
        mask_tempalte = np.ones_like(image, dtype=np.float32) * np.array([170, 20, 145], dtype=np.float32)
        result = (1 - weight) * image.astype(np.float32) + weight * mask_tempalte
        result = result.astype(np.uint8)
        cv2.imwrite(image_path.replace('.jpg', f'_{i}.jpg'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def draw_boxes(image, boxes, image_name, output_path):
    output_image = os.path.join(output_path, f'boxes_{image_name}')
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    cv2.imwrite(output_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
