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
import logging

if __name__=='__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    precision_aligned_res = np.loadtxt('../data/use_for_precision_align.txt', dtype=np.float32)
    grasp_group_array = np.loadtxt('../out/txt/color_sorted_grasp_group.txt', dtype=np.float32)
    logging.info(f"precision_aligned_res is a numpy array of type {precision_aligned_res.dtype} and shape {precision_aligned_res.shape}")
    logging.info(f"grasp_group_array is a numpy array of type {grasp_group_array.dtype} and shape {grasp_group_array.shape}")
    mae = np.mean(np.abs(precision_aligned_res - grasp_group_array))
    rmse = np.sqrt(np.mean((precision_aligned_res - grasp_group_array) ** 2))

    logging.info(f"平均绝对误差: {mae}")
    logging.info(f"均方根误差: {rmse}")