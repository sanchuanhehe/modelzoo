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

import scipy.io as scio
import logging

if __name__=='__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # 读取相机参数
    camera_para = scio.loadmat('../data/meta.mat')
    intrinsic_matrix = camera_para['intrinsic_matrix']
    depth_scale_factor = camera_para['factor_depth']

    with open('../data/camera_params.txt', 'w') as f:
        f.write("# 内参矩阵fx: 相机在x轴方向的焦距（像素单位）\n")
        f.write(f"fx={intrinsic_matrix[0][0]}\n")
        f.write("# 内参矩阵fy: 相机在y轴方向的焦距（像素单位）\n")
        f.write(f"fy={intrinsic_matrix[1][1]}\n")
        f.write("# 内参矩阵cx: 主点在x轴方向的坐标（像素）\n")
        f.write(f"cx={intrinsic_matrix[0][2]}\n")
        f.write("# 内参矩阵cy: 主点在y轴方向的坐标（像素）\n")
        f.write(f"cy={intrinsic_matrix[1][2]}\n")
        f.write("# 深度缩放因子: 将深度图像像素值转换为真实世界距离的缩放因子，深度值除以这个值得到真实深度\n")
        f.write(f"depth_scale_factor={depth_scale_factor[0][0]}\n")

    logging.info(f"相机参数解析成功，解析参数已保存至 data/camera_params.txt")
