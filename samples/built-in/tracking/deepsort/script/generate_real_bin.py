# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成真实量化校准数据集 (.bin)
读取 MOT16 图像并根据 YOLOv5s 和 ReID 的输入格式预处理。
"""

import os
import glob
import cv2
import numpy as np

DATASET_ROOT = "../datasets/MOT16/train"
OUTPUT_DIR = "../datasets/prep_data_aipp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 寻找几张真实图片
image_paths = glob.glob(os.path.join(DATASET_ROOT, "*", "img1", "*.jpg"))
if not image_paths:
    print(f"[错误] 未找到图片, 请检查路径: {DATASET_ROOT}")
    exit(1)

# 挑选前 10 张图片作为校准
sample_paths = image_paths[:10]
print(f"找到 {len(image_paths)} 张图片，选取 10 张进行预处理...")

yolo_bin_path = os.path.join(OUTPUT_DIR, "yolo_real.bin")
reid_bin_path = os.path.join(OUTPUT_DIR, "reid_real.bin")

# 打开文件以追加写入
with open(yolo_bin_path, "wb") as fyolo, open(reid_bin_path, "wb") as freid:
    for path in sample_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 1. 预处理 YOLOv5s (1x3x640x640) [0, 1] FP32 NCHW
        # letterbox: 保持长宽比缩放并在四周 pad
        # ==========================================
        h, w = img_rgb.shape[:2]
        r = min(640 / h, 640 / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = (640 - new_unpad[0]) / 2, (640 - new_unpad[1]) / 2

        # resize
        if img_rgb.shape[:2] != new_unpad[::-1]: 
            img_yolo = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_yolo = img_rgb.copy()
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_yolo = cv2.copyMakeBorder(img_yolo, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # to NCHW float32 [0, 1]
        img_yolo = img_yolo.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB? wait, it is already rgb. -> no [::-1]
        # Since I did cvtColor to RGB above, transpose is enough.
        img_yolo_chw = img_yolo.transpose((2, 0, 1))  # 此时已经是 RGB HWC, 所以这里转成 CHW 
        # (Wait, transpose above has a bug. Let's rewrite safely)
        pass

        # 安全转换 YOLO
        img_yolo_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_yolo_rgb.shape[:2]
        r = min(640 / h, 640 / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        dw, dh = (640 - new_w) / 2, (640 - new_h) / 2
        img_rs = cv2.resize(img_yolo_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_pad = cv2.copyMakeBorder(img_rs, int(round(dh - 0.1)), int(round(dh + 0.1)), 
                                     int(round(dw - 0.1)), int(round(dw + 0.1)), 
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        yolo_tensor = img_pad.astype(np.float32) / 255.0
        yolo_tensor = np.transpose(yolo_tensor, (2, 0, 1))  # HWC -> CHW
        fyolo.write(yolo_tensor.tobytes())

        # ==========================================
        # 2. 预处理 ReID (1x3x128x64) Normalized FP32 NCHW
        # 通常是直接 resize 到 64x128 (width=64, height=128)
        # 归一化参数：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # ==========================================
        # 由于我们使用全图作为假目标直接传入特征提取，其实不够严谨，但为了量化校准跑通是够用的
        img_reid = cv2.resize(img_yolo_rgb, (64, 128), interpolation=cv2.INTER_LINEAR)
        img_reid_f32 = img_reid.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_reid_norm = (img_reid_f32 - mean) / std
        
        reid_tensor = np.transpose(img_reid_norm, (2, 0, 1)) # CHW
        freid.write(reid_tensor.tobytes())

print(f"[成功] 生成真实图片校准库: {yolo_bin_path}, {reid_bin_path}")
