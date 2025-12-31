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
import glob
import argparse
import torch
import numpy as np
import cv2
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from ultralytics.utils.metrics import ap_per_class

# 类别名称
CLASS_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
]

# 进程数 (建议设置为 CPU 核心数)
NUM_WORKERS = 6


# --- 核心算法 ---
def _get_covariance_matrix(boxes: torch.Tensor):
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def batch_probiou(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # 保持在 CPU 计算
    obb1 = obb1.cpu()
    obb2 = obb2.cpu()
    
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def match_predictions(pred_classes, true_classes, iou, iouv):
    """
    匹配预测框和GT
    注意：这里现在返回 Numpy 数组，而不是 Tensor
    """
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    iou = iou.numpy() # 确保转为 numpy
    
    for i, threshold in enumerate(iouv.tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            
    # 【关键修改】返回 Numpy 数组，避免多进程传递 Tensor 报错
    return correct 

def poly2obb_np(poly_pts):
    obbs = []
    for pts in poly_pts:
        pts = pts.reshape(4, 2).astype(np.float32)
        ((cx, cy), (w, h), angle) = cv2.minAreaRect(pts)
        r = angle * math.pi / 180.0
        obbs.append([cx, cy, w, h, r])
    return np.array(obbs, dtype=np.float32)

# --- 单个图像处理函数 (Worker) ---
def process_one_image(img_path, iouv_list, class_names, gt_label_dir, pred_txt_dir):
    # iouv_list 传入 list 避免 tensor 共享
    iouv_tensor = torch.tensor(iouv_list) 
    name2id = {name: i for i, name in enumerate(class_names)}
    
    basename = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. 读取图片尺寸
    img = cv2.imread(img_path)
    if img is None: 
        return None
    h, w = img.shape[:2]

    # 2. 加载 GT
    gt_path = os.path.join(gt_label_dir, basename + ".txt")
    tcls, tobb = [], []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9: continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                poly = [
                    coords[0] * w, coords[1] * h,
                    coords[2] * w, coords[3] * h,
                    coords[4] * w, coords[5] * h,
                    coords[6] * w, coords[7] * h
                ]
                tcls.append(cls_id)
                tobb.append(poly)
    
    # 3. 加载 Pred
    pred_path = os.path.join(pred_txt_dir, basename + ".txt")
    pcls, pobb, pconf = [], [], []
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10: continue
                coords = [float(x) for x in parts[:8]]
                class_name = parts[-2]
                score = float(parts[-1])
                if class_name in name2id:
                    pobb.append(coords)
                    pcls.append(name2id[class_name])
                    pconf.append(score)

    if len(tcls) == 0 and len(pcls) == 0:
        return None

    # 转 Tensor 进行计算 (计算仍用 PyTorch，但返回用 Numpy)
    if len(tcls) > 0:
        tobb_xywhr = poly2obb_np(np.array(tobb))
        tbox = torch.tensor(tobb_xywhr, dtype=torch.float32)
        tcls_tensor = torch.tensor(tcls, dtype=torch.float32)
    else:
        tbox = torch.zeros(0, 5)
        tcls_tensor = torch.zeros(0)

    if len(pcls) > 0:
        pobb_xywhr = poly2obb_np(np.array(pobb))
        pbox = torch.tensor(pobb_xywhr, dtype=torch.float32)
        pcls_tensor = torch.tensor(pcls, dtype=torch.float32)
        pconf_tensor = torch.tensor(pconf, dtype=torch.float32)
    else:
        pbox = torch.zeros(0, 5)
        pcls_tensor = torch.zeros(0)
        pconf_tensor = torch.zeros(0)

    # 4. 评估逻辑
    # 【关键修改】所有返回值必须转为 Numpy 数组
    
    # 情况 A: 只有预测，全是 FP
    if len(tcls) == 0:
        if len(pcls) > 0:
            fp = np.zeros((len(pcls), len(iouv_list)), dtype=bool)
            return (fp, np.array(pconf), np.array(pcls), np.array(tcls))
        return None
    
    # 情况 B: 只有 GT，全是 FN
    if len(pcls) == 0:
        if len(tcls) > 0:
            return (np.zeros((0, len(iouv_list)), dtype=bool), 
                    np.array([]), np.array([]), np.array(tcls))
        return None

    # 情况 C: 正常计算
    iou = batch_probiou(tbox, pbox)
    matches_np = match_predictions(pcls_tensor, tcls_tensor, iou, iouv_tensor)
    
    return (matches_np, np.array(pconf), np.array(pcls), np.array(tcls))

def main():
    parser = argparse.ArgumentParser(description='YOLOv8-OBB mAP Evaluate')
    parser.add_argument('--result_dir', required=True, help='Prediction TXT directory')
    parser.add_argument('--gt_annotations', required=True, help='Ground Truth Labels directory')
    parser.add_argument('--img_dir', required=True, help='Images directory (for image list and size)')
    
    args = parser.parse_args()
    
    pred_txt_dir = args.result_dir
    gt_label_dir = args.gt_annotations
    img_dir = args.img_dir

    print(f"====== mAP 评估 (多进程稳定版: {NUM_WORKERS} Workers) ======")
    
    iouv = torch.linspace(0.5, 0.95, 10)
    # 将 Tensor 转为 list 传入子进程，避免共享内存问题
    iouv_list = iouv.tolist()
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*")))
    
    if not img_files:
        print("未找到图片文件。")
        return

    print(f"发现 {len(img_files)} 张图片")
    
    # 过滤：只保留在 result_dir 中有对应 txt 的图片
    pred_basenames = set(os.path.splitext(f)[0] for f in os.listdir(pred_txt_dir) if f.endswith('.txt'))
    img_files = [f for f in img_files if os.path.splitext(os.path.basename(f))[0] in pred_basenames]
    
    if not img_files:
        print("未找到与预测结果匹配的图片文件。")
        return
        
    print(f"过滤后剩余 {len(img_files)} 张待评估图片 (基于预测结果), 开始并行处理...")

    # 使用 partial 固定参数
    worker_func = partial(process_one_image, iouv_list=iouv_list, class_names=CLASS_NAMES, 
                          gt_label_dir=gt_label_dir, pred_txt_dir=pred_txt_dir)

    stats = []
    
    # 启动进程池
    # 使用 'spawn' 可能会更慢，默认 'fork' 配合 Numpy 传递是最快的
    with Pool(NUM_WORKERS) as p:
        results_iter = p.imap(worker_func, img_files, chunksize=10)
        
        for res in tqdm(results_iter, total=len(img_files)):
            if res is not None:
                stats.append(res)

    # 汇总结果
    if not stats:
        print("未收集到有效统计数据。")
        return

    print("\n正在汇总数据并计算 mAP...")
    
    # 将 Numpy 列表转回 Tensor (这一步在主进程做，很安全)
    # zip(*stats) 将 [(tp, conf, pcls, tcls), ...] 转为 ([tp...], [conf...], ...)
    unzipped = list(zip(*stats))
    
    tp = torch.from_numpy(np.concatenate(unzipped[0], axis=0)).float() # 转 float 防止减法报错
    conf = torch.from_numpy(np.concatenate(unzipped[1], axis=0))
    pred_cls = torch.from_numpy(np.concatenate(unzipped[2], axis=0))
    target_cls = torch.from_numpy(np.concatenate(unzipped[3], axis=0))

    # 计算 AP
    results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names=dict(enumerate(CLASS_NAMES)))
    ap, unique_classes = results[5], results[6]
    
    map50 = ap[:, 0].mean()
    map50_95 = ap.mean()
    
    print("\n" + "="*60)
    print(f"{'Class':<20} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<15}")
    print("-" * 60)
    for i, c in enumerate(unique_classes):
        print(f"{CLASS_NAMES[int(c)]:<20} | {ap[i, 0]:.4f}     | {ap[i, :].mean():.4f}")
    print("-" * 60)
    print(f"{'ALL':<20} | {map50:.4f}     | {map50_95:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()