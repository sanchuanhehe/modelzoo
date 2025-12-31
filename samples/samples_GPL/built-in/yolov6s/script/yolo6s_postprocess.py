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
import numpy as np
import cv2
import torch
from torchvision.ops import batched_nms
import argparse

# 定义常量：数据类型字节数和输出维度相关参数
FLOAT32_BYTES = 4  # float32类型占4字节
BATCH_SIZE = 1  # 批处理大小
DETECTIONS = 8400  # 检测框数量
PC_OUTPUT_COLS = 85  # PC平台输出列数（含类别和坐标信息）
DEV3403_OUTPUT_COLS = 88  # 3403开发板depico核输出列数（含对齐冗余数据）
VALID_COLS = 85  # 有效数据列数（4 个坐标信息（x、y、w、h）、1 个置信度得分，以及 80 个类别的概率）

# COCO 类别信息
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# YOLO80到COCO90的官方ID映射表
yolo80_to_coco90 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def calculate_scale_and_pad(original_h, original_w, target_w, target_h):
    """计算缩放比例和填充量"""
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    dw, dh = target_w - new_w, target_h - new_h
    pad_w, pad_h = dw // 2, dh // 2
    return scale, pad_w, pad_h


def postprocess(preds, scale, pad_w, pad_h, original_h, original_w,
                conf_threshold=0.03, iou_threshold=0.65):
    """后处理：解析预测结果、过滤、NMS"""
    # 解析预测结果 (1, 8400, 85) -> (8400, 85)
    dets = preds[0] if isinstance(preds, list) else preds
    dets = dets.reshape(-1, VALID_COLS)

    # 解析边界框、目标置信度和类别分数
    boxes = dets[:, :4]  # 中心坐标+宽高 (x_center, y_center, w, h)
    obj_conf = dets[:, 4]
    cls_scores = dets[:, 5:]

    # 计算类别和置信度
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_scores = np.max(cls_scores, axis=1)
    scores = obj_conf * cls_scores

    # 置信度过滤
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if boxes.size == 0:
        return []

    # 转换为左上角+右下角坐标 (x1, y1, x2, y2)
    x_center, y_center, w, h = boxes.T
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    # 调整边界框到原始图像尺寸（去除填充和缩放）
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    # 确保边界框在图像范围内
    x1 = np.clip(x1, 0, original_w)
    y1 = np.clip(y1, 0, original_h)
    x2 = np.clip(x2, 0, original_w)
    y2 = np.clip(y2, 0, original_h)

    # 转换为torch张量进行NMS
    boxes_tensor = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    cls_ids_tensor = torch.tensor(cls_ids, dtype=torch.int64)

    # 执行NMS
    keep = batched_nms(boxes_tensor, scores_tensor, cls_ids_tensor, iou_threshold)

    # 整理结果
    results = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_tensor[idx].numpy()
        score = scores_tensor[idx].item()
        cls_id = cls_ids_tensor[idx].item()
        # 转换为COCO类别ID
        coco_cls_id = yolo80_to_coco90[cls_id] if cls_id < len(yolo80_to_coco90) else -1
        if coco_cls_id == -1:
            continue
        results.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO格式：x, y, width, height
            "score": score,
            "category_id": coco_cls_id
        })

    return results


def run_postprocess(args):
    """执行后处理并生成txt结果文件"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有推理结果文件
    bin_files = [f for f in os.listdir(args.bin_dir) if f.endswith('.bin')]
    if not bin_files:
        print(f"错误: 在 {args.bin_dir} 中未找到任何bin文件")
        return

    # 处理每个推理结果
    total = len(bin_files)
    for idx, bin_file in enumerate(bin_files):
        # 提取图片ID（假设bin文件名与图片名对应，如xxx.bin对应xxx.jpg）
        img_name = bin_file.split("_output_0.bin")[0]
        img_path = os.path.join(args.img_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            print(f"警告: 图片文件 {img_path} 不存在，跳过")
            continue

        # 获取原始图片尺寸
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图片 {img_path}，跳过")
            continue
        original_h, original_w = image.shape[:2]

        # 计算缩放比例和填充量
        target_w, target_h = args.target_size
        scale, pad_w, pad_h = calculate_scale_and_pad(
            original_h, original_w, target_w, target_h
        )

        # 加载推理结果
        bin_path = os.path.join(args.bin_dir, bin_file)
        try:
            preds = np.fromfile(bin_path, dtype=np.float32)
            # 获取文件大小并计算元素数量
            file_size = os.path.getsize(bin_path)
            element_count = file_size // FLOAT32_BYTES
            # 根据不同平台的输出格式进行处理
            if element_count == BATCH_SIZE * DETECTIONS * PC_OUTPUT_COLS:  # PC平台输出
                preds = preds.reshape(BATCH_SIZE, DETECTIONS, PC_OUTPUT_COLS)
            elif element_count == BATCH_SIZE * DETECTIONS * DEV3403_OUTPUT_COLS:  # 3403开发板depico核输出
                # 只取有效数据部分(85列)，忽略对齐产生的冗余数据(3列)
                preds = preds.reshape(BATCH_SIZE, DETECTIONS, DEV3403_OUTPUT_COLS)[:, :, :VALID_COLS]
        except Exception as e:
            print(f"警告: 处理文件 {bin_file} 失败: {e}，跳过")
            continue

        # 后处理
        results = postprocess(
            preds,
            scale=scale,
            pad_w=pad_w,
            pad_h=pad_h,
            original_h=original_h,
            original_w=original_w,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.nms_threshold
        )

        # 保存为txt文件
        output_txt = os.path.join(args.output_dir, f"{img_name}.txt")
        with open(output_txt, 'w') as f:
            for res in results:
                # 格式: category_id score x y width height
                line = f"{res['category_id']} {res['score']:.6f} "
                line += " ".join([f"{x:.6f}" for x in res['bbox']]) + "\n"
                f.write(line)

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {bin_file}")

    print(f"所有后处理结果已保存至: {args.output_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解析YOLO模型推理输出的bin文件并生成检测结果txt')
    parser.add_argument('--bin_dir', default='../out/result_pc/bin', help='存放YOLO模型推理输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/coco2017/val2017', help='原始图片目录')
    parser.add_argument('--output_dir', default='../out/result_pc/txt', help='输出结果txt文件目录')
    parser.add_argument('--nms_threshold', type=float, default=0.65, help='NMS的IOU阈值，默认0.65')
    parser.add_argument('--conf_threshold', type=float, default=0.03, help='置信度过滤阈值，默认0.03')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640], help='模型输入尺寸，格式为 宽 高，默认640 640')

    args = parser.parse_args()

    # 验证输入
    if not os.path.isdir(args.bin_dir):
        print(f"错误: 推理结果目录 {args.bin_dir} 不存在")
        return
    if not os.path.isdir(args.img_dir):
        print(f"错误: 图片目录 {args.img_dir} 不存在")
        return

    # 执行后处理
    run_postprocess(args)


if __name__ == "__main__":
    main()
