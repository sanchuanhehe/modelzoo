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

# COCO类别信息
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

# YOLOv4核心配置（与模型训练时保持一致）
ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]  # 608x608
MASK_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]  # 三个输出层对应的锚点掩码
STRIDES = [8, 16, 32]  # 三个输出层的下采样倍数
NUM_CLASSES = 80  # YOLOv4默认80类


def calculate_scale_and_pad(original_h, original_w, target_w, target_h):
    """计算缩放比例和填充量"""
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    dw, dh = target_w - new_w, target_h - new_h
    pad_w, pad_h = dw // 2, dh // 2
    return scale, pad_w, pad_h


def decode_output(output, mask, anchors, stride, num_classes):
    """解码单个输出层的预测结果"""
    batch_size, channels, grid_h, grid_w = output.shape
    num_anchors = len(mask)
    bbox_attrs = 5 + num_classes

    # 重塑输出形状:
    output = output.reshape(batch_size, num_anchors, bbox_attrs, grid_h, grid_w)  # [batch,anchors,attrs,grid_h,grid_w]
    output = output.transpose(0, 1, 3, 4, 2)  # 形状: (batch, anchors, grid_h, grid_w, attrs)

    # 生成网格坐标
    grid_x = np.tile(
        np.arange(grid_w).reshape(1, 1, 1, grid_w),
        (batch_size, num_anchors, grid_h, 1)
    )
    grid_y = np.tile(
        np.arange(grid_h).reshape(1, 1, grid_h, 1),
        (batch_size, num_anchors, 1, grid_w)
    )
    grid_xy = np.stack([grid_x, grid_y], axis=-1)

    # 获取锚点
    masked_anchors = np.array([[anchors[i * 2], anchors[i * 2 + 1]] for i in mask])
    anchor_wh = masked_anchors.reshape(1, num_anchors, 1, 1, 2)

    # 解码边界框
    box_xy = 1.0 / (1.0 + np.exp(-output[..., :2]))  # sigmoid
    box_wh = np.exp(output[..., 2:4])

    box_xy += grid_xy  # 加上网格偏移
    box_xy *= stride  # 乘以步长
    box_wh *= anchor_wh  # 乘以锚点尺寸

    # 解码置信度和类别分数
    conf = 1.0 / (1.0 + np.exp(-output[..., 4]))  # sigmoid
    cls_scores = 1.0 / (1.0 + np.exp(-output[..., 5:]))  # sigmoid

    # 转换为左上角和右下角坐标
    x1 = box_xy[..., 0] - box_wh[..., 0] / 2
    y1 = box_xy[..., 1] - box_wh[..., 1] / 2
    x2 = box_xy[..., 0] + box_wh[..., 0] / 2
    y2 = box_xy[..., 1] + box_wh[..., 1] / 2

    # 展平输出
    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
    conf = conf.reshape(-1)
    cls_scores = cls_scores.reshape(-1, num_classes)

    return boxes, conf, cls_scores


def postprocess(preds, scale, pad_w, pad_h, original_h, original_w,
                conf_threshold=0.001, iou_threshold=0.6):
    """后处理：解析预测结果、过滤、NMS"""
    all_boxes = []
    all_conf = []
    all_cls_scores = []

    # 解码三个输出层
    for i in range(3):
        boxes, conf, cls_scores = decode_output(
            output=preds[i],
            mask=MASK_GROUPS[i],
            anchors=ANCHORS,
            stride=STRIDES[i],
            num_classes=NUM_CLASSES
        )
        all_boxes.append(boxes)
        all_conf.append(conf)
        all_cls_scores.append(cls_scores)

    # 合并所有输出层结果
    boxes = np.concatenate(all_boxes, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    cls_scores = np.concatenate(all_cls_scores, axis=0)

    # 计算最终分数（置信度 × 类别分数）
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_ids)), cls_ids]
    scores = conf * cls_conf

    # 过滤低置信度框
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if boxes.size == 0:
        return []

    # 将边界框从模型输入尺寸映射回原始图像尺寸
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes /= scale

    # 确保边界在图像范围内
    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_h - 1)

    # 转换为torch张量进行NMS
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    cls_ids_tensor = torch.tensor(cls_ids, dtype=torch.int64)

    # 执行按类别NMS
    keep = batched_nms(
        boxes=boxes_tensor,
        scores=scores_tensor,
        idxs=cls_ids_tensor,
        iou_threshold=iou_threshold
    ).numpy()

    # 筛选NMS后的结果
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # 整理结果为COCO格式
    results = []
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if cls_id >= len(yolo80_to_coco90):
            continue
        coco_cls_id = yolo80_to_coco90[cls_id]
        results.append({
            "bbox": [float(box[0]), float(box[1]),
                     float(box[2] - box[0]), float(box[3] - box[1])],  # [x, y, w, h]
            "score": float(score),
            "category_id": coco_cls_id
        })

    return results


def run_postprocess(args):
    """执行后处理并生成txt结果文件"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有推理结果文件（按前缀分组）
    bin_files = [f for f in os.listdir(args.bin_dir) if f.endswith('.bin')]
    if not bin_files:
        print(f"错误: 在 {args.bin_dir} 中未找到任何bin文件")
        return

    # 按图片分组（假设文件命名格式为xxx_output_0.bin, xxx_output_1.bin, xxx_output_2.bin）
    file_groups = {}
    for f in bin_files:
        prefix = f.split('_output_')[0]
        if prefix not in file_groups:
            file_groups[prefix] = []
        file_groups[prefix].append(f)

    # 处理每个图片的三个输出层
    total = len(file_groups)
    for idx, (prefix, files) in enumerate(file_groups.items()):
        # 检查是否有三个输出层文件
        if len(files) != 3:
            print(f"警告: 图片 {prefix} 缺少输出层文件，跳过")
            continue

        # 提取图片ID和路径
        img_name = prefix
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

        # 加载三个输出层的推理结果
        preds = []
        valid = True
        for i in range(3):
            # 查找对应输出层的文件
            layer_file = next((f for f in files if f.endswith(f"_{i}.bin")), None)
            if not layer_file:
                print(f"警告: 图片 {prefix} 缺少输出层 {i} 文件，跳过")
                valid = False
                break

            bin_path = os.path.join(args.bin_dir, layer_file)
            try:
                # 加载二进制文件
                pred = np.fromfile(bin_path, dtype=np.float32)

                # 计算网格尺寸
                grid_size = target_h // STRIDES[i]
                channels = 255  # 255通道=3个锚点×(4个边界框参数+1个目标置信度+80个类别分数)

                # 重塑并裁剪（处理可能的填充）
                actual_w = len(pred) // (1 * channels * grid_size)
                pred_reshaped = pred.reshape(1, channels, grid_size, actual_w)
                if actual_w >= grid_size:
                    pred_cropped = pred_reshaped[:, :, :, :grid_size]
                    preds.append(pred_cropped)
                else:
                    print(f"警告: 输出层 {i} 尺寸不足，跳过图片 {prefix}")
                    valid = False
                    break
            except Exception as e:
                print(f"警告: 处理文件 {layer_file} 失败: {e}，跳过")
                valid = False
                break

        if not valid:
            continue

        # 执行后处理
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
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 图片: {img_name}")

    print(f"所有后处理结果已保存至: {args.output_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解析YOLO模型推理输出的bin文件并生成检测结果txt')
    parser.add_argument('--bin_dir', default='../out/result_pc/bin', help='存放YOLO模型推理输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/coco2017/val2017', help='原始图片目录')
    parser.add_argument('--output_dir', default='../out/result_pc/txt', help='输出结果txt文件目录')
    parser.add_argument('--nms_threshold', type=float, default=0.6, help='NMS的IOU阈值，默认0.6')
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='置信度过滤阈值，默认0.001')
    parser.add_argument('--target_size', type=int, nargs=2, default=[608, 608], help='模型输入尺寸，格式为 宽 高，默认608 608')

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
