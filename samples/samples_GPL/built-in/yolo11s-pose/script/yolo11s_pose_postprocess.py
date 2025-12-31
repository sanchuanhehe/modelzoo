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
import os
import glob
import numpy as np
import torch
import argparse
from torchvision.ops import batched_nms


def get_pad_scale(img, target_size=(640, 640)):
    """改进的预处理：使用LetterBox缩放（保持比例+填充）"""
    shape = img.shape[:2]
    input_h, input_w = target_size[0], target_size[1]

    # 计算缩放比例
    r = min(input_h / shape[0], input_w / shape[1])

    # 计算填充量（居中填充）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (input_w - new_unpad[0]) / 2, (input_h - new_unpad[1]) / 2

    # 缩放和填充
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return r, (left, top)


def read_yolo_seg_bin(bin_path):
    """读取YOLOv11-seg模型输出的bin文件，解析为NumPy数组"""
    # 读取detections bin（8400×56 float32）
    expected_detections_size = 8400 * 56 * 4
    try:
        with open(bin_path, 'rb') as f:
            detections_data = f.read()
            if len(detections_data) != expected_detections_size:
                raise ValueError(
                    f"detections bin文件大小异常！预期{expected_detections_size}字节，实际{len(detections_data)}字节"
                )
            detections = np.frombuffer(detections_data, dtype=np.float32).reshape(56, 8400)
    except Exception as e:
        raise RuntimeError(f"读取detections bin文件失败：{str(e)}") from e
    return detections


def parse_yolo_bin_files(bin_dir, img_dir, output_dir,
                         nms_threshold=0.6, conf_threshold=0.001,
                         target_size=(640, 640)):
    """
    解析bin文件，生成每个图片对应的txt结果文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
    print(f"找到 {len(bin_files)} 个bin文件")

    total = len(bin_files)
    for idx, bin_path in enumerate(bin_files):
        base_name = os.path.basename(bin_path).split("_output_0.bin")[0]
        file_name = base_name.replace("_result", "") if "_result" in base_name else base_name
        img_path = os.path.join(img_dir, f"{file_name}.jpg")

        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 不存在，跳过")
            continue

        # 获取图片宽高
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图片 {img_path}，跳过")
                continue
            orig_h, orig_w = img.shape[:2]
        except Exception as e:
            print(f"警告：处理图片 {img_path} 出错：{str(e)}，跳过")
            continue

        # 计算缩放和填充
        scale, pad = get_pad_scale(img, target_size)

        detections = read_yolo_seg_bin(bin_path)
        detections = detections.transpose()  # (56, 8400) → (8400, 56)
        detections = np.squeeze(detections, axis=-1) if detections.ndim == 3 else detections

        # 提取边界框、置信度、关键点
        bboxes = detections[:, :4]  # (N, 4)，xywh格式
        scores = detections[:, 4]  # (N, 1)，置信度
        key_points = detections[:, 5:]  # (N, 51)，17个关键点

        # 1. 过滤低置信度目标
        keep = scores > conf_threshold
        bboxes, scores, key_points = bboxes[keep], scores[keep], key_points[keep]

        if not np.any(keep):
            print(f"警告: {bin_path} 无高置信度目标，跳过")
            continue

        # 2. xywh → xyxy（模型输入尺寸）
        x1 = bboxes[:, 0] - bboxes[:, 2] / 2
        y1 = bboxes[:, 1] - bboxes[:, 3] / 2
        x2 = bboxes[:, 0] + bboxes[:, 2] / 2
        y2 = bboxes[:, 1] + bboxes[:, 3] / 2
        bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # 3. NMS过滤重叠框
        if len(bboxes_xyxy) > 0:
            # 转换为torch tensor
            bboxes_torch = torch.tensor(bboxes_xyxy, dtype=torch.float32)
            scores_torch = torch.tensor(scores, dtype=torch.float32)
            # top-k筛选（保留前1000个高置信度框）
            topk = min(1000, len(scores_torch))
            scores_topk, indices_topk = torch.topk(scores_torch, topk)
            bboxes_topk = bboxes_torch[indices_topk]
            key_points_topk = key_points[indices_topk.numpy()]
            # batched_nms（类别ID设为1）
            class_ids = torch.ones_like(scores_topk, dtype=torch.int64)
            nms_indices = batched_nms(
                bboxes_topk, scores_topk, class_ids, iou_threshold=nms_threshold
            )
            # 提取NMS后的结果
            if len(nms_indices) == 0:
                continue
            bboxes_xyxy = bboxes_topk[nms_indices].numpy()
            scores = scores_topk[nms_indices].numpy()
            key_points = key_points_topk[nms_indices.numpy()]

        processed_keypoints = []
        # 处理每个目标的关键点
        for kpts in key_points:
            kpts = kpts.reshape(17, 3)  # 重塑为(17, 3)
            for i in range(17):
                x, y, score = kpts[i]
                # 还原坐标并过滤低置信度关键点
                if score > 0.1:
                    x_clipped = np.clip((x - pad[0]) / scale, 0, orig_w)
                    y_clipped = np.clip((y - pad[1]) / scale, 0, orig_h)
                    kpts[i] = [x_clipped, y_clipped, score]
                else:
                    kpts[i] = [0, 0, 0]  # 无效点
            kpts = kpts.flatten().tolist()
            processed_keypoints.append(kpts)

        # 4. 修正边界框到原始图像尺寸
        bboxes_xyxy[:, 0] = np.clip((bboxes_xyxy[:, 0] - pad[0]) / scale, 0, orig_w)
        bboxes_xyxy[:, 1] = np.clip((bboxes_xyxy[:, 1] - pad[1]) / scale, 0, orig_h)
        bboxes_xyxy[:, 2] = np.clip((bboxes_xyxy[:, 2] - pad[0]) / scale, 0, orig_w)
        bboxes_xyxy[:, 3] = np.clip((bboxes_xyxy[:, 3] - pad[1]) / scale, 0, orig_h)

        # 保存为txt文件
        output_txt = os.path.join(output_dir, f"{file_name}.txt")
        with open(output_txt, 'w') as f:
            for i in range(len(bboxes_xyxy)):
                x1, y1, x2, y2 = bboxes_xyxy[i]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                # 格式: category_id score x y width height keypoint1_x keypoint1_y keypoint1_score ...
                line = f"1 {scores[i]:.6f} "
                line += " ".join([f"{x:.6f}" for x in bbox]) + " "
                line += " ".join([f"{k:.6f}" for k in processed_keypoints[i]]) + "\n"
                f.write(line)

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {bin_path}")

    print(f"所有后处理结果已保存至: {output_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解析YOLO输出的bin文件并生成关键点检测结果TXT')

    parser.add_argument('--bin_dir', default='../out/result_pc/bin',
                        help='存放YOLO输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/coco2017/val2017',
                        help='存放原始图片的目录')
    parser.add_argument('--output_dir', default='../out/result_pc/txt',
                        help='输出结果TXT文件目录')
    parser.add_argument('--nms_threshold', type=float, default=0.6,
                        help='NMS的IOU阈值，默认0.6')
    parser.add_argument('--conf_threshold', type=float, default=0.001,
                        help='置信度过滤阈值，默认0.001')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                        help='模型输入尺寸，格式为 宽 高，默认640 640')

    args = parser.parse_args()

    # 验证输入目录是否存在
    for dir_path in [args.bin_dir, args.img_dir]:
        if not os.path.isdir(dir_path):
            print(f"错误: 目录 {dir_path} 不存在")
            exit(1)

    # 解析bin文件并生成结果
    parse_yolo_bin_files(
        bin_dir=args.bin_dir,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        nms_threshold=args.nms_threshold,
        conf_threshold=args.conf_threshold,
        target_size=tuple(args.target_size)
    )


if __name__ == "__main__":
    main()
