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
import json
import glob
import cv2
import numpy as np
import argparse
import torch
from tqdm import tqdm
from torchvision.ops import batched_nms
from typing import Tuple, List

# YOLO80到COCO90类别的映射
yolo80_to_coco90 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def read_yolo_seg_bin(detections_bin_path: str, protos_bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取YOLOv11-seg输出的两个bin文件，解析为NumPy数组
    input shape (1, 3, 640, 640) NCHW and output shapes ((1, 116, 8400), (1, 32, 160, 160))
    """
    # 读取detections bin（8400×116 float32）
    expected_detections_size = 8400 * 116 * 4
    with open(detections_bin_path, 'rb') as f:
        detections_data = f.read()
        if len(detections_data) != expected_detections_size:
            raise ValueError(
                f"detections bin文件大小异常！预期{expected_detections_size}字节，实际{len(detections_data)}字节"
            )
        detections = np.frombuffer(detections_data, dtype=np.float32).reshape(116, 8400)

    # 读取protos bin（32×160×160 float32）
    expected_protos_size = 32 * 160 * 160 * 4
    with open(protos_bin_path, 'rb') as f:
        protos_data = f.read()
        if len(protos_data) != expected_protos_size:
            raise ValueError(
                f"protos bin文件大小异常！预期{expected_protos_size}字节，实际{len(protos_data)}字节"
            )
        protos = np.frombuffer(protos_data, dtype=np.float32).reshape(32, 160, 160)

    return detections, protos


def get_pad_scale(img, target_size=(640, 640)):
    """
    改进的预处理：使用LetterBox缩放（保持比例+填充）
    返回:
        scale: 缩放比例
        pad: (pad_w, pad_h) 填充尺寸
    """
    shape = img.shape[:2]
    input_h, input_w = target_size[0], target_size[1]

    # 计算缩放比例
    r = min(input_h / shape[0], input_w / shape[1])

    # 计算填充量（居中填充）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (input_w - new_unpad[0]) / 2, (input_h - new_unpad[1]) / 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    return r, (left, top)


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """根据边界框裁剪掩码区域"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    mask_region = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    return masks * mask_region.astype(np.float32)


def processMask(protos: np.ndarray, mask_coeffs: np.ndarray, bboxs: np.ndarray,
                img_shape: Tuple[int, int]) -> np.ndarray:
    """利用原型与系数的线性组合生成基础掩码 → 通过裁剪聚焦目标区域 → 缩放回输入尺度以匹配原图，最终实现从低分辨率原型到高分辨率目标掩码的转换。"""
    # 解析掩码原型的形状：c为通道数，mh和mw为掩码原型的高和宽
    c, mh, mw = protos.shape
    # 解析输入图像的尺寸：ih为图像高度，iw为图像宽度
    ih, iw = img_shape
    # 生成基础掩码：
    # 1. 将掩码原型展平为二维矩阵 [c, mh*mw]（空间维度合并）
    # 2. 通过矩阵乘法（@）用掩码系数（mask_coeffs）对原型进行线性组合
    # 3. 将结果重塑为 [目标数量, mh, mw] 的三维掩码
    masks = (mask_coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)

    # 计算边界框从图像尺寸缩放到掩码原型尺寸的比例
    width_ratio = mw / iw  # 宽度方向缩放比例（掩码宽度/图像宽度）
    height_ratio = mh / ih  # 高度方向缩放比例（掩码高度/图像高度）
    # 复制原始边界框并缩放到掩码原型尺寸（用于后续裁剪）
    downsampled_bbox = bboxs.copy()
    downsampled_bbox[:, [0, 2]] *= width_ratio  # 边界框x1和x2坐标缩放
    downsampled_bbox[:, [1, 3]] *= height_ratio  # 边界框y1和y2坐标缩放

    # 裁剪并缩放掩码
    # 1. 根据缩放后的边界框裁剪掩码，保留边界框内的区域
    masks = crop_mask(masks, downsampled_bbox)
    # 2. 将每个裁剪后的掩码缩放回原始图像尺寸（iw x ih）
    # 使用线性插值（INTER_LINEAR）保证缩放后掩码的平滑性
    masks_resize = np.array([
        cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)
        for m in masks
    ])
    # 返回缩放后的掩码（形状为 [目标数量, ih, iw]）
    return masks_resize


def mask_to_coco_segment(bbox, mask_binary: np.ndarray) -> List[List[float]]:
    """将二值掩码转为COCO格式分割轮廓"""
    x1, y1, x2, y2 = bbox
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_segs = []

    for cnt in contours:
        if len(cnt) >= 3:  # 至少3个点构成多边形
            # 提取轮廓点并转换为列表
            seg = cnt.squeeze().flatten().astype(float).tolist()

            # 确保坐标数为偶数
            if len(seg) % 2 != 0:
                seg = seg[:-1]

            # 将所有点裁剪到边界框内
            clipped_seg = []
            for i in range(0, len(seg), 2):
                # x坐标裁剪到 [x1, x2]
                x = max(x1, min(seg[i], x2))
                # y坐标裁剪到 [y1, y2]
                y = max(y1, min(seg[i + 1], y2))
                clipped_seg.extend([float(x), float(y)])

            # 检查裁剪后的轮廓是否仍有效（至少3个点）
            if len(clipped_seg) >= 6:
                valid_segs.append(clipped_seg)

    return valid_segs if valid_segs else [[]]  # 空轮廓返回[[]]


def postprocess(
        detections: np.ndarray, protos: np.ndarray, orig_shape: Tuple[int, int],
        target_size: Tuple[int, int], scale: float, pad: Tuple[int, int],
        img_id: str, conf_thres: float = 0.001, iou_thres: float = 0.6
) -> List[dict]:
    """后处理生成COCO格式结果"""
    # 提取基础信息
    bboxes_xywh = detections[:, :4]
    scores = detections[:, 4:84]
    mask_coeffs = detections[:, 84:]

    # 过滤低置信度目标
    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    keep = max_scores > conf_thres
    if not np.any(keep):
        return []

    # 筛选有效目标
    bboxes_xywh = bboxes_xywh[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    # 转换为xyxy格式，原始边界框输出是基于模型输入尺寸（640x640）的归一化XYWH格式坐标，边界框中心点X坐标、中心点Y坐标、宽度、高度
    x1 = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2
    y1 = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2
    x2 = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2
    y2 = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2
    bboxes_xyxy_input = np.stack([x1, y1, x2, y2], axis=1)

    # 转换为PyTorch张量，使用batched_nms按类别执行NMS
    bboxes_tensor = torch.tensor(bboxes_xyxy_input, dtype=torch.float32)
    scores_tensor = torch.tensor(max_scores, dtype=torch.float32)
    categories_tensor = torch.tensor(class_ids, dtype=torch.int64)

    # 按类别执行NMS
    keep_indices = batched_nms(
        boxes=bboxes_tensor,
        scores=scores_tensor,
        idxs=categories_tensor,  # 按类别ID分组
        iou_threshold=iou_thres
    ).numpy()  # 转换为numpy索引

    if len(keep_indices) == 0:
        return []
    object_num = len(keep_indices)

    # 筛选NMS后的目标
    bboxes_xyxy_input = bboxes_xyxy_input[keep_indices]
    max_scores = max_scores[keep_indices]
    class_ids = class_ids[keep_indices]
    mask_coeffs = mask_coeffs[keep_indices]

    # 修正边界框到原始尺寸
    orig_h, orig_w = orig_shape
    pad_w, pad_h = pad
    bboxes_xyxy_orig = bboxes_xyxy_input.copy()
    bboxes_xyxy_orig[:, [0, 2]] = np.clip((bboxes_xyxy_orig[:, [0, 2]] - pad_w) / scale, 0, orig_w)
    bboxes_xyxy_orig[:, [1, 3]] = np.clip((bboxes_xyxy_orig[:, [1, 3]] - pad_h) / scale, 0, orig_h)

    coco_results = []
    masks_orig = np.empty((orig_h, orig_w), dtype=np.float32)
    for i in range(object_num):
        # 生成输入尺寸(如640x640)下的掩码, 参数: 掩码原型(protos)、当前目标的掩码系数、输入尺寸下的边界框、模型输入尺寸
        masks_input = processMask(protos, mask_coeffs[i].reshape(1, -1), bboxes_xyxy_input[i].reshape(1, -1),
                                  target_size)
        # 裁剪掩码中预处理时添加的填充区域(上下填充和左右填充)
        masks_cropped = masks_input[:, pad_h:target_size[0] - pad_h, pad_w:target_size[1] - pad_w]
        # 移除批次维度(从[1, H, W]变为[H, W])
        masks_cropped = masks_cropped.squeeze(0)
        # 将裁剪后的掩码缩放到原始图像尺寸(orig_w x orig_h), 使用线性插值保证缩放平滑性
        masks_orig = cv2.resize(masks_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # 将浮点掩码二值化: 大于0的区域为目标(1)，否则为背景(0)
        mask_binary_orig = np.where(masks_orig > 0.0, 1, 0).astype(np.uint8)
        # 提取原始图像上的边界框坐标(XYXY格式)
        x1, y1, x2, y2 = bboxes_xyxy_orig[i]
        # 转换边界框格式为COCO要求的XYWH(左上角x, 左上角y, 宽, 高)
        bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        # 从二值掩码中提取轮廓，并转换为COCO格式的分割坐标列表
        segmentation = mask_to_coco_segment(bboxes_xyxy_orig[i], mask_binary_orig)

        # 确保类别ID有效
        if class_ids[i] >= len(yolo80_to_coco90):
            continue
        coco_cat_id = yolo80_to_coco90[class_ids[i]]
        if segmentation == [[]]:
            continue
        coco_results.append({
            "image_id": int(img_id),  # 图片ID使用文件名数字
            "category_id": coco_cat_id,
            "bbox": bbox_coco,
            "score": float(max_scores[i]),
            "segmentation": segmentation
        })

    return coco_results


def parse_yolo_bin_files(
        bin_dir: str, img_dir: str, output_dir: str,
        conf_threshold: float = 0.001, iou_threshold: float = 0.6,
        target_size: Tuple[int, int] = (640, 640)
) -> None:
    """批量解析bin文件并为每张图片生成独立的结果文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 按图片ID分组bin文件
    bin_groups = {}
    bin_files = glob.glob(os.path.join(bin_dir, "*_output_[01].bin"))

    for bin_path in bin_files:
        bin_name = os.path.basename(bin_path)
        img_id = bin_name.split("_output_")[0]  # 提取图片ID
        if "_output_0.bin" in bin_name:
            bin_groups.setdefault(img_id, [None, None])[0] = bin_path
        elif "_output_1.bin" in bin_name:
            bin_groups.setdefault(img_id, [None, None])[1] = bin_path

    # 过滤缺少成对bin的图片ID
    valid_img_ids = [img_id for img_id, bins in bin_groups.items() if all(bins)]
    print(f"找到 {len(valid_img_ids)} 组有效bin文件")
    if not valid_img_ids:
        raise RuntimeError("未找到有效bin文件组，请检查bin文件命名格式")

    # 批量处理
    for img_id in tqdm(valid_img_ids, desc="处理进度"):
        detections_bin, protos_bin = bin_groups[img_id]
        img_path = os.path.join(img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            print(f"警告：图片 {img_path} 不存在，跳过")
            continue

        # 读取图片和原始尺寸
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

        # 读取bin文件
        try:
            detections, protos = read_yolo_seg_bin(detections_bin, protos_bin)
        except Exception as e:
            print(f"警告：处理bin文件 {detections_bin} 出错：{str(e)}，跳过")
            continue

        # 转置为8400×116结构
        detections = detections.transpose()

        # 后处理生成结果
        results = postprocess(
            detections, protos, (orig_h, orig_w),
            target_size, scale, pad, img_id,
            conf_threshold, iou_threshold
        )

        # 为每张图片保存独立的结果文件
        output_file = os.path.join(output_dir, f"{img_id}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"已处理 {img_id}.jpg，检测到 {len(results)} 个目标，保存至 {output_file}")

    print(f"处理完成，结果已保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='解析YOLO11-Seg输出的bin文件并生成单张图片结果')

    # 必选参数
    parser.add_argument('--bin_dir', default='../out/result_pc/bin', help='存放YOLO输出bin文件的目录')
    parser.add_argument('--img_dir', default='../../../../../datasets/coco2017/val2017', help='存放原始图片的目录')
    parser.add_argument('--output_dir', default='../out/result_pc/json', help='输出单张图片结果的目录')

    # 可选参数
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='置信度过滤阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.7, help='NMS的IOU阈值')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640], help='模型输入尺寸')

    args = parser.parse_args()

    # 验证输入
    for dir_path in [args.bin_dir, args.img_dir]:
        if not os.path.isdir(dir_path):
            print(f"错误：目录 {dir_path} 不存在")
            return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理bin文件
    parse_yolo_bin_files(
        bin_dir=args.bin_dir,
        img_dir=args.img_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        target_size=tuple(args.target_size)
    )


if __name__ == "__main__":
    main()
