import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# 定义类别ID到名称的映射
CLASS_MAPPING = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

# 定义不同类别的颜色
COLORS = [
    (0, 255, 0),    # 绿色 (Class 0)
    (0, 0, 255),    # 红色 (Class 1)
    (255, 0, 0),    # 蓝色 (Class 2)
    (255, 255, 0),  # 青色 (Class 3)
    (255, 0, 255),  # 洋红色 (Class 4)
    (0, 255, 255),  # 黄色 (Class 5)
    (128, 0, 0),    # 深蓝色 (Class 6)
    (0, 128, 0),    # 深绿色 (Class 7)
    (0, 0, 128),    # 深红色 (Class 8)
    (128, 128, 0),  # 深青色 (Class 9)
]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='在图片上绘制标注框')
    parser.add_argument('--image', '-i', required=True, help='输入图片路径')
    parser.add_argument('--infer_result0', '-r1', required=True, help='模型推理结果0')
    parser.add_argument('--infer_result1', '-r2', required=True, help='模型推理结果1')
    parser.add_argument('--output_path', '-o', help='完成后处理输出图片路径，不指定则直接显示')
    return parser.parse_args()

def read_yolo_seg_bin(detections_bin_path: str, protos_bin_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取YOLOv11-seg模型输出的两个bin文件，解析为NumPy数组
    
    参数:
        detections_bin_path: 第一个bin文件路径（存储8400×116个float32，对应模型输出0）
        protos_bin_path: 第二个bin文件路径（存储32×160×160个float32，对应模型输出1）
    
    返回:
        detections: 解析后的detections数组，形状(8400, 116)，dtype=np.float32
        protos: 解析后的protos数组，形状(32, 160, 160)，dtype=np.float32
    
    异常:
        ValueError: 若文件大小与预期不符（确保bin文件是模型输出的原始二进制数据）
    """
    # ---------------------- 读取detections bin（8400×116 float32） ----------------------
    # 每个float32占4字节，计算预期文件大小
    expected_detections_size = 8400 * 116 * 4
    try:
        with open(detections_bin_path, 'rb') as f:
            detections_data = f.read()
            # 验证文件大小（避免读取不完整或错误的bin文件）
            if len(detections_data) != expected_detections_size:
                raise ValueError(
                    f"detections bin文件大小异常！预期{expected_detections_size}字节，实际{len(detections_data)}字节"
                )
            # 从二进制数据解析为NumPy数组，并重塑为(8400, 116)
            detections = np.frombuffer(detections_data, dtype=np.float32).reshape(116, 8400)
    except Exception as e:
        raise RuntimeError(f"读取detections bin文件失败：{str(e)}") from e

    # ---------------------- 读取protos bin（32×160×160 float32） ----------------------
    expected_protos_size = 32 * 160 * 160 * 4
    try:
        with open(protos_bin_path, 'rb') as f:
            protos_data = f.read()
            if len(protos_data) != expected_protos_size:
                raise ValueError(
                    f"protos bin文件大小异常！预期{expected_protos_size}字节，实际{len(protos_data)}字节"
                )
            # 从二进制数据解析为NumPy数组，并重塑为(32, 160, 160)
            protos = np.frombuffer(protos_data, dtype=np.float32).reshape(32, 160, 160)
    except Exception as e:
        raise RuntimeError(f"读取protos bin文件失败：{str(e)}") from e

    return detections, protos


conf_thres : float = 0.25
iou_thres: float = 0.45

def get_pad_scale(img, target_size=(640, 640)):
    """
    改进的预处理：使用LetterBox缩放（保持比例+填充）
    
    返回:
        img_preprocessed: 处理后的图像
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
    
    # 缩放和填充
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=[114, 114, 114])
    return r, (left, top)

def crop_mask(masks, boxes):
    """与原代码一致：根据边界框裁剪掩码区域"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # (N,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # (1,1,W)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # (1,H,1)
    mask_region = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)
    return masks * mask_region

def processMask(protos, mask_coeffs, bboxs, img_shape: Tuple[int, int]) -> np.ndarray:
    """与原代码一致：生成并缩放掩码（修复了原代码的sigmoid激活缺失）"""
    c, mh, mw = protos.shape  # 掩码原型尺寸 (32, 160, 160)
    ih, iw = img_shape        # 模型输入尺寸 (如640, 640)

    masks = (mask_coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)

    # 计算边界框从「模型输入尺寸」到「掩码原型尺寸」的缩放比例
    width_ratio = mw / iw 
    height_ratio = mh / ih 

    # 缩放边界框到掩码尺寸
    downsampled_bbox = bboxs.copy()
    downsampled_bbox[:, [0, 2]] *= width_ratio  # x1, x2（宽度维度）
    downsampled_bbox[:, [1, 3]] *= height_ratio  # y1, y2（高度维度）

    # 裁剪掩码+缩放到模型输入尺寸
    masks = crop_mask(masks, downsampled_bbox)
    masks_resize = np.array([
        cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR) 
        for m in masks
    ])

    return masks_resize

def postprocess(detections: np.ndarray, protos: np.ndarray, orig_shape: Tuple[int, int], 
                img_shape: Tuple[int, int], scale: float, pad: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    与原代码一致的后处理，仅将「模型输出」替换为「bin读取的detections和protos」
    """
    # 处理detections（原代码中outputs[0]的解析逻辑）
    detection = detections.transpose()  # (8400, 116) → (116, 8400)？原代码逻辑，保持不变
    detection = np.squeeze(detection, axis=-1) if detection.ndim == 3 else detection  # 确保形状为(8400, 116)

    # 提取边界框、置信度、掩码系数
    bboxes = detection[:, :4]          # (N, 4)，xywh格式（模型输入尺寸）
    scores = detection[:, 4:84]        # (N, 80)，80个类别的置信度（COCO数据集）
    mask_coeffs = detection[:, 84:]    # (N, 32)，掩码系数（对应protos的32个通道）

    # 1. 过滤低置信度目标
    class_ids = np.argmax(scores, axis=1)  # 每个目标的类别ID
    max_scores = np.max(scores, axis=1)    # 每个目标的最高置信度
    keep = max_scores > conf_thres    # 筛选置信度合格的目标
    bboxes, max_scores, class_ids, mask_coeffs = bboxes[keep], max_scores[keep], class_ids[keep], mask_coeffs[keep]
    
    if len(bboxes) == 0:
        return np.array([]), np.array([])  # 无目标时返回空数组

    # 2. xywh → xyxy（模型输入尺寸）
    x1 = bboxes[:, 0] - bboxes[:, 2] / 2
    y1 = bboxes[:, 1] - bboxes[:, 3] / 2
    x2 = bboxes[:, 0] + bboxes[:, 2] / 2
    y2 = bboxes[:, 1] + bboxes[:, 3] / 2
    bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (N, 4)

    # 3. NMS过滤重叠框
    indices = cv2.dnn.NMSBoxes(
        bboxes_xyxy.tolist(), max_scores.tolist(),
        conf_thres, iou_thres
    )
    if len(indices) == 0:
        return np.array([]), np.array([])
    indices = indices.flatten()  # 展平索引
    bboxes_xyxy, max_scores, class_ids, mask_coeffs = bboxes_xyxy[indices], max_scores[indices], class_ids[indices], mask_coeffs[indices]

    # 4. 生成掩码（使用bin读取的protos）
    masks = processMask(protos, mask_coeffs, bboxes_xyxy, img_shape)

    # 5. 修正边界框到原始图像尺寸（减去填充、除以缩放比例）
    orig_h, orig_w = orig_shape
    bboxes_xyxy[:, 0] = np.clip((bboxes_xyxy[:, 0] - pad[0]) / scale, 0, orig_w)
    bboxes_xyxy[:, 1] = np.clip((bboxes_xyxy[:, 1] - pad[1]) / scale, 0, orig_h)
    bboxes_xyxy[:, 2] = np.clip((bboxes_xyxy[:, 2] - pad[0]) / scale, 0, orig_w)
    bboxes_xyxy[:, 3] = np.clip((bboxes_xyxy[:, 3] - pad[1]) / scale, 0, orig_h)

    # 6. 修正掩码到原始图像尺寸（裁剪填充区域+缩放）
    ih, iw = img_shape
    masks = masks[:, pad[1]: ih-pad[1], pad[0]: iw-pad[0]]  # 裁剪填充黑边
    masks_resize = np.array([
        cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) 
        for m in masks
    ])
    mask_binary = np.where(masks_resize > 0.5, 1, 0).astype(np.uint8)

    return bboxes_xyxy, mask_binary, class_ids, max_scores

def process_from_bin(img_path: str, detections_bin_path: str,
                        protos_bin_path: str, output_path: str) -> np.ndarray:
    """
    核心函数：从bin文件读取结果，处理并可视化
    
    参数:
        img_path: 原始图像路径（用于可视化）
        detections_bin_path: detections bin文件路径
        protos_bin_path: protos bin文件路径
    
    返回:
        detections: 修正到原始图像尺寸的边界框数组 (N, 4)，xyxy格式
    """
    # 1. 读取原始图像（用于可视化和获取原始尺寸）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    orig_h, orig_w = img.shape[:2]
    img_vis = img.copy()  # 复制一份用于绘制（避免修改原图）

    input_shape = (640, 640)  # 模型输入尺寸 (H, W)
    scale, pad = get_pad_scale(img, input_shape)

    # 3. 从bin文件读取模型输出结果
    detections, protos = read_yolo_seg_bin(detections_bin_path, protos_bin_path)

    # 4. 后处理（修正坐标、生成掩码）
    bboxes_xyxy, mask_binary, class_id, max_score= postprocess(
        detections=detections,
        protos=protos,
        orig_shape=(orig_h, orig_w),
        img_shape=input_shape,
        scale=scale,
        pad=pad
    )

    # 5. 可视化（绘制边界框和掩码轮廓）
    if len(bboxes_xyxy) > 0:
        overlap = img.copy() 
        for i in range(mask_binary.shape[0]):
            # 目标区域设为红色（BGR 顺序：(0, 0, 255)），可替换为其他颜色（如绿色 (0,255,0)）
            overlap[mask_binary[i] == 1] = COLORS[i % 10]

        # 5. 半透明叠加（alpha 控制 mask 透明度，0=完全透明，1=完全不透明）
        alpha = 0.5  # 推荐 0.3~0.7，平衡 mask 可见性和原图清晰度
        img_vis = cv2.addWeighted(
            src1=img_vis,        # 原图
            alpha=1 - alpha, # 原图权重（1 - mask 透明度）
            src2=overlap, # 彩色 mask
            beta=alpha,      # mask 权重（透明度）
            gamma=0          # 亮度补偿，默认 0
        )
        for i in range(bboxes_xyxy.shape[0]):
            # 绘制边界框（红色，线宽2）
            x1, y1, x2, y2 = map(int, bboxes_xyxy[i])
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), COLORS[i % 10], 2)

            class_name = CLASS_MAPPING.get(class_id[i], f"Class {class_id[i]}")

            # 绘制标签背景
            label = f"{class_name}: {max_score[i]:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            label_w, label_h = label_size
            cv2.rectangle(img_vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), COLORS[i % 10], -1)
            
            # 绘制标签文本
            cv2.putText(img_vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img_vis)
        print(f"已保存结果到: {output_path}")
    else:
        # 显示结果（BGR→RGB，因为matplotlib默认RGB）
        img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_vis_rgb)
        plt.axis('off')
        plt.show()

# 使用示例
if __name__ == "__main__":
    args = parse_arguments()
    process_from_bin(args.image, args.infer_result1, args.infer_result0, args.output_path)
