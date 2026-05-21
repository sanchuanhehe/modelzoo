# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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
import struct
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision.ops import batched_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_yolo_labels_to_coco(img_dir, labels_dir, output_gt_json, name_to_id):
    """
    遍历labels_dir下的所有txt文件，将其YOLO相对坐标转换为COCO绝对坐标，
    并保存为COCO evaluation所需的JSON格式。
    """
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "fire"}]
    }

    ann_id = 1
    # 查找所有图片
    img_files = glob.glob(os.path.join(img_dir, "*.*"))
    img_exts = {'.jpg', '.png', '.jpeg'}
    img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in img_exts]

    print(f"在 {img_dir} 找到 {len(img_files)} 张图片用于构建 Ground Truth。")

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img_id = name_to_id[base_name]

        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"无法打开GT图片 {img_path}: {e}")
            continue

        dataset["images"].append({
            "id": img_id,
            "width": img_width,
            "height": img_height,
            "file_name": os.path.basename(img_path)
        })

        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # 火焰检测仅有火焰类，强制统一归一化为类 0
                        if cls_id != 0:
                            cls_id = 0

                        x_center_rel = float(parts[1])
                        y_center_rel = float(parts[2])
                        w_rel = float(parts[3])
                        h_rel = float(parts[4])

                        w = w_rel * img_width
                        h = h_rel * img_height
                        x_min = (x_center_rel * img_width) - (w / 2)
                        y_min = (y_center_rel * img_height) - (h / 2)

                        dataset["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                            "area": round(w * h, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1

    with open(output_gt_json, 'w') as f:
        json.dump(dataset, f)
    print(f"GT 转换完成，Ground Truth JSON 已包含 {len(dataset['images'])} 张图片, {len(dataset['annotations'])} 个真实标注框，存至: {output_gt_json}")
    return output_gt_json


def parse_yolo_bin_files(bin_dir, img_dir, output_file, name_to_id, nms_threshold=0.6, conf_threshold=0.001, target_size=(640, 640)):
    """
    解析bin文件，使用torchvision的batched_nms按类别执行NMS，保存结果
    """
    image_results = {}
    bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
    print(f"找到 {len(bin_files)} 个 bin 预测文件...")

    for bin_path in bin_files:
        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        file_name = base_name.replace("_result", "") if "_result" in base_name else base_name

        # 通过在映射中取id确保文件名与预测能稳定对应（即使非数字文件也能支持）
        if file_name not in name_to_id:
            continue
        img_id = name_to_id[file_name]

        # 寻找对应的图片
        # 兼容 .jpg/.png
        img_path = os.path.join(img_dir, f"{file_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{file_name}.png")

        if not os.path.exists(img_path):
            print(f"警告: 预测结果找不着对应的原图 {file_name}，跳过该预测")
            continue

        # 获取图片宽高
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"警告: 无法打开图片 {img_path}，错误: {e}，跳过该预测")
            continue

        # 解析bin文件并收集所有框信息
        try:
            with open(bin_path, 'rb') as f:
                data = f.read()
                total_floats = len(data) // 4
                rows = total_floats // 84

                if rows != 8400:
                    print(f"警告: {bin_path} 包含 {rows} 行，预期8400行，跳过")
                    continue

                # 存储当前图像的所有框（筛选后）
                all_bboxes = []  # 存储[x_min, y_min, x_max, y_max]格式
                all_scores = []
                all_categories = []
                all_infos = []

                for row in range(rows):
                    start_idx = row * 84 * 4
                    end_idx = start_idx + 84 * 4
                    row_data = data[start_idx:end_idx]

                    floats = struct.unpack('84f', row_data)

                    x_center_rel, y_center_rel, w_rel, h_rel = floats[:4]
                    class_scores = floats[4:]
                    class_id = class_scores.index(max(class_scores))
                    confidence = max(class_scores)

                    # 🔥 火焰检测这里仅计算 class_id == 0 这一类框
                    if class_id != 0:
                        continue

                    # 过滤低置信度框
                    if confidence > conf_threshold:
                        # 坐标反变换映射，结合推理图的 padding 和 scale
                        scale = min(target_size[0] / img_width, target_size[1] / img_height)
                        new_w, new_h = int(img_width * scale), int(img_height * scale)
                        pad_w, pad_h = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2

                        x_center = (x_center_rel - pad_w) / scale
                        y_center = (y_center_rel - pad_h) / scale
                        width = w_rel / scale
                        height = h_rel / scale

                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        # 边界裁剪防溢出
                        x_min = np.clip(x_min, 0, img_width)
                        y_min = np.clip(y_min, 0, img_height)
                        x_max = np.clip(x_max, 0, img_width)
                        y_max = np.clip(y_max, 0, img_height)

                        category_id = 0 # 强制只包含单类类别 0 的火焰

                        # 存储NMS所需格式（xyxy）和额外信息
                        all_bboxes.append([x_min, y_min, x_max, y_max])
                        all_scores.append(confidence)
                        all_categories.append(category_id)
                        all_infos.append({
                            "image_id": img_id,
                            "category_id": category_id,
                            "score": confidence,
                            "coco_bbox": [x_min, y_min, x_max - x_min, y_max - y_min]  # 预存COCO格式bbox
                        })

            # 使用 torchvision 执行 NMS
            if all_bboxes:
                # 转换为PyTorch张量
                bboxes_tensor = torch.tensor(all_bboxes, dtype=torch.float32)
                scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
                categories_tensor = torch.tensor(all_categories, dtype=torch.int64)

                # 按类别执行NMS
                keep_indices = batched_nms(
                    boxes=bboxes_tensor,
                    scores=scores_tensor,
                    idxs=categories_tensor,  # 按类别ID分组
                    iou_threshold=nms_threshold
                ).numpy()  # 转换为numpy索引

                # 记录最终幸存框信息
                for idx in keep_indices:
                    info = all_infos[idx]
                    image_results.setdefault(img_id, []).append({
                        "image_id": info["image_id"],
                        "category_id": info["category_id"],
                        "bbox": [float(v) for v in info["coco_bbox"]],
                        "score": float(info["score"])
                    })

        except Exception as e:
            print(f"处理预测文件 {bin_path} 时出错: {e}")
            continue

    # 汇总写入输出
    final_results = []
    for img_id in image_results:
        final_results.extend(image_results[img_id])

    with open(output_file, 'w') as f:
        json.dump(final_results, f)

    print(f"预测数据处理完成，经过 NMS 过滤后留存 {len(final_results)} 个预测框，存至: {output_file}")
    return output_file


def evaluate_coco(gt_annotations, dt_results):
    """使用COCO API评估单类火焰检测结果"""
    coco_gt = COCO(gt_annotations)
    if not dt_results or not os.path.exists(dt_results) or os.path.getsize(dt_results) <= 2:
        print("未生成任何有效预测框，无法进行指标评估！")
        return

    coco_dt = coco_gt.loadRes(dt_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():
    parser = argparse.ArgumentParser(description='解析火焰检测YOLO的bin预测结果，读取YOLO格式GT，一键完成COCO基准测试')

    parser.add_argument('--bin_dir', required=True, help='存放YOLO输出bin文件的预测结果目录')
    parser.add_argument('--img_dir', required=True, help='存放原始测试图片的目录 (例如 fasdd/test/images)')
    parser.add_argument('--labels_dir', required=True, help='存放YOLO格式 txt真值标注的目录 (例如 fasdd/test/labels)')

    parser.add_argument('--output_json', default='dt_result.json', help='预测结果临时COCO表示JSON的保存路径')
    parser.add_argument('--gt_json_out', default='gt_annotations.json', help='真值结果转化为了COCO格式后的临时JSON存储路径')

    parser.add_argument('--nms_threshold', type=float, default=0.6, help='NMS的IOU阈值，默认0.6')
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='置信度过滤阈值，默认0.001')
    parser.add_argument('--target_size', type=int, nargs=2, default=[384, 640], help='模型输入长边宽度和高度，例如"640 640"')

    args = parser.parse_args()

    for dir_path in [args.bin_dir, args.img_dir, args.labels_dir]:
        if not os.path.isdir(dir_path):
            print(f"严重错误: 检查到目录 {dir_path} 不存在，停止运行。")
            exit(1)

    # 为适应非数字文件名的映射，对所有图片统一按字母排序生成固定的正整数 Image ID 索引
    img_files = glob.glob(os.path.join(args.img_dir, "*.*"))
    img_exts = {'.jpg', '.png', '.jpeg'}
    img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in img_exts]
    img_files.sort()

    name_to_id = {}
    for idx, f in enumerate(img_files):
        base_name = os.path.splitext(os.path.basename(f))[0]
        name_to_id[base_name] = idx + 1 # COCO需要大于0的整数id

    print("------ 第一步：解析 Ground Truth (YOLO转COCO) ------")
    gt_file = parse_yolo_labels_to_coco(args.img_dir, args.labels_dir, args.gt_json_out, name_to_id)

    print("\n------ 第二步：解析 Predictions (bin推盘反解与NMS) ------")
    dt_file = parse_yolo_bin_files(
        bin_dir=args.bin_dir,
        img_dir=args.img_dir,
        output_file=args.output_json,
        name_to_id=name_to_id,
        nms_threshold=args.nms_threshold,
        conf_threshold=args.conf_threshold,
        target_size=tuple(args.target_size)
    )

    print("\n================ 第三步：正式执行 COCO AP/AR 精度评估 ================\n")
    evaluate_coco(gt_file, dt_file)

if __name__ == "__main__":
    main()
