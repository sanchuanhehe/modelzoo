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

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import glob
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path


def parse_yolo_output_with_mapping(detection_line):
    try:
        # 多种可能的格式解析
        if "|" in detection_line:
            parts = detection_line.split("|")
            if len(parts) >= 3:
                class_part = parts[0].strip()
                score_part = parts[1].strip()
                box_part = parts[2].strip()

                # 提取类别ID
                if "Class" in class_part:
                    class_id = int(class_part.replace("Class", "").strip())
                else:
                    class_id = int(class_part)
                # 提取置信度
                if "Score" in score_part:
                    score = float(score_part.replace("Score:", "").strip())
                else:
                    score = float(score_part)

                # 提取边界框
                if "Box" in box_part:
                    box_str = box_part.replace("Box:", "").strip().strip("[]")
                else:
                    box_str = box_part.strip().strip("[]")

                box_coords = [float(x.strip()) for x in box_str.split(",")]

                return class_id, score, box_coords
        else:
            # 可能是其他格式，尝试直接解析
            print(f"警告: 非常规格式: {detection_line}")

    except Exception as e:
        print(f"解析检测行时出错: {detection_line}, 错误: {e}")

    return None, None, None


def improved_process_all_results(txt_folder_path, gt_annotations_path):
    coco_gt = COCO(gt_annotations_path)

    result_files = glob.glob(os.path.join(txt_folder_path, "*.txt"))
    print(f"找到 {len(result_files)} 个结果文件")

    all_detections = []
    total_detections = 0

    for result_file in tqdm(result_files):
        try:
            filename = os.path.basename(result_file)
            image_id = int(Path(filename).stem)
            #print("image_id: " , image_id, coco_gt.getImgIds())
            if image_id not in coco_gt.getImgIds():
                continue

            with open(result_file, "r") as f:
                yolo_outputs = [line.strip() for line in f if line.strip()]

            if not yolo_outputs:
                continue

            # 转换检测结果
            image_detections = []
            for detection_line in yolo_outputs:
                class_id, score, box_coords = parse_yolo_output_with_mapping(
                    detection_line
                )

                if (
                    class_id is not None
                    and score is not None
                    and box_coords is not None
                    and len(box_coords) == 4
                ):
                    x1_orig, y1_orig, width, height = box_coords

                    coco_detection = {
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x1_orig, y1_orig, width, height],
                        "score": score,
                    }
                    image_detections.append(coco_detection)
                    total_detections += 1

            all_detections.extend(image_detections)

        except Exception as e:
            print(f"处理文件 {result_file} 时出错: {e}")

    print(f"总检测数量: {total_detections}")

    # 统计检测结果
    if all_detections:
        scores = [det["score"] for det in all_detections]
        print(f"置信度范围: {min(scores):.3f} - {max(scores):.3f}")
        print(f"平均置信度: {np.mean(scores):.3f}")

    return all_detections, coco_gt


def evaluate_and_save_results(
    all_detections, coco_gt, output_dir="out/evaluation_results"
):
    """评估并保存结果"""

    print("\n=== 评估检测结果 ===")

    if not all_detections:
        print("错误: 没有检测结果可评估")
        return None

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存检测结果到临时文件
    temp_det_file = os.path.join(output_dir, "all_detections.json")
    with open(temp_det_file, "w") as f:
        json.dump(all_detections, f)

    # 加载检测结果
    coco_dt = coco_gt.loadRes(temp_det_file)
    # 清理临时文件
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # 创建评估器
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # 运行评估
    print("运行评估...")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 accuracy.")
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        default="./datasets/coco/instances_val2017.json",
    )
    parser.add_argument("--output", default="./out/result/txt", type=str)
    opt = parser.parse_args()
    txt_folder_path = opt.output  # 包含所有结果文件的文件夹
    gt_annotations_path = opt.ground_truth_json  # COCO真实标注文件

    print("\n" + "=" * 60)
    all_detections, coco_gt = improved_process_all_results(
        txt_folder_path,
        gt_annotations_path,
    )

    if all_detections:
        print("\n" + "=" * 60)
        coco_eval = evaluate_and_save_results(all_detections, coco_gt)
    else:
        print("错误: 没有生成任何检测结果")
