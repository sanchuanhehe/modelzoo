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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def run_evaluation(args):
    """执行COCO关键点评估"""
    # 获取所有txt结果文件
    txt_files = [f for f in os.listdir(args.result_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"错误: 在 {args.result_dir} 中未找到任何txt文件")
        return

    # 加载COCO标注
    coco_gt = COCO(args.gt_annotations)
    all_img_ids = coco_gt.getImgIds()

    # 创建结果列表
    detection_results = []

    # 处理每个txt结果
    total = len(txt_files)
    for idx, txt_file in enumerate(txt_files):
        # 提取图片名称
        img_name = os.path.splitext(txt_file)[0]

        # 查找图片ID
        img_ids = coco_gt.getImgIds(imgIds=[int(img_name)] if img_name.isdigit() else [])
        if not img_ids:
            # 尝试通过文件名查找
            img_infos = coco_gt.loadImgs(all_img_ids)
            matched = [info for info in img_infos if os.path.splitext(info['file_name'])[0] == img_name]
            if matched:
                img_id = matched[0]['id']
            else:
                print(f"警告: 未找到图片 {img_name} 对应的ID，跳过")
                continue
        else:
            img_id = img_ids[0]

        # 读取txt文件内容
        txt_path = os.path.join(args.result_dir, txt_file)
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # 解析每行内容
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 格式: category_id score x y width height (17*3 keypoints)
                if len(parts) != 1 + 1 + 4 + 51:  # 1+1+4+51=57
                    print(f"警告: {txt_file} 中格式错误，跳过此行: {line}")
                    continue

                category_id = int(parts[0])
                score = float(parts[1])
                bbox = [float(x) for x in parts[2:6]]  # x, y, width, height
                keypoints = [float(x) for x in parts[6:57]]  # 17*3=51个关键点

                detection_results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score,
                    "keypoints": keypoints
                })

        except Exception as e:
            print(f"警告: 处理文件 {txt_file} 失败: {e}，跳过")
            continue

        # 打印进度
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            progress = (idx + 1) / total * 100
            print(f"处理进度: {idx + 1}/{total} ({progress:.1f}%) - 文件: {txt_file}")

    # 保存临时JSON结果（用于COCO评估）
    parent_dir = os.path.dirname(args.result_dir)
    temp_json = os.path.join(parent_dir, "temp_pose_results.json")
    with open(temp_json, "w") as f:
        json.dump(detection_results, f)
    print(f"临时评估文件已保存至: {temp_json}")

    # 执行COCO评估
    if not detection_results:
        print("警告: 没有有效的检测结果，无法进行评估")
        return

    coco_dt = coco_gt.loadRes(temp_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    # 确保评估使用与推理相同的图片集
    coco_eval.params.imgIds = [res["image_id"] for res in detection_results]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据TXT检测结果进行COCO关键点评估')
    parser.add_argument('--result_dir', default='../out/result/txt',
                        help='存放检测结果TXT文件的目录')
    parser.add_argument('--gt_annotations',
                        default="../../../../../datasets/coco2017/annotations/person_keypoints_val2017.json",
                        help='COCO的ground truth标注文件路径（关键点）')

    args = parser.parse_args()

    # 验证输入
    if not os.path.isdir(args.result_dir):
        print(f"错误: 结果目录 {args.result_dir} 不存在")
        return
    if not os.path.exists(args.gt_annotations):
        print(f"错误: 标注文件 {args.gt_annotations} 不存在")
        return

    # 执行评估
    run_evaluation(args)


if __name__ == "__main__":
    main()
