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
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def run_evaluation(args):
    """读取单张图片的JSON结果并执行COCO评估"""
    # 加载COCO标注
    coco_gt = COCO(args.gt_annotations)
    all_img_ids = coco_gt.getImgIds()

    # 创建结果列表
    detection_results = []

    # 获取所有JSON结果文件
    json_files = [f for f in os.listdir(args.result_dir) if f.endswith('.json')]
    if not json_files:
        print(f"错误: 在 {args.result_dir} 中未找到任何JSON文件")
        return

    # 处理每个JSON结果
    total = len(json_files)
    for idx, json_file in enumerate(tqdm(json_files, desc="加载结果文件")):
        # 提取图片名称
        img_name = os.path.splitext(json_file)[0]

        # 查找图片ID（确保与标注文件中的ID匹配）
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

        # 读取JSON文件内容
        json_path = os.path.join(args.result_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)

            # 确保结果中的image_id正确
            for res in results:
                res["image_id"] = img_id
                detection_results.append(res)

        except Exception as e:
            print(f"警告: 处理文件 {json_file} 失败: {e}，跳过")
            continue

    # 保存合并的JSON结果（用于COCO评估）
    temp_json = os.path.join(args.result_dir, "../merged_coco_results.json")
    with open(temp_json, "w") as f:
        json.dump(detection_results, f)
    print(f"合并的评估文件已保存至: {temp_json}")

    # 执行COCO评估
    if not detection_results:
        print("警告: 没有有效的检测结果，无法进行评估")
        return

    coco_dt = coco_gt.loadRes(temp_json)

    # -------------------------- 1. 评估框检测（bbox）--------------------------
    print("=" * 50)
    print("框检测（bbox）评估结果：")
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, iouType='bbox')
    # 确保评估使用与推理相同的图片集
    coco_eval_bbox.params.imgIds = list(set(res["image_id"] for res in detection_results))
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    # -------------------------- 2. 评估分割（segm）--------------------------
    print("\n" + "=" * 50)
    print("分割（segm）评估结果：")
    coco_eval_segm = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval_segm.params.imgIds = list(set(res["image_id"] for res in detection_results))
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='根据单张图片的JSON结果进行COCO评估')
    parser.add_argument('--result_dir', default='../out/result/json',
                        help='存放单张图片JSON结果的目录')
    parser.add_argument('--gt_annotations',
                        default="../../../../../datasets/coco2017/annotations/instances_val2017.json",
                        help='COCO的ground truth标注文件路径')

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
