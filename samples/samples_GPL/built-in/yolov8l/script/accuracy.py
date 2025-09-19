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

# YOLO80到COCO90类别的映射
yolo80_to_coco90 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

def parse_yolo_bin_files(bin_dir, img_dir, output_file, 
                        nms_threshold=0.6, conf_threshold=0.001, 
                        target_size=(640, 640)):
    """
    解析bin文件，使用torchvision的batched_nms按类别执行NMS，保存结果
    """
    image_results = {}
    bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
    print(f"找到 {len(bin_files)} 个bin文件")
    
    for bin_path in bin_files:
        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        file_name = base_name.replace("_result", "") if "_result" in base_name else base_name
        img_path = os.path.join(img_dir, f"{file_name}.jpg")
        
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 不存在，跳过")
            continue
        
        # 获取图片宽高
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"警告: 无法打开图片 {img_path}，错误: {e}，跳过")
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
                    
                    # 过滤低置信度框
                    if confidence > conf_threshold:
                        # 坐标转换（计算x_max, y_max用于NMS）
                        scale = min(target_size[0] / img_width, target_size[1] / img_height)
                        new_w, new_h = int(img_width * scale), int(img_height * scale)
                        pad_w, pad_h = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2

                        x_center = (x_center_rel - pad_w) / scale
                        y_center = (y_center_rel - pad_h) / scale
                        width = w_rel / scale
                        height = h_rel / scale

                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2  # 直接计算x_max（NMS需要）
                        y_max = y_center + height / 2  # 直接计算y_max（NMS需要）

                        # 边界裁剪
                        x_min = np.clip(x_min, 0, img_width)
                        y_min = np.clip(y_min, 0, img_height)
                        x_max = np.clip(x_max, 0, img_width)
                        y_max = np.clip(y_max, 0, img_height)

                        # 转换为COCO类别ID
                        category_id = yolo80_to_coco90[class_id]
                        
                        # 存储NMS所需格式（xyxy）和额外信息
                        all_bboxes.append([x_min, y_min, x_max, y_max])
                        all_scores.append(confidence)
                        all_categories.append(category_id)
                        all_infos.append({
                            "image_id": int(file_name),
                            "category_id": category_id,
                            "score": confidence,
                            "coco_bbox": [x_min, y_min, x_max - x_min, y_max - y_min]  # 预存COCO格式bbox
                        })
            
            # 执行按类别NMS（使用torchvision的batched_nms）
            if all_bboxes:
                # 转换为PyTorch张量（batched_nms要求的格式）
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
                
                # 保留NMS后的结果
                for idx in keep_indices:
                    info = all_infos[idx]
                    image_results.setdefault(int(file_name), []).append({
                        "image_id": info["image_id"],
                        "category_id": info["category_id"],
                        "bbox": [float(v) for v in info["coco_bbox"]],
                        "score": float(info["score"])
                    })
        
        except Exception as e:
            print(f"处理文件 {bin_path} 时出错: {e}")
            continue
    
    # 保存最终结果
    final_results = []
    for img_id in image_results:
        final_results.extend(image_results[img_id])
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f)
    
    print(f"处理完成，使用torchvision NMS过滤后保留 {len(final_results)} 个框，已保存到 {output_file}")
    return output_file

def evaluate_coco(gt_annotations, dt_results):
    """使用COCO API评估检测结果"""
    # 加载验证集标注文件
    coco_gt = COCO(gt_annotations)
    
    # 创建COCOeval对象
    coco_dt = coco_gt.loadRes(dt_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # 'bbox'表示目标检测
    
    # 执行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解析YOLO输出的bin文件并进行COCO评估')
    
    # 必选参数
    parser.add_argument('--bin_dir', required=True, 
                      help='存放YOLO输出bin文件的目录')
    parser.add_argument('--img_dir', required=True, 
                      help='存放原始图片的目录')
    parser.add_argument('--output_json', required=True, 
                      help='输出结果JSON文件路径')
    parser.add_argument('--gt_annotations', required=True, 
                      help='COCO格式的 ground truth 标注文件路径')
    
    # 可选参数
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
    
    # 验证标注文件是否存在
    if not os.path.isfile(args.gt_annotations):
        print(f"错误: 标注文件 {args.gt_annotations} 不存在")
        exit(1)
    
    # 解析bin文件并生成结果
    result_file = parse_yolo_bin_files(
        bin_dir=args.bin_dir,
        img_dir=args.img_dir,
        output_file=args.output_json,
        nms_threshold=args.nms_threshold,
        conf_threshold=args.conf_threshold,
        target_size=tuple(args.target_size)
    )
    
    # 执行COCO评估
    evaluate_coco(args.gt_annotations, result_file)

if __name__ == "__main__":
    main()