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
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 定义类别ID到名称的映射
YOLO_CLASS_NUM = 80
CLASSES_TXT = "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush"
CLASS_MAPPING = dict(zip(range(YOLO_CLASS_NUM), CLASSES_TXT.split('|')))
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
    parser.add_argument('--annotation', '-a', required=True, help='标注文件路径')
    parser.add_argument('--output', '-o', help='输出图片路径，不指定则直接显示')
    parser.add_argument('--class-map', '-cm', help='自定义类别映射文件路径')
    return parser.parse_args()

def parse_annotation_file(file_path):
    """解析标注文件"""
    annotations = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Class"):
                    continue
                    
                # 分割各部分
                parts = line.split('|')
                if len(parts) != 3:
                    continue
                    
                # 解析类别
                class_part = parts[0].strip()
                class_id = int(class_part.split()[1])
                
                # 解析分数
                score_part = parts[1].strip()
                score = float(score_part.split(':')[1])
                
                # 解析边界框
                box_part = parts[2].strip()
                box_str = box_part.split(':')[1].strip(' []')
                box = [float(x) for x in box_str.split(',')]
                
                annotations.append({
                    'class_id': class_id,
                    'score': score,
                    'box': box  # [x1, y1, x2, y2] 格式，归一化坐标
                })
                
    except Exception as e:
        print(f"解析标注文件出错: {e}")
        return []
        
    return annotations

def load_custom_class_mapping(file_path):
    """加载自定义类别映射文件"""
    mapping = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                    
                parts = line.split('=')
                if len(parts) != 2:
                    continue
                    
                try:
                    class_id = int(parts[0].strip())
                    class_name = parts[1].strip()
                    mapping[class_id] = class_name
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"加载自定义类别映射出错: {e}")
        
    return mapping

def visualize_detections(image_path, annotations, class_mapping=None, output_path=None):
    """在原图上绘制检测结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 使用默认或自定义类别映射
    if class_mapping is None:
        class_mapping = CLASS_MAPPING
    
    # 绘制边界框和标签
    for ann in annotations:
        box = ann['box']
        class_id = ann['class_id']
        score = ann['score']
        
        # 确保类别ID有对应的颜色和名称
        color_idx = class_id % len(COLORS)
        color = COLORS[color_idx]
        
        class_name = class_mapping.get(class_id, f"Class {class_id}")
        
        scale = min(640 / width, 640 / height)
        new_w, new_h = int(width * scale), int(height * scale)
        
        # 计算填充量（上下左右居中填充）
        # pad_w = (640 - new_w) // 2
        # pad_h = (640 - new_h) // 2
        dw, dh = round((640 - new_w) / 2 - 0.1), round((640 - new_h) / 2 - 0.1)

        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # 修正填充偏移 + 缩放到原始尺寸
        x1 = (x1 - dw) / scale
        y1 = (y1 - dh) / scale
        x2 = (x2 - dw) / scale
        y2 = (y2 - dh) / scale
        
        # 限制坐标在原图范围内
        x1 = int(np.clip(x1, 0, width))
        y1 = int(np.clip(y1, 0, height))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))
        
        print(f"{x1} {y1} {x2} {y2}")
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_w, label_h = label_size
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示或保存结果
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        print(f"已保存结果到: {output_path}")
    else:
        # 使用matplotlib显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    return image

def main():
    args = parse_arguments()
    
    # 加载自定义类别映射（如果有）
    if args.class_map:
        class_mapping = load_custom_class_mapping(args.class_map)
        print(f"已加载自定义类别映射，包含 {len(class_mapping)} 个类别")
    else:
        class_mapping = None
    
    # 解析标注文件
    annotations = parse_annotation_file(args.annotation)
    print(f"从 {args.annotation} 解析出 {len(annotations)} 个标注")
    
    # 可视化结果
    visualize_detections(args.image, annotations, class_mapping, args.output)

if __name__ == "__main__":
    main()