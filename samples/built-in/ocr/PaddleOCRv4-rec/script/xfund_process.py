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
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os
import math
import shutil
import argparse

class XFund2PaddleOCRRec:
    def __init__(self, 
                 xfund_anno_path: str,  # XFund标注文件路径（如 train.json）
                 xfund_img_dir: str,    # XFund图片存放目录
                 output_dir: str,       # 裁剪后图片输出目录
                 rec_height: int = 32,  # PaddleOCR-REC输入高度
                 rec_width: int = 100): # PaddleOCR-REC输入宽度
        self.anno_path = xfund_anno_path
        self.img_dir = xfund_img_dir
        self.output_dir = output_dir
        self.rec_h = rec_height
        self.rec_w = rec_width

        # 解析XFund标注
        self.annotations = self._parse_xfund_anno()

    def _parse_xfund_anno(self) -> list:
        """解析XFund JSON标注，提取图片名和检测框（归一化坐标转像素坐标）"""
        """将XFUND格式转换为PaddleOCR格式"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # 验证集
        json_path = self.anno_path
        image_dir = Path(self.img_dir)
        print("path: ", json_path, "  " , image_dir, " " , self.output_dir)
        if not Path(json_path).exists():
            print(f"警告：json_path不存在 {json_path}")
            return
                
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 创建标注文件
        label_file = Path(self.output_dir) / 'zh_val_labels.txt'
            
        with open(label_file, 'w', encoding='utf-8') as f_label:
            # 每个文档
            for item in data['documents']:
                img_file = item['img']['fname']
                img_path = image_dir / img_file
                img_name = img_file     
                if not img_path.exists():
                    print(f"警告：图片不存在 {img_path}")
                    continue
                    
                # 读取图片获取实际尺寸
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                
                
                # 处理每个标注
                idx = 0
                for ann in item['document']:
                    
                    if 'box' in ann:
                        # 获取边界框
                        box = ann['box']
                        # XFUND格式: [x1, y1, x2, y2, x3, y3, x4, y4]
                        # 转换为PaddleOCR格式
                        points = []
                        # 根据box的长度判断格式
                        if len(box) == 4:
                            # 矩形框 [x1, y1, x2, y2]
                            x1, y1, x2, y2 = box
                            # 转换为四点
                            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        elif len(box) == 8:
                            # 多边形框 [x1, y1, x2, y2, x3, y3, x4, y4]
                            points = [[box[i], box[i+1]] for i in range(0, 8, 2)]
                        else:
                            print(f"未知的box格式: {box}")
                            continue
                            
                        # 转换为字符串
                        points_str = [f"{p[0]:.1f},{p[1]:.1f}" for p in points]
                        text = ann.get('text', '')

                        # 裁剪并水平化文本
                        cropped_img = self._crop_and_rotate(img, np.array(points))
                        if cropped_img is None or cropped_img.size == 0:
                            print(f"警告：裁剪失败 {img_name} - {text}，跳过")
                            continue

                         # 调整为REC输入尺寸
                        rec_img = cropped_img
                        if rec_img is None:
                            print(f"警告：调整尺寸失败 {img_name} - {text}，跳过")
                            continue
                        
                        # 保存图片（文件名格式：img_name_索引_文本内容.png，避免冲突）
                        # 过滤非法字符（如路径分隔符）
                        safe_text = text  # 限制文本长度
                        output_subdir = Path(self.output_dir) / "img"  # 按图片名创建子目录
                        output_subdir.mkdir(exist_ok=True)
                        output_path = output_subdir / f"{img_name.split('.')[0]}_{idx}.png"
                        print("output_path: " , output_path)
                        f_label.write("img/" + img_name.split('.')[0] + "_"+ str(idx) + ".png" + "   " + safe_text + '\n')
                        # 保存图片（PaddleOCR支持BGR或RGB，此处保持BGR格式）
                        cv2.imwrite(str(output_path), rec_img)
                        idx = idx + 1
                        # 打印进度（每100个输出一次）
                        if (idx + 1) % 100 == 0:
                            print(f"已处理 {idx + 1}个文本框")


    def _split_to_rec_blocks(self, img: np.ndarray) -> list:
        """将图片按320宽度切分为多个32×320的块（不截断、不缩放）"""
        if img is None or img.size == 0:
            return []
        
        h, w = img.shape[:2]
        # 步骤1：按高度缩放至32（保持长宽比）
        scale = self.rec_h / h
        new_w = int(w * scale)
        new_h = self.rec_h
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 步骤2：计算需要切分的块数
        num_blocks = math.ceil(new_w / self.rec_w)
        blocks = []
        
        for i in range(num_blocks):
            # 每个块的起始/结束横坐标
            start_x = i * self.rec_w
            end_x = min((i + 1) * self.rec_w, new_w)
            
            # 截取当前块
            block = resized_img[:, start_x:end_x]
            block_h, block_w = block.shape[:2]
            
            # 补白至32×320（不足部分补白色）
            rec_block = np.ones((self.rec_h, self.rec_w, 3), dtype=np.uint8) * 255
            rec_block[:, :block_w] = block  # 左对齐填充
            
            blocks.append(rec_block)
        
        return blocks

    def _crop_and_rotate(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        """根据检测框裁剪图片，并水平化倾斜文本（最小外接矩形）"""
        # 计算检测框的最小外接矩形（处理倾斜文本）
        rect = cv2.minAreaRect(box)
        (center_x, center_y), (w, h), angle = rect
        
        # 处理角度：确保宽>高（文本行水平）
        if w < h:
            w, h = h, w
            angle += 90  # 旋转90度，使文本水平
       # print("w: xxx: " , w)
        # 旋转图片，使文本行水平
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        
        # 裁剪旋转后的文本区域（基于最小外接矩形的边界框）
        box_rotated = cv2.boxPoints(rect).astype(np.int32)
        x_min = max(0, int(np.min(box_rotated[:, 0])))
        x_max = min(img.shape[1], int(np.max(box_rotated[:, 0])))
        y_min = max(0, int(np.min(box_rotated[:, 1])))
        y_max = min(img.shape[0], int(np.max(box_rotated[:, 1])))
        
        cropped_img = rotated_img[y_min:y_max, x_min:x_max]
        return cropped_img

    def _resize_to_rec_input(self, img: np.ndarray) -> np.ndarray:
        """将裁剪后的图片调整为PaddleOCR-REC输入尺寸（保持长宽比，空白处补白）"""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return None
        
        # 计算缩放比例（优先满足高度，宽度自适应；若宽度超上限则按宽度缩放）
        scale = self.rec_h / h
        new_w = int(w * scale)
        new_h = self.rec_h
        
        if new_w > self.rec_w:
            scale = self.rec_w / w
            new_w = self.rec_w
            new_h = int(h * scale)
        
        # 缩放图片
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 创建空白画布（白色背景，符合PaddleOCR默认输入）
        rec_img = np.ones((self.rec_h, self.rec_w, 3), dtype=np.uint8) * 255
        
        # 将缩放后的图片居中放置
        y_offset = (self.rec_h - new_h) // 2
        x_offset = (self.rec_w - new_w) // 2
        rec_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_img
        
        return rec_img

    def convert_xfund_to_ppocr(self, label_path, xfund_dir, output_dir):
        """将XFUND格式转换为PaddleOCR格式"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 验证集
        json_path = label_path
        image_dir = Path(xfund_dir)
        print("path: ", json_path, "  " , image_dir)
        if not Path(json_path).exists():
            print(f"警告：json_path不存在 {json_path}")
            return
                
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保目标目录存在
        target_base_dir = output_dir /"images/"
        os.makedirs(target_base_dir, exist_ok=True)  
        # 创建标注文件
        label_file = output_dir / 'zh_val_labels.txt'
        json_gt = []    
        with open(label_file, 'w', encoding='utf-8') as f_label:
            # 每个文档
            for item in data['documents']:
                img_file = item['img']['fname']
                img_path = image_dir / img_file
                    
                if not img_path.exists():
                    print(f"警告：图片不存在 {img_path}")
                    continue
                    
                # 读取图片获取实际尺寸
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                json_tmp = {}
                json_tmp["img"] = str(img_path) 
                    
                # 写入图片路径
                f_label.write(f"images/{img_file}\t")
                # 保存图像
                cv2.imwrite(f"{target_base_dir}/{img_file}", img)    
                annotations = []
                gt_info_list = []
                # 处理每个标注
                for ann in item['document']:
                    if 'box' in ann:
                        # 获取边界框
                        box = ann['box']
                        # XFUND格式: [x1, y1, x2, y2, x3, y3, x4, y4]
                        # 转换为PaddleOCR格式
                        points = []
                        # 根据box的长度判断格式
                        if len(box) == 4:
                            # 矩形框 [x1, y1, x2, y2]
                            x1, y1, x2, y2 = box
                            # 转换为四点
                            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        elif len(box) == 8:
                            # 多边形框 [x1, y1, x2, y2, x3, y3, x4, y4]
                            points = [[box[i], box[i+1]] for i in range(0, 8, 2)]
                        else:
                            print(f"未知的box格式: {box}")
                            continue
                            
                        # 转换为字符串
                        points_str = [f"{p[0]:.1f},{p[1]:.1f}" for p in points]
                        text = ann.get('text', '')
                        content = {
                            "transcription": text,
                            "points": points
                        }
                        gt_point = {"points": points, "text": text, "ignore":False}
                        # PaddleOCR格式: 坐标,文本
                        gt_info_list.append(gt_point)
                        annotations.append(content)
                json_tmp["gt"] = gt_info_list
                json_gt.append(json_tmp) 
                # 写入所有标注
                f_label.write(json.dumps(annotations, ensure_ascii=False) + '\n')
        pred_json_file = "./end_gt_xfund.json"
        #print("json_result: ", json_result)
        with open(pred_json_file, 'w') as f:
            json.dump(json_gt, f)
        print(f"已转换 det 数据")

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model inference')
    parser.add_argument('--data_path', type=str, default="./datasets/xfund/zh.val/")
    parser.add_argument('--label_path', type=str, default="./datasets/xfund/zh.val.json")
    parser.add_argument('--rec_out_path', type=str, default='./datasets/paddleocr_rec_input')
    parser.add_argument('--det_out_path', default="./datasets/xfund_ppocr", type=str)
    parser.add_argument('--rec_height', type=int, default=48)
    parser.add_argument('--rec_width', type=int, default=320) 
    parser.add_argument('--det_height', type=int, default=960)
    parser.add_argument('--det_width', type=int, default=960) 

    opt = parser.parse_args()
    converter = XFund2PaddleOCRRec(
        xfund_anno_path=opt.label_path,  # XFund标注文件路径
        xfund_img_dir=opt.data_path,        # XFund图片目录
        output_dir=opt.rec_out_path,    # 输出目录
        rec_height=opt.rec_height,                                # PaddleOCR-REC高度
        rec_width=opt.rec_width                                 # PaddleOCR-REC宽度
    )
    #det数据
    converter.convert_xfund_to_ppocr(opt.label_path, opt.data_path, opt.det_out_path)