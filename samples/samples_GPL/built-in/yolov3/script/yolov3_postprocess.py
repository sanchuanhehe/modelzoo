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

import yaml
import json
import argparse
import re
import os

import torch
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torchvision
import pandas as pd

def postprocess(opt, cfg):
    reference_list = os.listdir(opt.output)
    input_list = os.listdir()
    print("postprocess: ")
        # the output shapes
    shapes = [[1, 255, 80, 80], [1, 255, 40, 40], [1, 255, 20, 20]]
    with open("./data/shapes.json", "r") as f:
        loaded_data = json.load(f)
        shapes_data = {k: tuple(v) for k, v in loaded_data.items()}
    pred_results = []
    
    #with open('./data/file_list.txt', 'r') as f:
       # lines = f.readlines()
    # 读取 JSON 文件
    with open('./data/file_list.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用 pandas 处理
    df = pd.DataFrame(data['fileList'], columns=['file_path'])
    file_array = df['file_path'].tolist()
    print("conf: ", cfg["conf_thres"] , "  iou:  ", cfg["iou_thres"])  
    for i in tqdm(range(len(reference_list) // 3)):
        line = file_array[i].strip().split()
        print('img_path: ', line[0])
        out_path = Path(line[0]).stem
        out = []
        for output_num in range(3):
            file_path = out_path
            out_filepath = f"{opt.output}/{file_path}_{2-output_num}.bin"
            print("out_filepath ", out_filepath)
            inference_result = torch.tensor(np.fromfile(out_filepath, dtype=np.float32))
            inference_result = inference_result.reshape(shapes[output_num])# x(bs, 255, H, W) 
            bs, _, ny, nx = inference_result.shape  # x(bs, 255, H, W) → x(bs, anchors, H, W, box_attrs)
            inference_result = inference_result.reshape(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            print("inference_result ", inference_result.shape)
            anchors = torch.tensor(cfg['anchors'])
            stride = torch.tensor(cfg['stride'])
            cls_num = cfg['class_num']
            decord_infer_result(inference_result, anchors[output_num], stride[output_num], cls_num, out)
        
        all_decorded_box = torch.cat(out, 1)
        box_after_nms = do_class_nms(all_decorded_box, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"], multi_label=False)
        #按照预处理的resize和padding，还原位置框   
        for _, infer_box in enumerate(box_after_nms):
            try:
                infer_box[:, :4] = scale_boxes((640, 640), infer_box[:, :4], shapes_data[out_path]).round()
                 
            except:
                infer_box = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # append to COCO-JSON dictionary
            path = out_path # Path(path_list[i][idx])
            image_id = int(path) if path.isnumeric() else path
            save_coco_json(infer_box, pred_results, image_id, coco80_to_coco91_class())
            # print("     pred_results:     ", pred_results)
    print("json===============: ")
    pred_json_file = "end_predictions.json"
    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)
    evaluate(opt.ground_truth_json, pred_json_file)

def save_coco_json(predn, pred_dict, image_id, class_map):
  
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
 
    for p, b in zip(predn.tolist(), box.tolist()):
        pred_dict.append({'image_id': image_id,
                          'category_id': class_map[int(p[5])],
                          'bbox': [round(x, 3) for x in b],
                          'score': round(p[4], 5)})

#img0_shape = > hw
def scale_boxes(img1_shape, scale_boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    scale_boxes[:, [0, 2]] -= pad[0]  # x padding
    scale_boxes[:, [1, 3]] -= pad[1]  # y padding
    scale_boxes[:, :4] /= gain
    clip_boxes(scale_boxes, img0_shape)
    return scale_boxes

def clip_boxes(scale_boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(scale_boxes, torch.Tensor):  # faster individually
        scale_boxes[:, 0].clamp_(0, shape[1])  # x1
        scale_boxes[:, 1].clamp_(0, shape[0])  # y1
        scale_boxes[:, 2].clamp_(0, shape[1])  # x2
        scale_boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        scale_boxes[:, [0, 2]] = scale_boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        scale_boxes[:, [1, 3]] = scale_boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def nms(all_decorded_box, conf_thres=0.4, iou_thres=0.5):
    try:
        box_after_nms = do_class_nms(all_decorded_box, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=False)
    except:
        box_after_nms = do_class_nms(all_decorded_box, conf_thres=conf_thres, iou_thres=iou_thres)

    return box_after_nms

def make_grid(anchors, stride, nx=20, ny=20, cls_num=None):
    na = len(anchors) // 2  # number of anchors
   # print(" na: ", na)
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).expand((1, na, ny, nx, 2)).float()
    anchor_grid = anchors.view((1, na, 1, 1, 2)).expand((1, na, ny, nx, 2)).float()
    return grid, anchor_grid


def decord_infer_result(result, anchors, stride, cls_num, out):
    bs, _ , ny, nx , _ = result.shape
    grid, anchor_grid = make_grid(anchors, stride, nx, ny)
    y = result.float().sigmoid()
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
    out.append(y.contiguous().view(bs, -1, cls_num+5))


def do_class_nms(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        multi_label=False,
        nm=0,  # number of masks
):
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    nc = prediction.shape[2] - nm - 5  # number of classes
  
    mi = 5 + nc  # mask start index
    output = []
    print("conf: ", conf_thres , "  iou:  ", iou_thres)  
    for xi, x in enumerate(prediction):  # image index, image inference
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
       # conf, j = x[:, 5:].max(1, keepdim=True)
       # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        n = x.shape[0]  # number of boxes
        boxes, scores, classes = x[:, :4] , x[:, 4], x[:, 5]
        """
        i = local_nms(boxes, scores, iou_thres)
        output.append(x[i])
        """
        unique_classes = torch.unique(classes)
        for c in unique_classes:
            indexs = torch.where(classes == c)[0]
            b = boxes[indexs]
            s = scores[indexs]
            i = local_nms(b, s, iou_thres)  # NMS
            original_indices = indexs[i]
            output.append(x[original_indices])
        
    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def local_nms(boxes, scores, iou_thres):

    # 将列表转换为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 获取根据分数降序排序的索引
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []  # 保留的索引列表
    
    while sorted_indices.size > 0:
        # 选择当前最高分数的框
        current_index = sorted_indices[0]
        keep.append(current_index)
        
        # 如果只剩一个框，直接结束
        if sorted_indices.size == 1:
            break
        
        # 获取当前框坐标
        current_box = boxes[current_index]
        
        # 计算当前框与其他所有框的IoU
        other_boxes = boxes[sorted_indices[1:]]
        
        # 计算交集区域的坐标
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        # 计算交集区域面积
        intersection = np.maximum(0.0, x2 - x1 + 0.00001) * np.maximum(0.0, y2 - y1 + 0.00001)
        
        # 计算两个框各自的面积
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        # 计算并集面积和IoU
        union = current_area + other_areas - intersection
        iou = intersection / union
        
        # 保留IoU小于阈值的框索引
        remaining_indices = np.where(iou <= iou_thres)[0]
        
        # 更新排序后的索引列表（跳过当前框+保留的框）
        sorted_indices = sorted_indices[remaining_indices + 1]
    return keep

def coco80_to_coco91_class():
    # converts 80-index (val2014/val2017) to 91-index (paper)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def evaluate(cocoGt_file, cocoDt_file):
    print("cocoGt_file: ", cocoGt_file, " cocoDt_file: " , cocoDt_file)
    cocoGt = COCO(cocoGt_file + 'instances_val2017.json')
    cocoDt = cocoGt.loadRes(cocoDt_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    print("postprocess: ")
    parser = argparse.ArgumentParser(description='YOLOv3 offline model inference.')
    parser.add_argument('--ground_truth_json',
                        type=str,
                        default="coco/instances_val2017.json")
    parser.add_argument('--cfg_file',
                        type=str,
                        default='./script/model.yaml')
    parser.add_argument('--output',
                        default="./data/result/bin",
                        type=str)
    
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    pred_json_file = "end_predictions.json"

    postprocess(opt, cfg)
