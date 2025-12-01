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
import time
import argparse
import torch
import multiprocessing
import numpy as np
from PIL import Image
import torch
from torch import Tensor

"""
    计算两个掩码图像中指定类别的并集像素数量
    
    并集指的是在预测掩码或真实掩码中至少有一个标记为目标类别的像素位置
    
    参数:
        infer_img: 二维数组，模型预测的分割掩码
        label: 二维数组，真实的标注掩码
        i: 整数，要计算的目标类别标识符
        
    返回:
        int: 并集像素的总数量
"""
def get_union(infer_img, label, i):

    counter = 0
    for infer_height, label_height in zip(infer_img, label):
        for w_infer, w_label in zip(infer_height, label_height):
            if w_infer == i or w_label == i:
                counter += 1 # 满足条件则计数器+1
    return counter # 返回统计结果

"""
    计算两个掩码图像中指定类别的交集像素数量
    
    交集指的是在预测掩码和真实掩码中都标记为目标类别的像素位置
    
    参数:
        infer_img: 二维数组，模型预测的分割掩码
        label: 二维数组，真实的标注掩码
        i: 整数，要计算的目标类别标识符
        
    返回:
        int: 交集像素的总数量
"""
def get_intersection(infer_img, label, i):
    
    counter = 0
    # 并行遍历图像的行
    for infer_height, label_height in zip(infer_img, label):
        for w_infer, w_label in zip(infer_height, label_height):
            # 检查当前像素位置：只要推断图或标签图中有一个值为i即计数
            if w_infer == i and w_label == i:
                counter += 1 # 满足条件则计数器+1
    return counter # 返回统计结果

"""
    计算预测掩码和真实掩码之间的平均交并比(IoU)，仅考虑IoU值不低于0.5的类别
    
    IoU(Intersection over Union)是分割任务中常用的评估指标，计算公式为：
    IoU = (预测与真实的交集面积) / (预测与真实的并集面积)
    
    参数:
        prediction_mask: numpy数组，模型预测的分割掩码
        ground_truth_mask: numpy数组，真实的标注掩码
        
    返回:
        float: 符合条件的类别IoU平均值，如果没有符合条件的类别则返回0
"""
def getIoU(prediction_mask, ground_truth_mask):
    iou = 0.0
    counter = 0
    for i in np.unique(prediction_mask):
        if i == 0 or i > 21:
            # 跳过背景类别（0）
            #跳过无效类别（ID > 21）
            continue

        #通过得到交并集来计算IOU
        intersection = get_intersection(prediction_mask, ground_truth_mask, i)
        union = get_union(prediction_mask, ground_truth_mask, i)

        # 避免除零错误
        if union == 0:
            continue
        sub_iou_result = float(intersection) / union

        # 只考虑IoU不低于0.5的类别
        if sub_iou_result < 0.5: 
            continue
        iou += sub_iou_result
        counter += 1

    # 计算平均IoU（如果有有效类别）
    if counter == 0:
        return 0
    else:
        return iou / counter

def label_process(label_img):
    print('label_process post: ', label_img)
    label_img = Image.open(label_img)
    label_img = label_img.resize((572, 572))
    label_img = np.array(label_img, dtype=np.uint8)
    return label_img

def postprocess(file):
    print('infer post: ', file)
    infer_result = torch.from_numpy(np.fromfile(os.path.join(output_file, file), np.float32).reshape((572, 572)))
    infer_result = torch.sigmoid(infer_result)
    infer_result = (infer_result.numpy() > 0.5).astype(np.uint8)
    return infer_result

"""
    计算模型预测结果与真实标签之间的Dice相似系数(DSC)
    
    Dice系数是医学图像分割中常用的评估指标，衡量两个样本的相似度，
    计算公式为: DSC = 2 * |X ∩ Y| / (|X| + |Y|)
    其中X是预测结果，Y是真实标签，值域为[0, 1]，1表示完全匹配
    
    参数:
        prediction_file_path: str, 模型预测结果的二进制文件路径
        ground_truth_file_path: str, 真实标注标签的图像文件路径
        
    返回:
        float: 预测结果与真实标签之间的Dice系数
"""
def evaluate_dice_coefficient(prediction_file_path, ground_truth_file_path):
    print('eval_res')
    # 1. 加载并处理模型预测结果
    # 从二进制文件读取预测数据并转换为572x572的张量
    infer_result = torch.from_numpy(
        np.fromfile(os.path.join(output_file, prediction_file_path),
        np.float32)
        .reshape((572, 572)))
    # 应用sigmoid函数将输出转换为概率值(0到1之间)
    infer_result = torch.sigmoid(infer_result)
    # 二值化处理：概率大于0.5的像素视为正类(前景)，其余为负类(背景)
    infer_result = infer_result > 0.5
    infer_result = infer_result.to(dtype=torch.float32)

    # 2. 加载并处理真实标签
    # 打开真实标签图像文件
    binary_prediction = Image.open(os.path.join(label_file, ground_truth_file_path))
    # 调整图像大小以匹配预测结果的分辨率
    binary_prediction = binary_prediction.resize((572, 572))
    binary_prediction = np.array(binary_prediction )
    binary_prediction = torch.from_numpy(binary_prediction )
    binary_prediction = binary_prediction.to(dtype=torch.float32)
    # 3. 计算Dice系数
    return dice_coeff(infer_result, binary_prediction ).item()

"""
    dice参数计算：Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    参数:
        input: 预测结果的张量
        target: 真实标签的张量
        reduce_batch_first: 是否首先在批次维度上进行归约
        epsilon: 平滑项，防止除零错误
        
"""
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    #计算交集
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    #计算并集
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

"""
    处理一个文件批次的IoU评估，提供详细日志和错误处理
    
    参数:
        batch_list: 包含所有文件批次的列表
        batch_index: 当前要处理的批次索引
        
"""
def get_iou(batch_list, batch_index, accuray_file):
    print("start batch_index: ", batch_index ) 
    sum_evaluate = 0.0
    for file in batch_list[batch_index]:
        # 计算Dice系数
        evaluate = evaluate_dice_coefficient(file, file.replace('_0.bin', '_mask.gif'))
        sum_evaluate += evaluate

        # 处理预测和标签数据
        infer_val = postprocess(file)
        label_val = label_process(os.path.join(label_file, file.replace('_0.bin', '_mask.gif')))
        # 计算IoU
        iou = getIoU(infer_val, label_val)
        if iou == 0:  # it's difficult
            print("  ====== " 
            + str(batch_index)
            + " infer_ result"
            + file
            + " has iou : "
            + str(0)
            + "  ====== " ) 
            continue
        print("  ====== " 
            + str(batch_index)
            + " infer_ result"
            + file
            + " has iou : "
            + str(iou)
            + "  ====== " ) 
        #由于是并发处理数据，所有写文件的时候必须加锁
        lock.acquire()
        try:
            with open(accuray_file, 'a') as f:
                f.write('{}, '.format(iou))
        except:
            lock.release()
        lock.release()
        
    print("eval value is", sum_evaluate / len(batch_list[batch]))

"""计算最终结果"""
def calculate_final_results(accuracy_file):
    #从文件中读取处理后的结果，计算最终IOU值
    try:
        if not os.path.exists(accuracy_file):
            raise FileNotFoundError(f"文件不存在: {accuracy_file}")
        
        # 检查文件是否为空
        if os.path.getsize(accuracy_file) == 0:
            raise ValueError(f"文件为空: {accuracy_file}")
        with open(accuracy_file) as f:
            ret = list(map(float, f.read().replace(', ', ' ').strip().split(' ')))
        print('IOU Average : {}'.format(sum(ret) / len(ret)))
    except Exception as e:
        print('处理数据失败...')
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")

"""
    多进程IoU评估主程序
    
    此程序使用多进程并行处理图像分割结果，计算每个结果的IoU(交并比)指标，
    并统计所有结果的平均IoU值。
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./out/result/bin')
    parser.add_argument('--label', default='./carvana/train_masks')
    parser.add_argument('--result', default='./accuracy.txt')
    args = parser.parse_args()

    output_file = args.output
    label_file = args.label
    accuray_file = args.result

    if accuray_file in os.listdir(os.getcwd()):
        os.remove(accuray_file)

    #每一个批次处理300个文件
    infer_list = os.listdir(output_file)
    batch_list = [infer_list[i:i + 300] for i in range(0, 5000, 300) if infer_list[i:i + 300] != []]

    lock = multiprocessing.Lock()
    multipool = multiprocessing.Pool(len(batch_list))
    for batch in range(len(batch_list)):
        multipool.apply_async(get_iou, args=(batch_list, batch, accuray_file))
        print(f"批次 {batch} 处理完成: ")
    multipool.close()
    multipool.join()
    calculate_final_results(accuray_file)


