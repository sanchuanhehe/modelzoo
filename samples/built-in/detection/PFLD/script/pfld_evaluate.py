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
import numpy as np
import argparse
from scipy.integrate import simps


def compute_nme(preds, target):
    """计算归一化平均误差（NME）"""
    N = preds.shape[0]
    L = preds.shape[1]
    nmes = np.zeros(N)

    for index in range(N):
        npu_pred, gt = preds[index,], target[index,]
        if L == 19:
            interocular = 34  # 固定值（根据原代码逻辑）
        elif L == 29:
            interocular = np.linalg.norm(gt[8,] - gt[9,])
        elif L == 68:
            interocular = np.linalg.norm(gt[36,] - gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(gt[60,] - gt[72,])
        else:
            raise ValueError(f'不支持的关键点数量: {L}')

        nmes[index] = np.sum(np.linalg.norm(npu_pred - gt, axis=1)) / (interocular * L)

    return nmes


def compute_auc(errors, failureThreshold, step=0.0001):
    """计算AUC和失败率"""
    x_axis = np.arange(0., failureThreshold + step, step)
    n_errors = len(errors)
    ced = [float(np.count_nonzero(errors <= x)) / n_errors for x in x_axis]

    auc = simps(ced, x=x_axis) / failureThreshold
    failure_rate = 1. - ced[-1]
    return auc, failure_rate


def load_pred_landmarks(txt_dir):
    """加载预测的关键点结果（从txt文件）"""
    pred_landmarks = {}
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        img_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(txt_dir, txt_file)

        try:
            with open(txt_path, 'r') as f:
                landmarks = []
                for line in f.readlines():
                    x, y = map(float, line.strip().split())
                    landmarks.append([x, y])
                pred_landmarks[img_name] = np.array(landmarks, dtype=np.float32)
        except Exception as e:
            print(f"警告: 处理预测文件 {txt_file} 失败: {e}，跳过")

    return pred_landmarks


def load_gt_landmarks(list_file, target_size=(112, 112)):
    """加载真实关键点标注（从list.txt）"""
    gt_landmarks = {}
    with open(list_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        if not line:
            continue
        img_path = line[0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 解析真实关键点（前196个值为98个关键点的坐标）
        try:
            landmark_gt = np.asarray(line[1:197], dtype=np.float32)
            landmark_gt = landmark_gt.reshape(-1, 2) * target_size  # 缩放至目标尺寸
            gt_landmarks[img_name] = landmark_gt
        except Exception as e:
            print(f"警告: 处理标注文件 {img_name} 失败: {e}，跳过")

    return gt_landmarks


def calculate_metrics(gt_landmarks, pred_landmarks):
    """计算NME、AUC等评估指标"""
    nme_list = []
    missing_gts = []
    missing_preds = []

    # 检查所有预测结果对应的真实标注
    for img_name, pred in pred_landmarks.items():
        if img_name not in gt_landmarks:
            missing_gts.append(img_name)
            continue
        gt = gt_landmarks[img_name]

        # 确保关键点数量一致
        if pred.shape != gt.shape:
            print(f"警告: {img_name} 预测与真实关键点数量不匹配，跳过")
            continue

        # 计算NME
        nme = compute_nme(pred[np.newaxis, ...], gt[np.newaxis, ...])[0]
        nme_list.append(nme)

    # 检查所有真实标注是否都有预测结果
    for img_name in gt_landmarks:
        if img_name not in pred_landmarks:
            missing_preds.append(img_name)

    # 打印缺失情况
    if missing_gts:
        print(f"警告: 以下图像无真实标注: {', '.join(missing_gts[:5])}{'...' if len(missing_gts) > 5 else ''}")
    if missing_preds:
        print(f"警告: 以下图像无预测结果: {', '.join(missing_preds[:5])}{'...' if len(missing_preds) > 5 else ''}")

    if not nme_list:
        return None, None, None

    mean_nme = np.mean(nme_list)
    auc, failure_rate = compute_auc(nme_list, 0.1)
    return mean_nme, auc, failure_rate


def main():
    parser = argparse.ArgumentParser(description='PFLD评估脚本（计算NME和AUC）')
    parser.add_argument('--pred_txt_dir', default="../out/result/txt",
                        help='预测关键点txt文件目录')
    parser.add_argument('--gt_list_file', default="../data/wflw_test_cropped/list.txt",
                        help='真实标注list文件路径')
    parser.add_argument('--output_file', default="../out/result/metrics_results.txt",
                        help='评估结果保存文件')
    parser.add_argument('--target_size', type=int, nargs=2, default=[112, 112],
                        help='坐标缩放尺寸，需与后处理保持一致')

    args = parser.parse_args()

    # 验证文件存在性
    if not os.path.isdir(args.pred_txt_dir):
        print(f"错误: 预测结果目录 {args.pred_txt_dir} 不存在")
        return
    if not os.path.exists(args.gt_list_file):
        print(f"错误: 真实标注文件 {args.gt_list_file} 不存在")
        return

    # 加载数据
    print("加载预测结果...")
    pred_landmarks = load_pred_landmarks(args.pred_txt_dir)
    print("加载真实标注...")
    gt_landmarks = load_gt_landmarks(args.gt_list_file, args.target_size)

    if not pred_landmarks:
        print("错误: 未加载到有效的预测结果")
        return
    if not gt_landmarks:
        print("错误: 未加载到有效的真实标注")
        return

    # 计算指标
    print("计算评估指标...")
    mean_nme, auc, failure_rate = calculate_metrics(gt_landmarks, pred_landmarks)

    if mean_nme is None:
        print("错误: 没有有效的数据对用于计算指标")
        return

    # 输出结果
    result_str = (f"评估结果（有效样本数: {len(pred_landmarks) - len([1 for k in pred_landmarks if k not in gt_landmarks])})\n"
                  f"平均NME: {mean_nme:.4f}\n"
                  f"AUC@0.1: {auc:.4f}\n"
                  f"失败率: {failure_rate:.4f}")
    print(result_str)

    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        f.write(result_str)
    print(f"评估结果已保存至: {args.output_file}")


if __name__ == "__main__":
    main()
