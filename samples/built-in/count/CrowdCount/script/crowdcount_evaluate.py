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
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def load_gt_counts(mat_file_path):
    """加载真实人数标注（从mat文件中读取）"""
    gt_counts = {}
    # 加载mat标注文件
    mat_annotation = sio.loadmat(mat_file_path)
    # 提取真实人数、坐标点和文件名
    counts_true = mat_annotation['count']
    filenames = mat_annotation['img']

    # 转换为字典格式：{图像名: 真实人数}
    for i in range(len(filenames)):
        # 确保文件名提取正确（处理可能的数组嵌套格式）
        img_name = filenames[i] if isinstance(filenames[i], str) else filenames[i][0]
        img_name = img_name.split(".")[0]
        # 真实人数通常为单元素数组，需要取出具体值
        gt_count = int(counts_true[0][i])
        gt_counts[img_name] = gt_count

    return gt_counts


def load_pred_counts(txt_dir):
    """加载预测人数结果（从txt目录中读取）"""
    pred_counts = {}
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        img_name = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(txt_dir, txt_file)

        try:
            with open(txt_path, 'r') as f:
                count_str = f.read().strip()
                pred_counts[img_name] = int(count_str)
        except ValueError:
            print(f"警告: 无效的预测值格式 '{txt_file}'，已跳过")
        except Exception as e:
            print(f"警告: 处理文件 '{txt_file}' 时出错: {e}，已跳过")

    return pred_counts


def calculate_metrics(gt_counts, pred_counts):
    """计算MAE（平均绝对误差）和RMSE（均方根误差）"""
    maes = []
    rmses = []
    missing_gts = []

    for img_name in pred_counts.keys():
        if img_name not in gt_counts:
            missing_gts.append(img_name)
            continue

        gt = gt_counts[img_name]
        pred = pred_counts[img_name]

        # 计算误差
        abs_error = abs(gt - pred)
        maes.append(abs_error)
        rmses.append(abs_error ** 2)

    # 打印缺失情况
    if missing_gts:
        print(f"警告: 以下图像无真实标注: {', '.join(missing_gts[:5])}{'...' if len(missing_gts) > 5 else ''}")

    # 计算平均值
    if not maes:
        return None, None

    mae = np.mean(maes)
    rmse = np.sqrt(np.mean(rmses))  # 计算均方根误差
    return mae, rmse


def save_count_results(pred_counts, output_csv):
    """保存预测计数结果到CSV文件"""
    try:
        with open(output_csv, 'w') as f:
            f.write("image_name,count\n")
            for img_name, count in pred_counts.items():
                f.write(f"{img_name},{count}\n")
        print(f"计数结果已保存至: {output_csv}")
    except Exception as e:
        print(f"警告: 保存计数结果CSV失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='CrowdCount评估脚本（计算MAE和RMSE）')
    parser.add_argument('--gt_mat', type=str, default='../../../../../datasets/CrowDensity/Crowdata/crow.mat',
                        help='真实人数标注mat文件路径（如crow.mat）')
    parser.add_argument('--pred_txt_dir', type=str, default='../out/result/txt',
                        help='预测人数结果txt文件目录')
    parser.add_argument('--output_file', type=str, default='../out/result/metrics_results.txt',
                        help='评估结果保存文件')
    parser.add_argument('--count_csv', type=str, default='../out/result/count_results.csv',
                        help='预测计数结果CSV保存路径')

    args = parser.parse_args()

    # 验证文件存在性
    if not os.path.exists(args.gt_mat):
        print(f"错误: 真实标注文件 {args.gt_mat} 不存在")
        return
    if not os.path.isdir(args.pred_txt_dir):
        print(f"错误: 预测结果txt目录 {args.pred_txt_dir} 不存在")
        return

    # 加载数据
    print("加载真实标注...")
    gt_counts = load_gt_counts(args.gt_mat)
    print("加载预测结果...")
    pred_counts = load_pred_counts(args.pred_txt_dir)

    if not gt_counts:
        print("错误: 未加载到有效的真实标注数据")
        return
    if not pred_counts:
        print("错误: 未加载到有效的预测结果数据")
        return

    # 保存计数结果CSV
    save_count_results(pred_counts, args.count_csv)

    # 计算指标
    print("计算评估指标...")
    mae, rmse = calculate_metrics(gt_counts, pred_counts)

    if mae is None or rmse is None:
        print("错误: 没有有效的数据对用于计算指标")
        return

    # 输出结果
    result_str = (f"评估结果 (预测样本数: {len(pred_counts)})\n"
                  f"MAE (平均绝对误差): {mae:.4f}\n"
                  f"RMSE (均方根误差): {rmse:.4f}")
    print(result_str)

    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        f.write(result_str)
    print(f"评估结果已保存至 {args.output_file}")


if __name__ == '__main__':
    main()
