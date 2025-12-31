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
import math
from sklearn.model_selection import KFold
from scipy import interpolate
from tqdm import tqdm


def read_pairs(pairs_filename):
    """读取配对信息"""
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:  # 跳过第一行
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)


def get_paths(txt_dir, pairs):
    """获取图像路径和是否为同一人的标记"""
    path_list = []
    issame_list = []
    nrof_skipped_pairs = 0

    for pair in pairs:
        try:
            if len(pair) == 3:
                # 同一人配对
                name = pair[0]
                img1 = f"{name}_{int(pair[1]):04d}"
                img2 = f"{name}_{int(pair[2]):04d}"
                issame = True
            elif len(pair) == 4:
                # 不同人配对
                name1, name2 = pair[0], pair[2]
                img1 = f"{name1}_{int(pair[1]):04d}"
                img2 = f"{name2}_{int(pair[3]):04d}"
                issame = False
            else:
                nrof_skipped_pairs += 1
                continue

            # 构建特征文件路径
            path1 = os.path.join(txt_dir, f"{img1}.txt")
            path2 = os.path.join(txt_dir, f"{img2}.txt")

            if os.path.exists(path1) and os.path.exists(path2):
                path_list += [path1, path2]
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        except Exception as e:
            print(f"警告: 处理配对 {pair} 失败: {e}")
            nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print(f'跳过 {nrof_skipped_pairs} 个无效配对')
    return path_list, issame_list


def load_features(path_list):
    """加载特征向量"""
    embeddings = []
    for path in tqdm(path_list, desc="加载特征"):
        try:
            with open(path, 'r') as f:
                features = list(map(float, f.read().split()))
                embeddings.append(np.array(features, dtype=np.float32))
        except Exception as e:
            print(f"警告: 加载 {path} 失败: {e}")
            embeddings.append(np.zeros(512, dtype=np.float32))  # 用零向量替代
    return np.array(embeddings)


# 以下评估函数完成重命名整改，功能与原代码完全一致
def compute_feature_distance(feat_mat1, feat_mat2, dist_type=0):
    if dist_type == 0:
        # 欧氏距离计算
        diff_mat = np.subtract(feat_mat1, feat_mat2)
        dist_arr = np.sum(np.square(diff_mat), 1)
    elif dist_type == 1:
        # 基于余弦相似度的距离计算
        dot_prod = np.sum(np.multiply(feat_mat1, feat_mat2), axis=1)
        norm_val = np.linalg.norm(feat_mat1, axis=1) * np.linalg.norm(feat_mat2, axis=1)
        cos_sim = dot_prod / norm_val
        dist_arr = np.arccos(cos_sim) / math.pi
    else:
        raise ValueError(f'未定义的距离度量类型 {dist_type}')

    return dist_arr


def calc_roc_curve(threshold_list, feat_mat1, feat_mat2, label_same, fold_num=10, dist_type=0,
                   mean_subtract=False):
    assert (feat_mat1.shape[0] == feat_mat2.shape[0])
    assert (feat_mat1.shape[1] == feat_mat2.shape[1])
    pair_num = min(len(label_same), feat_mat1.shape[0])
    threshold_num = len(threshold_list)
    k_fold = KFold(n_splits=fold_num, shuffle=False)

    tpr_mat = np.zeros((fold_num, threshold_num))
    fpr_mat = np.zeros((fold_num, threshold_num))
    acc_arr = np.zeros((fold_num))

    false_pos_list = []
    false_neg_list = []

    index_arr = np.arange(pair_num)

    for fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(index_arr)):
        if mean_subtract:
            mean_val = np.mean(np.concatenate([feat_mat1[train_idx], feat_mat2[train_idx]]), axis=0)
        else:
            mean_val = 0.0
        dist_arr = compute_feature_distance(feat_mat1 - mean_val, feat_mat2 - mean_val, dist_type)

        # 寻找当前折叠的最佳阈值
        train_acc = np.zeros((threshold_num))
        for thr_idx, thr_val in enumerate(threshold_list):
            _, _, train_acc[thr_idx], _, _ = calc_acc_score(thr_val, dist_arr[train_idx],
                                                           label_same[train_idx])
        best_thr_idx = np.argmax(train_acc)
        for thr_idx, thr_val in enumerate(threshold_list):
            tpr_mat[fold_idx, thr_idx], fpr_mat[fold_idx, thr_idx], _, _, _ = calc_acc_score(thr_val,
                                                                                             dist_arr[test_idx],
                                                                                             label_same[test_idx])
        _, _, acc_arr[fold_idx], fp_item, fn_item = calc_acc_score(threshold_list[best_thr_idx], dist_arr[test_idx],
                                                                   label_same[test_idx])

        tpr_mean = np.mean(tpr_mat, 0)
        fpr_mean = np.mean(fpr_mat, 0)
        false_pos_list.extend(fp_item)
        false_neg_list.extend(fn_item)

    return tpr_mean, fpr_mean, acc_arr, false_pos_list, false_neg_list


def calc_acc_score(threshold, dist_arr, label_same):
    pred_same = np.less(dist_arr, threshold)
    tp_count = np.sum(np.logical_and(pred_same, label_same))
    fp_count = np.sum(np.logical_and(pred_same, np.logical_not(label_same)))
    tn_count = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(label_same)))
    fn_count = np.sum(np.logical_and(np.logical_not(pred_same), label_same))

    fp_flag = np.logical_and(pred_same, np.logical_not(label_same))
    fn_flag = np.logical_and(np.logical_not(pred_same), label_same)

    tpr_val = 0 if (tp_count + fn_count == 0) else float(tp_count) / float(tp_count + fn_count)
    fpr_val = 0 if (fp_count + tn_count == 0) else float(fp_count) / float(fp_count + tn_count)
    acc_val = float(tp_count + tn_count) / dist_arr.size
    return tpr_val, fpr_val, acc_val, fp_flag, fn_flag


def calc_val_accuracy(threshold_list, feat_mat1, feat_mat2, label_same, far_target_val, fold_num=10, dist_type=0,
                      mean_subtract=False):
    assert (feat_mat1.shape[0] == feat_mat2.shape[0])
    assert (feat_mat1.shape[1] == feat_mat2.shape[1])
    pair_num = min(len(label_same), feat_mat1.shape[0])
    threshold_num = len(threshold_list)
    k_fold = KFold(n_splits=fold_num, shuffle=False)

    val_arr = np.zeros(fold_num)
    far_arr = np.zeros(fold_num)

    index_arr = np.arange(pair_num)

    for fold_idx, (train_idx, test_idx) in enumerate(k_fold.split(index_arr)):
        if mean_subtract:
            mean_val = np.mean(np.concatenate([feat_mat1[train_idx], feat_mat2[train_idx]]), axis=0)
        else:
            mean_val = 0.0
        dist_arr = compute_feature_distance(feat_mat1 - mean_val, feat_mat2 - mean_val, dist_type)

        # 找到使FAR = far_target_val的阈值
        train_far = np.zeros(threshold_num)
        for thr_idx, thr_val in enumerate(threshold_list):
            _, train_far[thr_idx] = calc_val_far_rate(thr_val, dist_arr[train_idx], label_same[train_idx])

        # 去重逻辑
        unique_idx = np.unique(train_far, return_index=True)[1]
        unique_idx = np.sort(unique_idx)
        train_far_unique = train_far[unique_idx]
        threshold_unique = threshold_list[unique_idx]

        if np.max(train_far_unique) >= far_target_val:
            interp_fun = interpolate.interp1d(train_far_unique, threshold_unique, kind='slinear')
            thr_selected = interp_fun(far_target_val)
        else:
            thr_selected = 0.0

        val_arr[fold_idx], far_arr[fold_idx] = calc_val_far_rate(thr_selected, dist_arr[test_idx], label_same[test_idx])

    val_mean = np.mean(val_arr)
    far_mean = np.mean(far_arr)
    val_std = np.std(val_arr)
    return val_mean, val_std, far_mean


def calc_val_far_rate(threshold, dist_arr, label_same):
    pred_same = np.less(dist_arr, threshold)
    true_accept_count = np.sum(np.logical_and(pred_same, label_same))
    false_accept_count = np.sum(np.logical_and(pred_same, np.logical_not(label_same)))
    same_num = np.sum(label_same)
    diff_num = np.sum(np.logical_not(label_same))
    val_acc = float(true_accept_count) / float(same_num) if same_num != 0 else 0.0
    far_rate = float(false_accept_count) / float(diff_num) if diff_num != 0 else 0.0
    return val_acc, far_rate


def feature_evaluation(feat_embeddings, label_same, fold_num=10, dist_type=0, mean_subtract=False):
    # 计算评估指标
    threshold_list = np.arange(0, 4, 0.01)
    feat_mat1 = feat_embeddings[0::2]
    feat_mat2 = feat_embeddings[1::2]
    tpr_curve, fpr_curve, acc_result, fp_result, fn_result = calc_roc_curve(threshold_list, feat_mat1, feat_mat2,
                                                                            np.asarray(label_same), fold_num=fold_num,
                                                                            dist_type=dist_type, mean_subtract=mean_subtract)
    threshold_list = np.arange(0, 4, 0.001)
    val_acc, val_std_dev, far_rate = calc_val_accuracy(threshold_list, feat_mat1, feat_mat2,
                                                       np.asarray(label_same), 1e-3, fold_num=fold_num,
                                                       dist_type=dist_type, mean_subtract=mean_subtract)
    return tpr_curve, fpr_curve, acc_result, val_acc, val_std_dev, far_rate, fp_result, fn_result


def run_evaluation(args):
    """执行评估"""
    # 读取配对信息
    pairs = read_pairs(args.pairs_path)

    # 获取图像路径和标记
    path_list, issame_list = get_paths(args.txt_dir, pairs)
    if not path_list:
        print("错误: 未找到有效的特征文件对")
        return

    # 加载特征向量
    embeddings = load_features(path_list)

    # 执行评估（使用整改后的评估逻辑，功能与原代码一致）
    tpr, fpr, accuracy, val, val_std, far, fp, fn = feature_evaluation(
        embeddings,
        issame_list,
        fold_num=10,
        dist_type=args.distance_metric,
        mean_subtract=args.subtract_mean
    )

    # 输出结果
    print(f"各折准确率: {accuracy}")
    print(f"平均准确率: {np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}")
    print(f"验证集准确率 (FAR=1e-3): {val:.4f} ± {val_std:.4f}")
    print(f"实际FAR: {far:.6f}")


def main():
    parser = argparse.ArgumentParser(description='FaceNet评估脚本')
    parser.add_argument('--pairs_path', default='../../../../../datasets/LFW/pairs.txt',
                        help='LFW配对文件路径')
    parser.add_argument('--txt_dir', default='../out/result/txt',
                        help='特征向量txt文件目录')
    parser.add_argument('--distance_metric', type=int, default=0, choices=[0, 1],
                        help='距离度量方式: 0-欧氏距离, 1-余弦距离')
    parser.add_argument('--subtract_mean', action='store_true',
                        help='是否在计算距离前减去均值')
    args = parser.parse_args()

    if not os.path.isdir(args.txt_dir):
        print(f"错误: 特征文件目录 {args.txt_dir} 不存在")
        return

    if not os.path.exists(args.pairs_path):
        print(f"错误: 配对文件 {args.pairs_path} 不存在")
        return

    run_evaluation(args)


if __name__ == "__main__":
    main()