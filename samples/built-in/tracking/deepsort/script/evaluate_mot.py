# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSORT NPU 多数据集批量评估工具
自动遍历所有 MOT16 序列，计算单序列及综合 MOTA/IDF1 等指标。

用法:
    # 评估全部序列 (默认从 ~/shared/ 读取 NPU 结果)
    python evaluate_mot.py

    # 指定结果目录和 GT 目录
    python evaluate_mot.py --ts_dir ~/shared --gt_dir /home/hispark/pic/MOT16/train

    # 只评估单个序列 (兼容旧用法)
    python evaluate_mot.py --gt /path/to/gt.txt --ts /path/to/result.txt

前置条件:
    pip install motmetrics pandas
"""

import os
import sys
import argparse
import motmetrics as mm
import pandas as pd


SEQS = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOT16 Tracking Results")
    # 批量模式
    parser.add_argument("--ts_dir", type=str, default=os.path.expanduser("~/shared"),
                        help="NPU 跟踪结果文件所在目录 (默认: ~/shared)")
    parser.add_argument("--gt_dir", type=str, default="/home/hispark/pic/MOT16/train",
                        help="MOT16 Ground Truth 根目录 (默认: /home/hispark/pic/MOT16/train)")
    # 单文件模式 (向后兼容)
    parser.add_argument("--gt", type=str, default=None, help="单个 GT 文件路径")
    parser.add_argument("--ts", type=str, default=None, help="单个跟踪结果文件路径")
    return parser.parse_args()


def evaluate_single(gt_path, ts_path, name='NPU_Evaluation'):
    """评估单个序列，返回 accumulator"""
    if not os.path.exists(gt_path):
        print(f"  [跳过] GT 不存在: {gt_path}")
        return None
    if not os.path.exists(ts_path):
        print(f"  [跳过] 结果不存在: {ts_path}")
        return None

    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=1)
    ts = mm.io.loadtxt(ts_path, fmt='mot15-2D')
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    return acc


def main():
    args = parse_args()

    # ---- 单文件模式 (向后兼容) ----
    if args.gt and args.ts:
        print("====================================")
        print("正在加载跟踪文件与之进行比对打分...")
        print(f"Ground Truth: {args.gt}")
        print(f"Tracker Res : {args.ts}")
        print("====================================")

        acc = evaluate_single(args.gt, args.ts)
        if acc is None:
            sys.exit(1)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='NPU_Evaluation')
        str_summary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
        print("\n------------------- 打分结果报告 -------------------")
        print(str_summary)
        return

    # ---- 批量模式 ----
    print("=" * 60)
    print("  DeepSORT NPU 多数据集批量评估")
    print(f"  GT 目录  : {args.gt_dir}")
    print(f"  结果目录 : {args.ts_dir}")
    print("=" * 60)

    accs = []
    names = []

    for seq in SEQS:
        gt_path = os.path.join(args.gt_dir, seq, "gt", "gt.txt")
        ts_path = os.path.join(args.ts_dir, f"{seq}-npu.txt")

        print(f"\n[{seq}] GT: {gt_path}")
        print(f"{'':8s} TS: {ts_path}")

        acc = evaluate_single(gt_path, ts_path, name=seq)
        if acc is not None:
            accs.append(acc)
            names.append(seq)
            print(f"  ✓ 加载成功")
        else:
            print(f"  ✗ 跳过")

    if not accs:
        print("\n[错误] 没有找到任何有效的评估对！")
        sys.exit(1)

    # 计算单序列指标 + 综合指标 (OVERALL)
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=names, generate_overall=True)

    # 格式化输出
    str_summary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )

    print("\n")
    print("=" * 60)
    print("  NPU 端 MOT16 评估结果报告")
    print("=" * 60)
    print(str_summary)

    # 同时保存到文件
    out_path = os.path.join(args.ts_dir, "npu_eval_report.txt")
    with open(out_path, 'w') as f:
        f.write("DeepSORT NPU MOT16 Evaluation Report\n")
        f.write(f"GT Dir : {args.gt_dir}\n")
        f.write(f"TS Dir : {args.ts_dir}\n\n")
        f.write(str_summary)
    print(f"\n报告已保存到: {out_path}")


if __name__ == '__main__':
    main()
