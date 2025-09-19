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

import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

from torch.utils.data import DataLoader
from dataset.datasets import WLFWDatasets
import argparse
import os
# from auto_optimizer import OnnxGraph
from tqdm import tqdm
from pathlib import Path

def compute_nme(preds, target):
    
    N = preds.shape[0]
    L = preds.shape[1]
    nmes = np.zeros(N)

    for index in range(N):
        npu_pred, gt = preds[index, ], target[index, ]
        if L == 19:
            interocular = 34
        elif L == 29:
            interocular = np.linalg.norm(gt[8, ] - gt[9, ])
        elif L == 68:
            interocular = np.linalg.norm(gt[36, ] - gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(gt[60, ] - gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        nmes[index] = np.sum(np.linalg.norm(npu_pred - gt,
                                        axis=1)) / (interocular * L)

    return nmes


def compute_auc(errors, failureThreshold, step=0.0001):
    xAxis = list(np.arange(0., failureThreshold + step, step))
    nErrors = len(errors)
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]
    return AUC, failureRate

def post():
    # onnx_session = OnnxGraph.parse(opt.onnx)
    # outputs = onnx_session.outputs
    output = os.listdir(args.output)
    sorted_files = sorted(output)
    print("postprocess: ")
    # the output shapes

    pred_results = []
    nme_list = []
    file_list = './data/test_data/list.txt'
    with open(file_list, 'r') as f:
        lines = f.readlines()
    for i in tqdm(range(len(sorted_files) // 2)):
        line = lines[i].strip().split()
        print('img_path: ', line[0])
        out_path = Path(line[0]).stem
        out = []
        out_filepath = f"{args.output}/{out_path}_1.bin"
    
        inference_result = np.fromfile(out_filepath, dtype=np.float32)
        landmarks = np.array(inference_result).reshape(1,-1, 2) * [112,112]  # landmark
        img = cv2.imread(line[0])
        landmark_gt = np.asarray(line[1:197], dtype=np.float32)
        landmark_gt = landmark_gt.reshape(1, -1,2) * [112,112]  # landmark_gt
        attribute = np.asarray(line[197:203], dtype=np.int32)
        euler_angle = np.asarray(line[203:206], dtype=np.float32)
        nme_temp = compute_nme(landmarks, landmark_gt)
        for item in nme_temp:
            nme_list.append(item)
    print('nme: {:.4f}'.format(np.mean(nme_list)))
    auc, failure_rate = compute_auc(nme_list, 0.1)
    print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
        0.1, auc))
    print('failure_rate: {:}'.format(failure_rate))

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--output', default="../out/result/bin",type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    post()