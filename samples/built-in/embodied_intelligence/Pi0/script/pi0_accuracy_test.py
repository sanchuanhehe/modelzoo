# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

import numpy as np
import logging
import argparse

def main():
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run pi0 accuracy test")
    parser.add_argument("--data-num", type=int, default=500, help="data nums for accuracy test") # 用于精度验证的数据组数
    parser.add_argument("--dataset-action-path", type=str, default="./data/dataset/", help="dataset action path")
    parser.add_argument("--out-action-path", type=str, default="./out/action/", help="output action path")
    args = parser.parse_args()
    dataset_action_arr = None
    output_action_arr = None
    for idx in range(args.data_num):
        action_filename = f"action_{idx:05d}.txt"
        # ../data/dataset/目录下为数据集中的action，可通过运行pi0_dataset_process.py获取
        dataset_action = np.loadtxt(args.dataset_action_path+ action_filename, dtype=np.float32)
        dataset_action = np.expand_dims(dataset_action, axis=0)
        # ../out/action/目录下为离线推理得到的action，可通过运行pi0_infer.py或cpp main程序获取
        output_action = np.loadtxt(args.out_action_path + action_filename, dtype=np.float32)
        output_action = output_action[0:1, :]
        if dataset_action_arr is None and output_action_arr is None:
            dataset_action_arr = dataset_action
            output_action_arr = output_action
        else:
            dataset_action_arr = np.vstack([dataset_action_arr, dataset_action])
            output_action_arr = np.vstack([output_action_arr, output_action])
    logging.info(f"dataset_action_arr is a numpy array of type {dataset_action_arr.dtype} and shape {dataset_action_arr.shape}")
    logging.info(f"output_action_arr is a numpy array of type {output_action_arr.dtype} and shape {output_action_arr.shape}")
    mae = np.mean(np.abs(dataset_action_arr - output_action_arr))
    rmse = np.sqrt(np.mean((dataset_action_arr - output_action_arr) ** 2))

    logging.info(f"平均绝对误差MAE: {mae}")
    logging.info(f"均方根误差RMSE: {rmse}")

if __name__=='__main__':
    main()