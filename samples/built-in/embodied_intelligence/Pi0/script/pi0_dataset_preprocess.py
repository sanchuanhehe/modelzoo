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

import os
import numpy as np
from PIL import Image
import av
import logging
import json
from datasets import load_dataset

# 使用 PyAV 提取视频帧
def extract_frames_from_video(extract_frame_num, video_path):
    output_dir = "../data/images"
    os.makedirs(output_dir, exist_ok=True)

    container = av.open(video_path)
    
    # 获取视频流
    video_stream = None
    for stream in container.streams:
        if stream.type == 'video':
            video_stream = stream
            break
    
    if video_stream is None:
        logging.info("not found video")
        return None
    
    # 跳到第一帧
    container.seek(0)
    idx = 0
    # 读取第一帧
    for frame in container.decode(video_stream):
        # 转换为 numpy 数组 (HWC 格式)
        img_array = frame.to_ndarray(format='rgb24')

        filename = f"frame_{idx:05d}.png"
        filepath = os.path.join(output_dir, filename)

        Image.fromarray(img_array).save(filepath)
        idx+=1
        if idx > extract_frame_num:
            break
    logging.info(f"extract frames success")


# 从数据集提取state和action数据
def extract_state_and_action(extract_frame_num, dataset_path):
    output_dir = "../data/dataset"
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset(dataset_path, streaming=True)
    features = dataset["train"].features
    field_names = list(features.keys())

    # 遍历整个数据集, 提取前extract_frame_num帧数据
    for idx, example in enumerate(dataset["train"]):
        state = None
        action = None
        if isinstance(example, tuple):
            # 将元组转换为字典
            example_dict = {}
            for idx, field_name in enumerate(field_names):
                if idx < len(example):
                    example_dict[field_name] = example[idx]
            state = np.array(example_dict["observation.state"])
            action = np.array(example_dict["action"])
        else:
            state = np.array(example["observation.state"])
            action = np.array(example["action"])
        state_filename = f"state_{idx:05d}.txt"
        action_filename = f"action_{idx:05d}.txt"
        state_filepath = os.path.join(output_dir, state_filename)
        action_filepath = os.path.join(output_dir, action_filename)
        np.savetxt(state_filepath, state)
        np.savetxt(action_filepath, action)
        if idx >= extract_frame_num:
            break
    logging.info(f"extract state and action success")

if __name__=='__main__':
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    dataset_path = "../data/aloha_sim_transfer_cube_scripted"

    extract_frame_num = 500 # 提取数据组数（视频帧数）
    extract_frames_from_video(extract_frame_num, dataset_path + "/videos/observation.images.top/chunk-000/file-000.mp4")

    extract_state_and_action(extract_frame_num, dataset_path)
