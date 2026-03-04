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

# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from collections import deque
import logging
import os
import time
import av
import numpy as np
import torch
from torch import Tensor
from datasets import load_dataset
import einops

from lerobot.policies.normalize import Unnormalize
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from ais_bench.infer.interface import InferSession
logger = logging.getLogger(__name__)

def extract_state_and_action(index, dataset_dir):
    dataset = load_dataset(dataset_dir, streaming=True)
    features = dataset["train"].features
    field_names = list(features.keys())
    state_arr = None
    action_arr = None
    for i, example in enumerate(dataset["train"].take(index)):
        state = None
        action = None
        if isinstance(example, tuple):
            # 将元组转换为字典
            example_dict = {}
            for idx, field_name in enumerate(field_names):
                if idx < len(example):
                    example_dict[field_name] = example[idx]
            
            logger.info(f"样本 {i} 转换为字典:")
            for key, value in example_dict.items():
                logger.info(f"  {key}: {type(value)}")
            # 现在可以用字符串访问
            state = example_dict["observation.state"]
            action = example_dict["action"]
        else:
            state = np.array(example["observation.state"])
            action = np.array(example["action"])

        if state_arr is None and action_arr is None:
            state_arr = np.array(state, dtype=np.float32)
            action_arr = np.array(action, dtype=np.float32)
        else:
            state_arr = np.vstack([state_arr, np.array(state, dtype=np.float32)])
            action_arr = np.vstack([action_arr, np.array(action, dtype=np.float32)])
    logger.info("state_arr.shape: %s", state_arr.shape)
    logger.info("action_arr.shape: %s", action_arr.shape)
    return state_arr, action_arr

def extract_frames_from_video(index, video_path):
    """
    使用 PyAV 提取视频帧
    """
    container = av.open(video_path)
    # 获取视频流
    video_stream = None
    for stream in container.streams:
        if stream.type == 'video':
            video_stream = stream
            break

    if video_stream is None:
        logger.info("未找到视频流")
        return None
    
    # 跳到第一帧
    container.seek(0)
    i = 0
    img_arr = None
    # 读取第一帧
    for frame in container.decode(video_stream):
        i+=1
        img_array = frame.to_ndarray(format='rgb24')
        img = torch.from_numpy(img_array)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        _, h, w, c = img.shape
        if not (c < h and c < w):
            logger.error(f"expect channel last images, but instead got {img.shape=}")

        if not (img.dtype == torch.uint8):
            logger.error(f"expect torch.uint8, but instead {img.dtype=}")

        # 将img的值转换到 [0,1]区间
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)

        if img_arr is None:
            img_arr = img.cpu().numpy()
        else:
            img_arr = np.vstack([img_arr, img.cpu().numpy()])
        
        if i >= index:
            logger.info("img_arr.shape: %s", img_arr.shape)
            return img_arr
    return img_arr

def build_lang_inputs(token_ids, max_len=48, pad_token_id=0):
    """
    token_ids: List[int]，原始的 token 序列
    max_len: 固定的序列长度
    pad_token_id: padding 用的 id，默认是 0
    """
    # 截断或者补齐
    tokens = token_ids[:max_len]
    if len(tokens) < max_len:
        tokens += [pad_token_id] * (max_len - len(tokens))
    lang_tokens = torch.tensor([tokens], dtype=torch.long, device="cpu")  # shape: [1, max_len]

    # 生成 mask
    lang_masks = (lang_tokens != pad_token_id) # True 表示有效 token

    return lang_tokens, lang_masks

def prepare_language(batch) -> tuple[Tensor, Tensor]:
    # 这里的token_ids对应aloha trans cube任务: Pick up the cube with the right arm and transfer it to the left arm.
    token_ids = [2, 21585, 908, 573, 28660, 675, 573, 1833, 7762, 578, 5853, 665, 577, 573, 2731, 7762, 235265]

    lang_tokens, lang_masks = build_lang_inputs(token_ids, max_len=48)
    return lang_tokens, lang_masks

class Pi0(object):
    def __init__(self, model_path, config):
        self._model_path = model_path
        self._action_queue = deque()

        def _get(cfg, key):
            return cfg[key] if isinstance(cfg, dict) else getattr(cfg, key)

        output_features = _get(config, "output_features")
        normalization_mapping = _get(config, "normalization_mapping")
        stats = _get(config, "stats")
        if stats is None:
            raise ValueError("Unnormalize stats are required; provide mean/std via CLI or config")

        self.unnormalize = Unnormalize(output_features, normalization_mapping, stats)
        self._session = InferSession(device_id=0, model_path=self._model_path)

    def interface(self, state, image, lang_tokens, lang_masks=None):
        if lang_tokens is None:
            raise ValueError("lang_tokens must be provided for language inputs")

        if isinstance(lang_tokens, np.ndarray):
            lang_tokens_t = torch.from_numpy(lang_tokens)
        else:
            lang_tokens_t = lang_tokens

        if lang_masks is None:
            lang_masks_t = (lang_tokens_t != 0)
        else:
            lang_masks_t = lang_masks if isinstance(lang_masks, torch.Tensor) else torch.from_numpy(lang_masks)

        lang_tokens_np = lang_tokens_t.cpu().numpy().astype(np.int64)
        lang_masks_np = lang_masks_t.cpu().numpy().astype(np.bool_)

        input_list = [lang_tokens_np, lang_masks_np, state.astype(np.float32), image.astype(np.float32)]

        start0 = time.time()
        output = self._session.infer(feeds=input_list)
        action_shape = (1, 50, 32)
        output[0] = output[0].reshape(action_shape)
        logger.info("[TIMER] om infer time: %.3f s", (time.time() - start0))

        actions = self.extract_valid_result(output[0], target_dim=14)
        tensor_actions = torch.from_numpy(actions)
        final_actions = self.unnormalize({"action": tensor_actions})["action"]
        logger.info("[TIMER] interface time: %.3f s", (time.time() - start0))
        return final_actions

    def select_action(self, state, image, lang_tokens, lang_masks=None):
        """
        根据模型输入离线推理得到action序列, 并存储到action queue
        """
        # if not self._action_queue:
        #     logger.info("Generate new action sequence")
        #     action_array = self.interface(state, image, lang_tokens, lang_masks)

        #     for i in range(action_array[0].shape[0]):
        #         self._action_queue.append(action_array[0][i])
        # return self._action_queue.popleft()
        logger.info("Generate new action sequence")
        action_array = self.interface(state, image, lang_tokens, lang_masks)
        return action_array[0]

    def extract_valid_result(self, actions: np.ndarray, target_dim: int = 14, indices: list[int] | None = None) -> np.ndarray:
        """
        将动作序列从 (..., 32) 转成 (..., 14)。
        """
        arr = np.asarray(actions)
        if arr.ndim == 3:
            _, _, feature_dim = arr.shape
        elif arr.ndim == 2:
            # (T, D) -> (1, T, D)
            _, feature_dim = arr.shape
            arr = arr[None, ...]
        else:
            raise ValueError(f"Expect actions of shape (B,T,D) or (T,D), got {arr.shape}")

        if indices is not None:
            idx = np.asarray(indices, dtype=int)
            if idx.shape[0] != target_dim:
                raise ValueError(f"indices length {idx.shape[0]} != {target_dim}")
            out = arr[:, :, idx]
        else:
            if feature_dim < target_dim:
                raise ValueError(f"Action dim {feature_dim} < target {target_dim}")
            out = arr[:, :, :target_dim]

        return np.ascontiguousarray(out)

def model_init(model_path, config):
    logger.info("Load Pi0 model")
    pi0 = Pi0(model_path, config)
    return pi0

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run pi0 om model inference")
    parser.add_argument("--model-path", type=str, default="../model/pi0.om", help="Om file path")
    parser.add_argument("--mean-path", type=str, default=None, help="Action mean.pt file")
    parser.add_argument("--std-path", type=str, default=None, help="Action std.pt file")
    parser.add_argument("--dataset-dir", type=str, default="./data/aloha_sim_transfer_cube_scripted", help="dataset path")
    parser.add_argument("--video-dir", type=str, default="./data/aloha_sim_transfer_cube_scripted/videos/observation.images.top/chunk-000/file-000.mp4", help="dataset video path")
    parser.add_argument("--extract-num", type=int, default=500, help="extract frames from dataset")
    parser.add_argument("--output-dir", type=str, default="./out/action/", help="output file path")
    args = parser.parse_args()

    config = {
        'output_features': {
            'action': PolicyFeature(
                type=FeatureType.ACTION,
                shape=(14,))
        },
        'normalization_mapping': {
            'VISUAL': NormalizationMode.IDENTITY,
            'STATE': NormalizationMode.MEAN_STD,
            'ACTION': NormalizationMode.MEAN_STD,
        },
        'stats': None,
    }

    # 加载action归一化参数. 注意：若模型更新需同步更新该文件或默认参数
    if args.mean_path or args.std_path:
        if not (args.mean_path and args.std_path):
            raise ValueError("Both --mean-path and --std-path must be provided together.")
        if not os.path.exists(args.mean_path):
            raise FileNotFoundError(f"Mean file not found: {args.mean_path}")
        if not os.path.exists(args.std_path):
            raise FileNotFoundError(f"Std file not found: {args.std_path}")

        loaded_mean = torch.load(args.mean_path)
        loaded_std = torch.load(args.std_path)
        config['stats'] = {
            'action': {
                'mean': loaded_mean,
                'std': loaded_std,
            }
        }
    else:
        # action默认归一化参数.
        default_mean = torch.tensor([
            -0.00542, -0.48033, 1.01024, -0.00422, -0.52976, 1.12144, 0.58750,
            0.01955, -0.31375, 0.47024, -0.02314, 0.77220, 0.03748, 0.59625
        ], dtype=torch.float32)
        default_std = torch.tensor([
            0.00370, 0.51984, 0.19778, 0.01629, 0.36045, 0.59959, 0.42409,
            0.11115, 0.49436, 0.44355, 0.14520, 0.29563, 0.22779, 0.38612
        ], dtype=torch.float32)
        config['stats'] = {
            'action': {
                'mean': default_mean,
                'std': default_std,
            }
        }

    model = model_init(args.model_path, config)
    # 数据集中使用的数据组数

    state, dataset_action = extract_state_and_action(args.extract_num, args.dataset_dir)
    image = extract_frames_from_video(args.extract_num, args.video_dir)

    logger.info("dataset image.shape: %s", image.shape)
    lang_tokens, lang_masks = prepare_language(1)
    logger.info("lang_tokens: %s", lang_tokens)
    logger.info("lang_masks: %s", lang_masks)

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(state.shape[0]):
        action = model.select_action(state[i:i+1, :], image[i:i+1, :, :, :], lang_tokens, lang_masks)
        action_filename = f"action_{i:05d}.txt"
        action_filepath = os.path.join(args.output_dir, action_filename)
        np.savetxt(action_filepath, action.cpu().numpy())

if __name__ == "__main__":
    main()
