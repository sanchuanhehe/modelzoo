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
import sys
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 获取脚本所在目录（【B目录】）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接utils所在的【A目录】路径（script的上层目录下的FastSpeech2/）
utils_dir = os.path.abspath(os.path.join(script_dir, "../FastSpeech2"))
# 把【A目录】加入Python搜索路径（最前面，优先查找）
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

from dataset import Dataset
from utils.model import get_model
from utils.tools import to_device
from utils.tools import get_mask_from_lengths

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastSpeech2MelLoss(nn.Module):
    """仅计算FastSpeech2的PostNet Mel Loss"""

    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss()  # 仅保留需要的L1Loss

    def forward(self, batch, postnet_mel_predictions, onnx_mel_len):
        # 仅提取计算Mel Loss必需的变量
        # 1. 从batch中获取目标Mel
        mel_targets = batch[6]  # 直接取inputs[6:]的第一个元素（mel_targets）
        mel_lens = torch.min(batch[7], torch.tensor(onnx_mel_len))
        max_mel_len = batch[8]
        mel_masks = get_mask_from_lengths(mel_lens, min(max_mel_len, onnx_mel_len))

        # 掩码处理（仅保留有效Mel部分）
        mel_masks = ~mel_masks  # 反转掩码：True表示有效区域
        # 裁剪目标Mel到掩码长度（避免维度不匹配）
        mel_targets = mel_targets[:, :mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        # 裁剪推理Mel到掩码长度（避免维度不匹配）
        postnet_mel_predictions = postnet_mel_predictions[:, :mel_masks.shape[1], :]

        # 应用掩码，只计算有效区域的Loss
        postnet_mel_pred = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_target = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # 计算PostNet Mel Loss
        postnet_mel_loss = self.mae_loss(postnet_mel_pred, mel_target)
        return postnet_mel_loss


def evaluate_mel_loss(model, args, train_config, dataset):
    """仅评估Mel PostNet Loss的核心函数"""
    # 初始化Loss函数
    mel_loss_fn = FastSpeech2MelLoss().to(device)

    # 数据加载（仅保留必要配置）
    # batch_size = train_config["optimizer"]["batch_size"]
    batch_size = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # 累计Loss
    total_mel_loss = 0.0
    total_samples = 0

    # 遍历数据集评估
    for batchs in tqdm(loader, desc="Evaluating Mel Loss"):
        for batch in batchs:
            text_id = batch[0][0]
            mel_bin_file_path = os.path.join(args.output_dir, f"{text_id}_mel.bin")
            if not os.path.exists(mel_bin_file_path):
                print(f"警告: mel文件 {mel_bin_file_path} 不存在，跳过")
                continue
            # 加载二进制文件
            postnet_mel_onnx_pred = np.fromfile(mel_bin_file_path, dtype=np.float32)
            postnet_mel_onnx_pred = postnet_mel_onnx_pred.reshape(1, 80, -1).transpose(0, 2, 1)

            batch = to_device(batch, device)
            batch_size_current = len(batch[0])

            with torch.no_grad():  # 禁用梯度，节省显存
                if model is not None:
                    # 模型前向推理（仅传入必要参数）
                    model_output = model(*(batch[2:]))
                    # 从模型输出中获取预测的PostNet Mel
                    postnet_mel_pred = model_output[1]

                    # 可以保留源PTH模型的预处理和推理结果方便对比
                    debug = False
                    if debug:
                        sequence = batch[3].detach().cpu().numpy().astype(np.int64)
                        sequence = np.squeeze(sequence)
                        bin_path = os.path.join("../data/preprocess/bin", f"{text_id}.bin")
                        sequence.tofile(bin_path)

                        postnet_mel_save = postnet_mel_pred.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 1)
                        postnet_mel_save[:, :, :args.onnx_mel_len].tofile(mel_bin_file_path)
                else:
                    postnet_mel_pred = torch.tensor(postnet_mel_onnx_pred)
                # 计算Loss
                mel_loss = mel_loss_fn(batch, postnet_mel_pred, args.onnx_mel_len)

                # 累计Loss（乘以当前batch样本数）
                total_mel_loss += mel_loss.item() * batch_size_current
                total_samples += batch_size_current

    # 计算平均Loss
    avg_mel_loss = total_mel_loss / total_samples
    # 生成评估结果信息
    message = f"Validation Step {args.restore_step}, Mel PostNet Loss: {avg_mel_loss:.4f}"
    return message


if __name__ == "__main__":
    # 命令行参数（仅保留必要参数）
    parser = argparse.ArgumentParser(description="Evaluate FastSpeech2 Mel PostNet Loss")
    parser.add_argument("--restore_step", type=int, default=900000, help="恢复的训练步数")
    parser.add_argument("-p", "--preprocess_config", type=str, default="config/LJSpeech/preprocess.yaml",
                        help="预处理配置文件路径")
    parser.add_argument("-t", "--train_config", type=str, default="config/LJSpeech/train.yaml", help="训练配置文件路径")
    parser.add_argument("-m", "--model_config", type=str, default="config/LJSpeech/model.yaml", help="模型配置文件路径")
    parser.add_argument("--onnx_mel_len", type=int, default=320, help="静态onnx模型输出的梅尔频谱的长度(pytorch模型是动态变化的)")
    parser.add_argument("--output_dir", default="../out/result/bin", help="推理结果输出目录")
    args = parser.parse_args()

    # 加载配置文件（仅保留必要的）
    with open(args.preprocess_config, "r") as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.train_config, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    # 加载数据集（仅验证集）
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )

    # 加载模型（评估模式）
    configs = (preprocess_config, model_config, train_config)
    model = get_model(args, configs, device, train=False).to(device)

    # 评估Mel Loss
    eval_pth_model = False
    if eval_pth_model:
        eval_message = evaluate_mel_loss(model, args, train_config, val_dataset)
        print(f"源pth模型是动态输入输出，损失最小，onnx模型为了适配开发板是静态的输入输出，损失会变大")
        print(f"pytorch model eval: {eval_message}")
    eval_message = evaluate_mel_loss(None, args, train_config, val_dataset)
    print(f"your model eval: {eval_message}")
