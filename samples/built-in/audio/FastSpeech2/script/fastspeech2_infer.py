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
import torch
import onnxruntime as ort
from tqdm import tqdm
from scipy.io import wavfile

INT16_MAX = 32767  # int16音频量化最大值（范围：-32768~32767）
SAMPLING_RATE = 22050  # FastSpeech2/HifiGAN默认采样率


def load_onnx_model(onnx_path):
    """加载ONNX模型"""
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_names = session.get_inputs()
    output_names = session.get_outputs()
    return session, input_names, output_names


def pad_phones_with_sp(phones, target_len, pad_value=357):
    """
    为整数编码的音素numpy数组补充指定填充值至目标长度（不足补值，超过则截断）
    Args:
        phones: 预处理后的音素整数编码一维numpy数组（dtype=int64，如np.array([12,45,357], dtype=np.int64)）
                直接接收np.fromfile读取的结果，无需类型转换
        target_len: 目标长度（即src_lens，如40），必须为正整数
        pad_value: 填充的整数（对应原sp符号，默认357），会自动转为int64类型
    Returns:
        padded_phones: 填充后的固定长度一维numpy数组（dtype=int64）
    Raises:
        ValueError: 当target_len为非正整数、phones非一维numpy数组时抛出
        TypeError: 当phones的dtype不是整数类型时抛出
    """
    # 第一步：严格校验输入类型和维度
    if not isinstance(phones, np.ndarray):
        raise TypeError(f"phones必须是numpy.ndarray类型，当前类型为{type(phones)}")
    if phones.ndim != 1:
        raise ValueError(f"phones必须是一维数组，当前维度为{phones.ndim}")
    if not np.issubdtype(phones.dtype, np.integer):
        raise TypeError(f"phones必须是整数类型数组，当前dtype为{phones.dtype}")

    # 第二步：校验目标长度合法性
    if not isinstance(target_len, int) or target_len <= 0:
        raise ValueError(f"target_len必须为正整数，当前值为{target_len}")

    # 第三步：核心逻辑（截断/填充，全程numpy操作，保留dtype=int64）
    phones_len = len(phones)
    # 统一将填充值转为int64，保证数据类型一致
    pad_value_int64 = np.int64(pad_value)

    if phones_len >= target_len:
        # 长度超过/等于目标值 → 截断到目标长度
        padded_phones = phones[:target_len].copy()  # copy避免视图引用
    else:
        # 长度不足 → 构造填充数组，拼接至原数组尾部
        pad_num = target_len - phones_len
        # 生成和原数组同dtype的填充数组
        pad_array = np.full(pad_num, pad_value_int64, dtype=np.int64)
        padded_phones = np.concatenate([phones, pad_array])

    # 确保最终输出是int64类型（兼容原数据类型）
    padded_phones = padded_phones.astype(np.int64)
    print(f"原始音素序列为        ：{phones.tolist()}")
    print(f"填充或截段后的音素序列为：{padded_phones.tolist()}")
    return padded_phones


def run_inference(args):
    """执行推理并保存梅尔频谱"""
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    session, input_names, output_names = load_onnx_model(args.model_path)

    # 读取预处理文件列表
    with open(args.file_list, "r") as f:
        bin_files = [line.strip() for line in f if line.strip()]

    # 批量推理
    for bin_path in tqdm(bin_files):
        text_id = os.path.splitext(os.path.basename(bin_path))[0]
        try:
            print(f"******开始处理: {text_id}******")
            # 加载输入数据
            text = np.fromfile(bin_path, dtype=np.int64)
            src_lens = np.array([[min(len(text), args.target_len)]], dtype=np.int64)

            # 为整数编码的音素列表补充指定填充值至目标长度（不足补值，超过则截断）
            pad_text = pad_phones_with_sp(text, target_len=args.target_len)

            # 推理
            input_feed = {
                'text': pad_text.reshape(1, -1),
                'src_lens': src_lens
            }
            print(f"len(text)：{len(text)}, len(pad_text)：{len(pad_text)}, src_lens：{src_lens}")
            wave, mel = session.run(output_names=['wave', 'mel'], input_feed=input_feed)

            # 保存梅尔频谱
            mel_path = os.path.join(args.output_dir, f"{text_id}_mel.bin")
            mel.astype(np.float32).tofile(mel_path)
            # 保存hifi生成的波形
            wave_path = os.path.join(args.output_dir, f"{text_id}_wave.bin")
            wave.astype(np.float32).tofile(wave_path)
            # 保存音频文件
            wave = wave.squeeze()
            # ========== 新增：后处理添加tanh，替代开发板的低精度tanh ==========
            wave = np.tanh(wave)
            # ===============================================================
            wave = (wave * INT16_MAX).astype(np.int16)
            wav_save_path = os.path.join(args.output_dir, f"{text_id}.wav")
            wavfile.write(wav_save_path, SAMPLING_RATE, wave)
            print(f"{text_id} 推理完成，梅尔频谱保存至 {mel_path}，波形保存至 {wave_path}，WAV音频文件保存至: {wav_save_path}")
        except Exception as e:
            print(f"推理失败 {text_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FastSpeech2推理脚本")
    parser.add_argument("--model_path", default="../model/fastspeech_hifigan_en.onnx", help="onnx模型文件路径")
    parser.add_argument("--file_list", default="../data/preprocess/text_list.txt", help="预处理生成的文本列表文件（text_list.txt）")
    parser.add_argument("--output_dir", default="../out/result_pc", help="推理结果输出目录")
    parser.add_argument("--target_len", default=40, help="静态模型输入长度（音素）")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
