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
import nltk
import onnxruntime as ort
import yaml
import torch.fx
import time
from scipy.io import wavfile

# 获取当前脚本（export_onnx.py）的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接deploy所在的目录（FastSpeech2/FastSpeech2/）
deploy_parent_dir = os.path.abspath(os.path.join(script_dir, "../FastSpeech2"))
# 将该目录加入Python搜索路径
if deploy_parent_dir not in sys.path:
    sys.path.insert(0, deploy_parent_dir)

from deploy.utils import convert_to_deploy
from utils.model import get_model, get_vocoder
from synthesize import preprocess_mandarin, preprocess_english

nltk.data.path.append('./nltk_data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INT16_MAX = 32767  # int16音频量化最大值（范围：-32768~32767）
SAMPLING_RATE = 22050  # FastSpeech2/HifiGAN默认采样率


class FastSpeech2AndHifigan(torch.nn.Module):
    def __init__(self, fast_speech2, hifi_gan, speakers=8):
        super().__init__()
        self.fast_speech2 = fast_speech2
        self.hifi_gan = hifi_gan
        self.speakers = speakers

    def forward(self,
                texts,
                src_lens,
                max_src_len=None,
                mel_lens=None,
                max_mel_len=None,
                p_targets=None,
                e_targets=None,
                d_targets=None,
                ):
        mel = self.fast_speech2(
            texts,
            src_lens,
            max_src_len=max_src_len,
            mel_lens=mel_lens,
            max_mel_len=max_mel_len,
            p_targets=p_targets,
            e_targets=e_targets,
            d_targets=d_targets,
            speakers=self.speakers,
        )[1]
        mel = mel.transpose(-2, -1)
        x = self.hifi_gan(mel)
        return x, mel


def combine_model(args, configs, model_config):
    model = get_model(args, configs, device, train=False)
    if args.npu == "SVP_NNN":
        model = convert_to_deploy(model)
    vocoder = get_vocoder(model_config, device)
    model = FastSpeech2AndHifigan(model, vocoder, speakers=args.speaker_id).eval()

    return model


def export_combine_onnx_model(args):
    # 1. 先将相对路径转为绝对路径（避免路径解析错误）
    args.onnx_path = os.path.abspath(args.onnx_path)
    # 2. 检查目录是否存在，不存在则创建（exist_ok=True 避免重复创建报错）
    os.makedirs(args.onnx_path, exist_ok=True)
    print(f"ONNX模型保存目录已确认/创建：{args.onnx_path}")

    # 根据配置选择对应的配置文件
    if args.language == "zh":
        args.preprocess_config = 'config/AISHELL3/preprocess.yaml'
        args.model_config = 'config/AISHELL3/model.yaml'
        args.train_config = 'config/AISHELL3/train.yaml'
        args.restore_step = 600000  # 加载的checkpoint对应的训练步长
    elif args.language == "en":
        args.preprocess_config = 'config/LJSpeech/preprocess.yaml'
        args.model_config = 'config/LJSpeech/model.yaml'
        args.train_config = 'config/LJSpeech/train.yaml'
        args.restore_step = 900000  # 加载的checkpoint对应的训练步长
    else:
        print(f"language {args.language} error, must be zh or en")
        raise ValueError("language must be zh or en")

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    if args.language == "zh":
        texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
    elif args.language == "en":
        texts = np.array([preprocess_english(args.text, preprocess_config)])
    else:
        print(f"language {args.language} error, must be zh or en")
        raise ValueError("language must be zh or en")

    # 加载fastspeech2+hifigan组合模型
    model = combine_model(args, configs, model_config)

    # 文本长度检查 + 截断
    if len(texts[0]) > args.target_len:
        print(f"警告：文本长度 {len(texts[0])} 超过上限 {args.target_len}，已自动截断")
        texts = texts[:, :args.target_len]
    if len(texts[0]) < args.target_len:
        print(f"警告：文本长度 {len(texts[0])} < {args.target_len}，生成的静态ONNX模型的输入长度将小于目标长度")
    texts = torch.from_numpy(texts).to(device)
    text_lens = torch.tensor([[len(texts[0])]]).to(device)

    # ====================== 新增：PTH模型推理并保存结果 ======================
    print("\n=== 开始PTH模型推理 ===")
    with torch.no_grad():
        # PTH模型推理
        pth_wave, pth_mel = model(texts, text_lens)
        # 转numpy格式（和ONNX输出对齐）
        pth_wave = pth_wave.detach().cpu().numpy()
        pth_mel = pth_mel.detach().cpu().numpy()

        # 处理PTH生成的wave（和ONNX后处理逻辑完全一致）
        pth_wav_data = pth_wave[0] if pth_wave.shape[0] == 1 else pth_wave
        print(f"PTH wave dtype: {pth_wav_data.dtype}, range: {np.min(pth_wav_data)} ~ {np.max(pth_wav_data)}")
        pth_wav_data = pth_wav_data.squeeze()
        # ========== 新增：后处理添加tanh，替代开发板的低精度tanh ==========
        pth_wav_data = np.tanh(pth_wav_data)
        # ===============================================================
        if pth_wav_data.dtype == np.float32 or pth_wav_data.dtype == np.float64:
            pth_wav_data = (pth_wav_data * INT16_MAX).astype(np.int16)

        # 构造保存路径（和ONNX同名，后缀加_pth）
        basename = f"fastspeech_hifigan_{args.language}"
        # 保存PTH生成的音频
        pth_wav_save_path = os.path.join(args.onnx_path, f"{basename}_output_pth.wav")
        wavfile.write(pth_wav_save_path, SAMPLING_RATE, pth_wav_data)
        # 保存PTH生成的梅尔频谱
        pth_mel_path = os.path.join(args.onnx_path, f"{basename}_mel_pth.bin")
        pth_mel.astype(np.float32).tofile(pth_mel_path)
    print(f"PTH infer gen WAV: {pth_wav_save_path}, gen mel: {pth_mel_path}")
    print("=== PTH模型推理完成 ===\n")
    # ====================== PTH推理逻辑结束 ======================

    # 导出输入数据和onnx模型
    texts.detach().cpu().numpy().astype(np.int16).tofile(f'{args.onnx_path}/text.bin')
    text_lens.detach().cpu().numpy().astype(np.int16).tofile(f'{args.onnx_path}/src_lens.bin')
    onnx_path = f'{args.onnx_path}/fastspeech_hifigan_{args.language}.onnx'
    torch.onnx.export(model, (texts, text_lens), onnx_path, opset_version=11,
                      input_names=['text', 'src_lens'],
                      output_names=['wave', 'mel'])
    print(f'The onnx is successfully saved in {onnx_path}')

    ################### 使用onnx推理, 生成文本对应的语音 ###################
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    input_feed = {
        'text': texts.detach().cpu().numpy(),
        'src_lens': text_lens.detach().cpu().numpy()
    }

    # ========== 性能测试：循环10次推理 + 统计平均时间 ==========
    # 初始化变量
    total_time = 0.0
    loop_times = 5  # 执行5次
    ort_outputs = None  # 保存最终的推理结果
    # 循环执行推理
    for i in range(loop_times):
        # 记录单次推理开始时间
        start_time = time.time()

        # 核心推理代码
        ort_outputs = ort_session.run(['wave', 'mel'], input_feed)

        # 计算单次耗时并累加
        end_time = time.time()
        single_time = end_time - start_time
        total_time += single_time
        # 打印单次耗时（可选，方便看波动）
        print(f"第{i + 1}次推理耗时: {single_time:.4f} 秒")

    # 计算平均耗时
    avg_time = total_time / loop_times
    # 打印统计结果
    print("=" * 50)
    print(f"总计执行 {loop_times} 次，总耗时: {total_time:.4f} 秒")
    print(f"平均每次推理耗时: {avg_time:.4f} 秒")
    print(f"推理QPS（次/秒）: {1 / avg_time:.2f}")
    print("=" * 50)

    onnx_wave = ort_outputs[0]
    onnx_mel = ort_outputs[1]
    wav_data = onnx_wave[0] if onnx_wave.shape[0] == 1 else onnx_wave
    print(f"ONNX wave dtype: {wav_data.dtype}, range: {np.min(wav_data)} ~ {np.max(wav_data)}")
    wav_data = wav_data.squeeze()
    # ========== 新增：后处理添加tanh，替代开发板的低精度tanh ==========
    wav_data = np.tanh(wav_data)
    # ===============================================================
    if wav_data.dtype == np.float32 or wav_data.dtype == np.float64:
        wav_data = (wav_data * INT16_MAX).astype(np.int16)

    basename = os.path.splitext(os.path.basename(onnx_path))[0]
    # 保存生成的音频
    wav_save_path = os.path.join(os.path.dirname(onnx_path), f"{basename}_output.wav")
    wavfile.write(wav_save_path, SAMPLING_RATE, wav_data)
    # 保存梅尔频谱
    mel_path = os.path.join(os.path.dirname(onnx_path), f"{basename}_mel.bin")
    onnx_mel.astype(np.float32).tofile(mel_path)
    print(f"ONNX infer gen WAV: {wav_save_path}, gen mel: {mel_path}")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_len", type=int, default=40, help="模型静态输入长度（音素）")
    parser.add_argument("--language", default="en", choices=["zh", "en"], help="模型支持的语言")
    parser.add_argument("--npu", default="SVP_NNN", choices=["SVP_NNN", "NNN"], help="NPU类型")
    # parser.add_argument("--text", default="湖南一餐馆推出石槽火锅生意火爆，老板称红薯叶等之前喂猪的食材现在人也在吃，这样的创新火锅你愿意尝试吗？")
    parser.add_argument("--text",
                        default="Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model")
    parser.add_argument("--onnx_path", default="../model", help="ONNX模型导出路径")
    # 使用默认配置即可
    parser.add_argument("--mode", type=str, default="single", help="单条推理")
    parser.add_argument("--pitch_control", type=float, default=1.0, help="音调控制参数")
    parser.add_argument("--energy_control", type=float, default=1.0, help="能量控制参数")
    parser.add_argument("--duration_control", type=float, default=1.0, help="时长控制参数")
    parser.add_argument("--speaker_id", type=int, default=8, help="说话人ID")

    args = parser.parse_args()
    args.source = None
    return args


if __name__ == '__main__':
    args = parser_args()
    export_combine_onnx_model(args)
