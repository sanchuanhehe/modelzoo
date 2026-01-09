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
import re
import argparse
import numpy as np
from string import punctuation
import yaml

from g2p_en import G2p
from tqdm import tqdm
from pypinyin import pinyin, Style

# -------------------------- 添加Fastspeech2目录到搜索路径 --------------------------
# 1. 获取当前脚本所在目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接Fastspeech2的绝对路径（上级目录下的Fastspeech2）
fastspeech2_dir = os.path.abspath(os.path.join(current_script_dir, "../FastSpeech2"))
# 3. 将路径添加到Python搜索路径（避免重复添加）
if fastspeech2_dir not in sys.path:
    sys.path.insert(0, fastspeech2_dir)

from text import text_to_sequence
from synthesize import read_lexicon

g2p = G2p()
lexicon = None


def preprocess_english(text, lexicon):
    """预处理英文文本，转换为音素序列"""
    text = text.rstrip(punctuation)
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    return phones


def preprocess_mandarin(text, lexicon):
    """预处理中文文本，转换为音素序列"""
    pinyins = [
        p[0] for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    phones = []
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    phones = "{" + " ".join(phones) + "}"
    return phones


def text_to_bin(text, preprocess_config):
    """将文本转换为模型输入的二进制序列"""
    language = preprocess_config["preprocessing"]["text"]["language"]

    global lexicon
    if lexicon is None:
        lexicon = read_lexicon(os.path.join("../FastSpeech2", preprocess_config["path"]["lexicon_path"]))

    # 生成音素序列
    if language == "en":
        phones = preprocess_english(text, lexicon)
    elif language == "zh":
        phones = preprocess_mandarin(text, lexicon)
    else:
        raise ValueError(f"不支持的语言: {language}")

    # 转换为模型输入序列
    sequence = text_to_sequence(
        phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    )
    print(f"原始文本：{text} \n 转换为音素：{phones} \n 转换为int：{sequence}")
    return np.array(sequence, dtype=np.int64)


def process_texts(args, preprocess_config):
    """处理输入文本文件，生成二进制特征文件"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    bin_list_path = os.path.join(output_dir, "../text_list.txt")

    with open(args.input_file, "r", encoding="utf-8") as f_in, \
            open(bin_list_path, "w", encoding="utf-8") as f_list:

        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
            # 解析ID和文本（格式：ID|文本）
            parts = line.split("|", 3)
            if len(parts) != 4:
                print(f"跳过无效行: {line}")
                continue
            text_id, text = parts[0], parts[3]
            print(f"******开始处理: {text_id}******")

            # 生成二进制文件
            bin_path = os.path.join(output_dir, f"{text_id}.bin")
            try:
                sequence = text_to_bin(text, preprocess_config)
                print(f"len(sequence) is {len(sequence)}")
                sequence.tofile(bin_path)
                f_list.write(f"{os.path.abspath(bin_path)}\n")
                print(f"处理完成: {text_id}")
            except Exception as e:
                print(f"处理失败 {text_id}: {e}")

    print(f"预处理完成，bin保存至 {output_dir}，文件列表保存至: {bin_list_path}")


def main():
    parser = argparse.ArgumentParser(description="FastSpeech2文本预处理脚本")
    parser.add_argument("--input_file", default="../FastSpeech2/preprocessed_data/LJSpeech/val.txt",
                        help="输入文本文件（格式：ID|x|x|文本）")
    parser.add_argument("--output_dir", default="../data/preprocess/bin", help="预处理结果输出目录")
    parser.add_argument("--language", default="en", choices=["zh", "en"], help="模型支持的语言")
    args = parser.parse_args()

    # 加载配置
    if args.language == "zh":
        preprocess_config = yaml.load(open("../FastSpeech2/config/AISHELL3/preprocess.yaml", "r"),
                                      Loader=yaml.FullLoader)
    elif args.language == "en":
        preprocess_config = yaml.load(open("../FastSpeech2/config/LJSpeech/preprocess.yaml", "r"),
                                      Loader=yaml.FullLoader)
    else:
        print(f"language {args.language} error, must be zh or en")
        raise ValueError("language must be zh or en")

    # 处理文本
    process_texts(args, preprocess_config)


if __name__ == "__main__":
    main()
