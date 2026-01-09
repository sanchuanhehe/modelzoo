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

import json
import sys
import os


def gen_file_list(val_file_path):
    """
    从val.txt生成file_list.json文件
    :param val_file_path: val.txt的路径（命令行传入）
    """
    # 1. 校验输入文件是否存在
    if not os.path.exists(val_file_path):
        print(f"错误：文件 {val_file_path} 不存在！")
        sys.exit(1)

    # 2. 读取val.txt内容并处理
    file_list = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去除换行符/空格
            if not line:  # 跳过空行
                continue

            # 按|分割行内容（分割为4部分以内，避免内容含|导致异常）
            parts = line.split("|", 3)
            if len(parts) < 3:
                print(f"警告：行内容格式错误，跳过该行：{line}")
                continue

            # 重构行内容：第三部分替换为xxx
            new_line = f"{parts[0]}|{parts[1]}|xxx|{parts[3] if len(parts) >= 4 else ''}"
            # 按要求封装为列表嵌套格式
            file_list.append([new_line])

    # 3. 生成最终的JSON数据结构
    result = {
        "fileList": file_list
    }

    # 4. 写入file_list.json（路径：../data/file_list.json）
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)  # 确保data目录存在
    output_path = os.path.join(output_dir, "file_list.json")

    with open(output_path, "w", encoding="utf-8") as f:
        # indent=4保证JSON格式化，便于查看
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"成功生成文件：{output_path}")
    print(f"共处理 {len(file_list)} 条数据")


if __name__ == "__main__":
    # 校验命令行参数
    if len(sys.argv) != 2:
        print("使用方法：python3 fastspeech2_gen_file_list.py <val.txt路径>")
        print("示例：python3 fastspeech2_gen_file_list.py ../FastSpeech2/preprocessed_data/LJSpeech/val.txt")
        sys.exit(1)

    # 执行生成逻辑
    val_file_path = sys.argv[1]
    gen_file_list(val_file_path)
