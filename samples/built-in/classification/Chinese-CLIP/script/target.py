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

import json
import os

def json_to_txt(json_path, txt_path, default_label=0):
    """
    解析JSON文件中的fileList，保存为txt（格式：文件名 标签值）
    Args:
        json_path: JSON文件路径
        txt_path: 输出txt文件路径
        default_label: 默认标签值（可自定义，如示例中的65/970等）
    """
    # 1. 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 提取fileList中的路径，并处理格式
    file_list = data.get('fileList', [])
    lines = []
    
    for idx, path_item in enumerate(file_list):
        # 路径项是嵌套列表，取第一个元素（如["../xxx.png"] → "../xxx.png"）
        img_path = path_item[0] if isinstance(path_item, list) and len(path_item) > 0 else ""
        if not img_path:
            continue
        
        # 提取文件名（如"../data/xxx.png" → "hooded_skunk_s_000014.png"）
        img_name = os.path.basename(img_path)
        dir_name = os.path.dirname(img_path)  # 获取上级目录
        label = int(os.path.basename(dir_name))  # "002" → 2
        
        lines.append(f"{img_name} {label}")
    
    # 3. 写入txt文件
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)  # 确保目录存在
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"成功解析JSON并保存到txt：{txt_path}")
    print(f"共处理 {len(lines)} 条记录")

# ========== 调用示例 ==========
if __name__ == "__main__":
    # 配置路径
    JSON_PATH = "./data/file_list.json"  # 替换为你的JSON文件路径
    TXT_PATH = "data/target.txt"  # 输出txt路径
    
    # 执行转换（默认标签设为0，可根据需求修改）
    json_to_txt(JSON_PATH, TXT_PATH, default_label=0)