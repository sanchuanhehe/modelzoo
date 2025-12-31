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
import json

def calculate_average_psnr(folder_path):
    """
    计算单个文件夹中所有txt文件的PSNR平均值
    :param folder_path: 文件夹路径
    :return: 该文件夹的PSNR平均值（保留4位小数），若无可计算数据则返回None
    """
    psnr_values = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取PSNR值（格式：PNSR: 数值dB）
                psnr_str = content.split("PNSR: ")[1].split("dB")[0].strip()
                psnr = float(psnr_str)
                psnr_values.append(psnr)
                
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错：{str(e)}")
    
    # 计算当前文件夹的平均值
    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        return round(average_psnr, 4)
    else:
        return None

# 使用示例
if __name__ == "__main__":
    # 定义需要处理的所有文件夹名称（可根据实际情况修改）
    target_folders = ["X2", "X3", "X4"]
    # 所有文件夹的共同上级目录（若文件夹在脚本同目录下，可设为当前目录"./"）
    parent_directory = "out/result/txt/"
    '''
    Set5	2	37.53(37.41)
    Set5	3	33.66(33.44)
    Set5	4	31.35(31.05)
    '''
    target = {
        "X2": 37.41,
        "X3": 33.44,
        "X4": 31.05
    }
    result = {}
    # 遍历每个目标文件夹
    for folder_name in target_folders:
        folder_path = os.path.join(parent_directory, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告：文件夹 {folder_path} 不存在，已跳过")
            continue
        
        # 计算当前文件夹的平均PSNR
        avg_psnr = calculate_average_psnr(folder_path)
        result[folder_name] = {
            "perdict": f"{avg_psnr:.2f} dB",
            "base": f"{target[folder_name]} dB",
            "err": f"{abs(target[folder_name] - avg_psnr)*100/target[folder_name]:.2f}%"
        }
        
        # 输出结果
        if avg_psnr is not None:
            print(f" {folder_name} 的PSNR平均值为：{avg_psnr} dB")
    with open(os.path.join(parent_directory, "result.json"), "w") as f:
        f.write(json.dumps(result, indent=4))