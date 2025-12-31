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
import torch
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm


def dynamic_import_dependencies():
    """动态导入依赖"""
    return {
        'torch': torch,
        'InceptionResnetV1': InceptionResnetV1
    }


def load_model(deps):
    """加载InceptionResnetV1模型"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deps['InceptionResnetV1'](
        classify=False,
        pretrained='vggface2'
    ).to(device)
    model.eval()
    return model, device


def run_inference(args):
    """执行推理"""
    deps = dynamic_import_dependencies()
    os.makedirs(args.output_bin_dir, exist_ok=True)

    # 加载bin文件列表
    if not os.path.exists(args.file_list_path):
        print(f"错误: 文件列表 {args.file_list_path} 不存在")
        sys.exit(1)

    with open(args.file_list_path, "r") as f:
        bin_file_paths = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]

    if not bin_file_paths:
        print("错误: 未找到有效的预处理bin文件")
        sys.exit(1)

    # 加载模型
    model, device = load_model(deps)

    # 批量推理
    for idx, bin_path in enumerate(tqdm(bin_file_paths, desc="推理进度")):
        file_name = os.path.basename(bin_path)
        base_name = os.path.splitext(file_name)[0]

        try:
            # 加载预处理数据
            input_data = np.fromfile(bin_path, dtype=np.float32).reshape(3, 160, 160)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

            # 执行推理
            with torch.no_grad():
                result = model(input_tensor).cpu().numpy().flatten()

            # 保存结果
            output_bin_path = os.path.join(args.output_bin_dir, f"{base_name}_output.bin")
            result.tofile(output_bin_path)

        except Exception as e:
            print(f"警告: 处理 {bin_path} 失败: {e}，跳过")
            continue

    print(f"推理完成，结果保存至: {args.output_bin_dir}")


def main():
    parser = argparse.ArgumentParser(description='FaceNet推理脚本')
    parser.add_argument('--file_list_path', default='../data/preprocess/bin_file_list.txt',
                        help='预处理生成的bin文件列表路径')
    parser.add_argument('--output_bin_dir', default='../out/result/bin',
                        help='推理结果bin文件输出目录')
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
