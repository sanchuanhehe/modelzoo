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

import os, sys, torch, timm, argparse

def main():
    # 解析参数
    parser_args = argparse.ArgumentParser(description='Path', add_help=False)
    parser_args.add_argument('--model_path', required=True, metavar='DIR',
                        help='Pth model path')
    parser_args.add_argument('--save_dir', default="model/", type=str,
                        help='Path to save the ONNX model')
    parser_args.add_argument('--model_name', required=True, type=str,
                        help='vit model name')
    arguments = parser_args.parse_args()               
   
    # 实例化vit模型对象
    vit_model = timm.create_model(arguments.model_name)
    
    # 加载vit模型权重
    vit_model.load_pretrained(arguments.model_path)
    
    # 设置评估模式
    vit_model.eval()
    
    # 获取输入大小
    input_size = int(arguments.model_name[-3:])
    input_tensor = torch.zeros(1, 3, input_size, input_size)

    if not os.path.exists(arguments.save_dir):
        os.makedirs(arguments.save_dir)
    
    # 保存转化后的onnx 模型
    output_path = os.path.join(arguments.save_dir,
                             f"{arguments.model_name}.onnx")
    # 导出模型
    torch.onnx.export(vit_model, input_tensor, output_path, opset_version=11,
                      do_constant_folding=True, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    main()
