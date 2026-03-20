"""
ACTWrapper.py

加载 act om 模型并推理
"""

import numpy as np
import subprocess
import os
import re
import struct
import json
import sys
import time
import argparse


class ACT3403Policy:
    def __init__(self, cpp_executable):
        super().__init__()
        self.cpp_executable = cpp_executable
        self.cpp_dir = os.path.dirname(cpp_executable)  # 获取二进制所在目录
        # self.cpp_process = self.model_init()

    def predict(self, batch) -> tuple:
        input_arr = batch
        start_time = time.perf_counter() 
        cpp_outputs = self.run_cpp_and_get_float_output(input_arr)

        if cpp_outputs is not None:
            data = cpp_outputs.get(2)
            if data is not None:
                action = []
                for i in range(0, len(data), 8):
                    action.append(data[i:i+6])
                action = np.array(action, dtype=np.float32)
                action = action.reshape([1, 100, 6])
                return action
        return None

    def run_cpp_and_get_float_output(self, input_arrays):
        # 启动C++进程，通过管道通信
        process = subprocess.Popen(
            self.cpp_executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # 二进制模式通信
            cwd=self.cpp_dir
        )

        try:
            # 发送输入数据
            for i, arr in enumerate(input_arrays):
                data_bytes = arr.tobytes()  # 获取原始二进制数据
                data_size = struct.pack('<I', len(data_bytes))  # 小端模式打包长度
                
                process.stdin.write(data_size)
                process.stdin.write(data_bytes)
            
            # 刷新缓冲区，确保数据发送完成
            process.stdin.flush()

            # 读取C++输出和错误（超时保护：避免无限等待）
            try:
                stdout, stderr = process.communicate(timeout=10)  # 10秒超时
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()  # 获取超时后的部分输出

            # 检查进程是否异常退出
            if process.returncode != 0:
                return None

            # 解析推理时间和输出（保持原有逻辑）
            time_match=re.search(r"INFERENCE_TIME:(\d+\.\d+)", stdout.decode('utf-8'))
            print(time_match)

            pattern = r"FLOAT_OUTPUT_START (\d+) (\d+)\n(.*?)\nFLOAT_OUTPUT_END \1"
            matches = re.findall(pattern, stdout.decode('utf-8'), re.DOTALL)

            float_outputs = {}
            for idx, count, data_str in matches:
                float_data = np.array(list(map(float, data_str.split())), dtype=np.float32)
                float_outputs[int(idx)] = float_data

            return float_outputs

        except BrokenPipeError:
            process.kill()
            return None
        except Exception as e:
            process.kill()
            return None

def read_bin_file(file_path, dtype=np.float32):
    """
    读取bin文件并解析为numpy数组
    Args:
        file_path: bin文件路径（如"data.bin"）
        dtype: 数据类型（如np.float32、np.int64，默认float32）
    Returns:
        numpy数组：解析后的数值数组
    """
    try:
        # 以二进制只读模式打开文件
        with open(file_path, 'rb') as f:
            # 读取所有二进制数据并解析为numpy数组
            # frombuffer：将二进制字节流转为数组，dtype指定数据类型
            data = np.frombuffer(f.read(), dtype=dtype)
        return data
    
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取bin文件失败：{str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, required=True)
    args = parser.parse_args()

    image_list = [item for item in args.image_list.split(';')]
    batch = []
    for bin_path in image_list:
        batch.append(read_bin_file(bin_path))

    om_model_path = "../out/main"
    model = ACT3403Policy(om_model_path)
    action = model.predict(batch)[0]
    action = np.around(action, 4)
    with open('result.txt', 'w', encoding='utf-8') as f:
        for num in action:
            f.write(str(num))
            f.write("\n")
