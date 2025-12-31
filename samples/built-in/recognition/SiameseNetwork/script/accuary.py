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
import re

DISSIMILARITY_THRESHOLD = 0.9

def parse_txt(file_txt):
    parts = re.split(r'\[Separate line\]|\[Result line\]', file_txt.strip())
    
    parts = [part.strip() for part in parts if part.strip()]
    
    result_info = {}
    result_part = parts[-1] if parts else ''
    # 匹配 dismillary: 数值 , label: 数值
    result_match = re.search(r'Dismillary: (\d+\.\d+) , label: (\d+)', result_part)
    if result_match:
        result_info['dismillary'] = float(result_match.group(1))
        result_info['label'] = int(result_match.group(2))
    
    return (result_info["dismillary"] > DISSIMILARITY_THRESHOLD and result_info["label"] == 1) or (result_info["dismillary"] <= DISSIMILARITY_THRESHOLD and result_info["label"] == 0)

if __name__ == "__main__":
    txt_dir = "out/result/txt"
    file_num  = 0
    correct_num = 0
    for root, dirs, files in os.walk(txt_dir):
        file_num = len(files)
        for file in files:
            with open(os.path.join(root, file), "r") as f:
                res = parse_txt(f.read())
                correct_num += 1 if res else 0
    print_result = f"acc: {(correct_num / file_num) * 100:.3f}%"
    with open("out/result/result.txt", "w") as f:
        f.write(print_result)
    print(print_result)