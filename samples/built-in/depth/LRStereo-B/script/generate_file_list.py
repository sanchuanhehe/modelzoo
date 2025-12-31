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
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    try:
        file_dir = sys.argv[1] # 输入文件目录
    except IndexError:
        print("empty input")
        exit(1)

    if not (os.path.exists(file_dir)):
        print("dir does not exist. path: {}".format(file_dir))
        exit(1)

    file_name_list_left = sorted([os.path.join(file_dir, "training/image_2", f) 
                                        for f in os.listdir(os.path.join(file_dir, "training/image_2")) 
                                        if f.endswith("_10.png")])
    file_name_list_right = sorted([os.path.join(file_dir, "training/image_3", f) 
                                         for f in os.listdir(os.path.join(file_dir, "training/image_3")) 
                                         if f.endswith("_10.png")])
    file_path_list = []
    for file_name in file_name_list_left:
        name = Path(file_name).name
        subfile_left = "../" + os.path.join(file_dir + "/training/image_2/", name)
        subfile_right = "../" + os.path.join(file_dir + "/training/image_3/", name)
        file_path_list.append([subfile_left, subfile_right])
            

    # 构建JSON数据结构
    json_data = {
        "fileList": file_path_list
    }

    # 将数据写入JSON文件
    with open("./data/file_list.json", "w") as f:
        json.dump(json_data, f, indent=4)

    print("JSON文件生成完成！共包含{}个文件路径。".format(len(file_path_list)))
