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
import itertools

if __name__ == "__main__":
    datasets_dir = r"datasets/faces"
    img_list = []
    for root, dirs, files in os.walk(os.path.join(datasets_dir, "testing")):
        for file in files:
            img_list.append(os.path.join(root, file).replace("\\", "/"))
    img_pair = list(itertools.combinations(img_list, 2))

    file_path_list = []

    for file_name in img_pair:
        file_path1 = "../" + os.path.join(
            file_name[0]
        )  # out文件夹相对于图片相对地址

        file_path2 = "../" + os.path.join(
            file_name[1]
        )  # out文件夹相对于图片相对地址

        file_path_list.append([file_path1, file_path2])

    # 构建JSON数据结构
    json_data = {"fileList": file_path_list}

    # 将数据写入JSON文件
    with open("data/file_list.json", "w") as f:
        json.dump(json_data, f, indent=4)

    print("JSON文件生成完成！共包含{}个文件路径。".format(len(file_path_list)))
