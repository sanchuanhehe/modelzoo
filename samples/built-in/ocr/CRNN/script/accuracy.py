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
from alphabets import alphabet
import json
from tqdm import tqdm

PERCENT_100 = 100

with open("../data/file_list.json", "r") as f:
    involed_file = [_[0].split("/")[-1] for _ in json.loads(f.read())["fileList"]]

with open("../data/test.txt", "r") as f:
    lines = [_ for _ in f.readlines() if _.strip()]

with open("../data/char_std_5990.txt", "r", encoding="gb18030") as f:
    truth_char = [_.strip() for _ in f.readlines() if _.strip()]


def seque2alphabet(m_str):
    res = "".join([alphabet[int(_) - 1] for _ in m_str.split('_')])
    return res

dict = {}
for _ in tqdm(lines):
    if _.strip() and _.strip().split()[0] in involed_file:
        dict[_.strip().split()[0]] = "_".join([str(alphabet.index(truth_char[int(__)]) + 1) for __ in _.split()[1:]])
file_list = os.listdir("../out/result/txt")
total = len(file_list)
correct = 0
with open("../out/result/fail_detail.txt", "w") as f:
    f.write("")
for _ in file_list:
    with open("../out/result/txt/" + _, "r") as f:
        perdiction = "_".join(f.read().strip().split())
    flag = perdiction == dict[_.replace("_result.txt", ".jpg")]
    if not flag:
        with open("../out/result/fail.txt", "a") as f:
            f.write(f"{_.replace('_result.txt', '.jpg')} perdiction:{seque2alphabet(perdiction)}\n{' '*len(_.replace('_result.txt', '.jpg'))}      truth:{seque2alphabet(dict[_.replace('_result.txt', '.jpg')])}\n")
    correct += flag

ctx = f"acc: {(correct * PERCENT_100 / total):.2f}%\n"
print(ctx)
with open("../out/result/result.txt", "w") as f:
    f.write(ctx)
