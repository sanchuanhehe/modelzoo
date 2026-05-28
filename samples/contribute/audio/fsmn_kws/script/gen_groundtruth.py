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

import os
import argparse

KEYWORD_PREFIX = "20200707_spk57db_storenoise52db_40cm_xiaoyun_sox"

def main():
    parser = argparse.ArgumentParser(description="Generate ground truth file from SCP")
    parser.add_argument("--scp_file", type=str, 
                        default="./iic/speech_charctc_kws_phone-xiaoyun/unittest/example_kws/test_wav.scp",
                        help="SCP file path")
    parser.add_argument("--output_file", type=str, 
                        default="./data/groundtruth.txt",
                        help="Output groundtruth file")
    args = parser.parse_args()
    
    lines = []
    if os.path.exists(args.scp_file):
        with open(args.scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    wav_basename = os.path.basename(parts[1])
                    groundtruth = 1 if filename.startswith(KEYWORD_PREFIX) else 0
                    lines.append(f"{wav_basename} {groundtruth}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    total = len(lines)
    keyword_count = sum(1 for line in lines if line.endswith('1'))
    
    print(f"Generated {args.output_file}")
    print(f"Total files: {total}")
    print(f"Keyword files (label=1): {keyword_count}")
    print(f"Non-keyword files (label=0): {total - keyword_count}")

if __name__ == "__main__":
    main()