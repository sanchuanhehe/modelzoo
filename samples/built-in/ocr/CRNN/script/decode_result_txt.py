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

from pathlib import Path

from alphabets import alphabet


INPUT_DIR = Path("./out/result/txt")
OUTPUT_DIR = Path("./out/result/decoded_txt")


def decode_tokens(content):
    chars = []
    for token in content.split():
        index = int(token)
        if index < 1 or index > len(alphabet):
            raise ValueError(f"index {index} out of range 1..{len(alphabet)}")
        chars.append(alphabet[index - 1])
    return "".join(chars)


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"input dir not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for input_file in sorted(INPUT_DIR.glob("*.txt")):
        decoded = decode_tokens(input_file.read_text(encoding="utf-8"))
        output_file = OUTPUT_DIR / input_file.name
        output_file.write_text(decoded + "\n", encoding="utf-8")
        count += 1

    print(f"decoded {count} files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
