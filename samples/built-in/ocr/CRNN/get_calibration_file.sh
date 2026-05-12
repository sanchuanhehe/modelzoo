#!/usr/bin/env bash
set -euo pipefail
repo_dir="CRNN_Chinese_Characters_Rec"
commit="a565687c4076b729d4059593b7570dd388055af4"

if [ ! -d "${repo_dir}/.git" ]; then
    git clone https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec "${repo_dir}"
fi
git -C "${repo_dir}" checkout "${commit}"
cp "${repo_dir}/lib/config/alphabets.py" script/
cp "${repo_dir}/lib/dataset/txt/test.txt" data/
cp "${repo_dir}/lib/dataset/txt/char_std_5990.txt" data/