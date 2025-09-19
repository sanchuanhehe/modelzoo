#!/bin/bash
#
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


# 默认值
output="../out/result"
loop=1

function setenv()
{
    export DDK_PATH=~/Ascend/ascend-toolkit/svp_latest/x86_64-linux
    export NPU_HOST_LIB=~/Ascend/ascend-toolkit/svp_latest/x86_64-linux/toolkit/tools/sim/
    export LD_LIBRARY_PATH=~/Ascend/ascend-toolkit/svp_latest/acllib/lib64/stub
	source ~/Ascend/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh
	echo "set env success."
}

function execute()
{
    echo "execute start."
    echo $model
    setenv
    mkdir build
    cd build
    cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc
    echo "cmake success."
    make
    echo "make success."
    ssh_contect
}

function ssh_contect()
{
    # 加载配置文件
    source ../ssh.cfg  # 或 . ./ssh.cfg
    # 使用变量
    user=$USER
    ip=$BOARD_IP
    port=$PORT
    echo "${user}@${ip}"
    ssh "${user}@${ip}" << EOF
    # 远程执行的命令
    echo "remote task start...."
    cd /mnt/modelzoo/samples/built-in/classification/Squeezenet1_1/out
    echo "model:::::"
    echo ${model}
    echo ${input}
    ./main --acl ../src/acl.json --model ${model} --input ${input} --loop ${loop}
    echo "remote task end...."
    exit
EOF
}



options=$(getopt -o m:i:o::l: --long model:,input:,output::,loop: -n "$0" -- "$@")
eval set -- "$options"

while true; do
    case "$1" in
        -m|--model) 
            echo "model: $2 "
            model="$2"; shift 2 ;;
        -i|--input)
            echo "input: $2 "
            input="$2"; shift 2 ;;
        -o|--output)
            echo "output: $2 "
            output="$2"; shift 2 ;;
        -l|--loop)
            echo "loop: $2 "
            loop="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "内部错误"; exit 1 ;;
    esac
done

execute
