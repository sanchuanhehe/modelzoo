#!/bin/bash

SOC=$1
DEF=$2
DIR=$3
TARGET=$4

#ss928v100_svp_nnn demo编译
function ss928v100_svp_nnn_build()
{
    cd $DIR
    rm -rf build
    rm -rf out
    mkdir build
    cd build

    echo "Conda env: $CONDA_DEFAULT_ENV"
    source /home/build/Ascend/ascend-toolkit/svp_latest/x86_64-linux/script/setenv.sh

    export NPU_INCLUDE_PATH=/home/build/Ascend/ascend-toolkit/svp_latest/acllib/include/acl
    export NPU_LIB_PATH=/home/build/Ascend/ascend-toolkit/svp_latest/acllib/lib64/stub

    if [ -f "../../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../../common/cmake/toolchain_aarch64_linux.cmake"
    elif [ -f "../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../common/cmake/toolchain_aarch64_linux.cmake"
    else
        echo "ERROR: 未找到工具链文件！"
    fi

    # 第二步：执行cmake命令（变量加双引号，避免路径含特殊字符）
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DSOC_VERSION=$DEF

    make
    make_exit_status=$?
}

#ss928v100_nnn demo编译
function ss928v100_nnn_build()
{
    cd ${DIR}
    rm -rf build
    rm -rf out
    mkdir build
    cd build

    echo "Conda env: $CONDA_DEFAULT_ENV"
    export DDK_PATH=/home/build/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub

    export NPU_INCLUDE_PATH=/home/build/Ascend/ascend-toolkit/latest/acllib/include/acl
    export NPU_LIB_PATH=/home/build/Ascend/ascend-toolkit/latest/acllib/lib64/stub

    if [ -f "../../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../../common/cmake/toolchain_aarch64_linux.cmake"
    elif [ -f "../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../common/cmake/toolchain_aarch64_linux.cmake"
    else
        echo "ERROR: 未找到工具链文件！"
    fi

    # 第二步：执行cmake命令（变量加双引号，避免路径含特殊字符）
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DSOC_VERSION=$DEF
    make
    make_exit_status=$?
}

#hi3591p demo编译
function hi3591p_build()
{
    cd ${DIR}
    rm -rf build
    rm -rf out
    mkdir build
    cd build

    echo "Conda env: $CONDA_DEFAULT_ENV"
    export DDK_PATH=/home/build/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/acllib/lib64/stub

    export NPU_INCLUDE_PATH=/home/build/Ascend/ascend-toolkit/latest/acllib/include/acl
    export NPU_LIB_PATH=/home/build/Ascend/ascend-toolkit/latest/acllib/lib64/stub

    if [ -f "../../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../../common/cmake/toolchain_aarch64_linux.cmake"
    elif [ -f "../../../common/cmake/toolchain_aarch64_linux.cmake" ]; then
        TOOLCHAIN_FILE="../../../common/cmake/toolchain_aarch64_linux.cmake"
    else
        echo "ERROR: 未找到工具链文件！"
    fi

    # 第二步：执行cmake命令（变量加双引号，避免路径含特殊字符）
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" -DSOC_VERSION=$DEF
    make
    make_exit_status=$?
}

#使用指南
function generate_usage()
{
    echo "Usage:  $0 [-option]"
    echo "options:"
    echo "./build_command.sh SOC DEF DIR TARGET            - build sample for ss928v100"
}

#解析参数
function parse_arg()
{
   printf "SOC: $SOC\n"
    printf "DEF: $DEF\n"
    printf "DIR: $DIR\n"
    printf "TARGET: $TARGET\n"
    case $SOC in
        "SS928V100")
             case $DEF in
                "SS928V100")
                    ss928v100_svp_nnn_build
                    ;;
                "OPTG")
                    ss928v100_nnn_build
                    ;;
            esac
            ;;
        *)
            generate_usage;
            ;;
    esac
    case $SOC in
        "Hi3591P")
             case $DEF in
                "Hi3591P")
                    hi3591p_build
                    ;;
            esac
            ;;
        *)
            generate_usage;
            ;;
    esac
}

parse_arg

if [ $make_exit_status -eq 0 ]; then
    exit 0
else
    exit -1
fi
