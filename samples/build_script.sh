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

    cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc -DSOC_VERSION=$DEF
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

    cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc -DSOC_VERSION=$DEF
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
}

parse_arg

if [ $make_exit_status -eq 0 ]; then
    exit 0
else
    exit -1
fi
