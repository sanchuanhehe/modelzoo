#!/bin/bash

SOC=$1
DEF=$2
DIR=$3
TARGET=$4

#使用指南
function generate_usage()
{
    echo "Usage:  $0 [-option]"
    echo "options:"
    echo " ./build.sh SOC DEF DIR TARGET            - build sample for ss928v100"
}

#解析参数
function parse_arg()
{
    case $SOC in
        "SS928V100")
             case $DEF in
                "SS928V100")
                    /home/build/miniconda3/bin/conda run -n ${SOC}_SVP_NNN /home/build/modelzoo/samples/build_script.sh $SOC $DEF $DIR $TARGET
                    exit_status=$?
                    ;;
                "OPTG")
                    /home/build/miniconda3/bin/conda run -n ${SOC}_NNN /home/build/modelzoo/samples/build_script.sh $SOC $DEF $DIR $TARGET
                    exit_status=$?
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
                    /home/build/modelzoo/samples/build_script.sh $SOC $DEF $DIR $TARGET
                    exit_status=$?
                    ;;
            esac
            ;;
        *)
            generate_usage;
            ;;
    esac
}

#开始时间
start_time=$(date +%s.%N)

#主函数
parse_arg

#结束时间
end_time=$(date +%s.%N)
cost_time=$(echo "$end_time $start_time" | awk '{printf "%.6f", $1 - $2}')
echo "$TARGET takes $cost_time s"

if [ $exit_status -eq 0 ]; then
    echo "######### Build target:$TARGET success #########"
    echo "Finished: SUCCESS"
else
    echo "######### Build target:$TARGET failure ########"
    echo "Finished: FAILURE"
fi
