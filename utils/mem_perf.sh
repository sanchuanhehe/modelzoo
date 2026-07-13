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

# 默认参数
MODEL_DIR="../"
OM_NAME="det-dpico.om"
INPUT_FILE="file_list_1.json"
DURATION=30
INTERVAL_US=1000
OUTPUT=""
EXECUTABLE="main"
CUSTOM_CMD=""

# 参数解析
while [ $# -gt 0 ]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --om_name)
            OM_NAME="$2"
            shift 2
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --interval)
            INTERVAL_US="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --executable)
            EXECUTABLE="$2"
            shift 2
            ;;
        --custom_cmd)
            CUSTOM_CMD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_dir       Model directory path (required)"
            echo "  --om_name         OM model filename (required)"
            echo "  --input           Input file list (default: file_list_1.json)"
            echo "  --duration        Collection duration in seconds (default: 30)"
            echo "  --interval        Collection interval in microseconds (default: 10000)"
            echo "  --output          Output log filename (default: mem_timestamp.log)"
            echo "  --executable      Executable name (default: main)"
            echo "  --custom_cmd      Custom inference command (overrides default)"
            echo "  --bspddrs-path    DDR bandwidth tool path (required, from environments.json)"
            echo "  --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$MODEL_DIR" ]; then
    echo "Error: --model_dir is required"
    exit 1
fi

if [ -z "$OM_NAME" ]; then
    echo "Error: --om_name is required"
    exit 1
fi

# 生成输出文件名
if [ -z "$OUTPUT" ]; then
    OUTPUT="mem_$(date +%Y_%m_%d_%H_%M).log"
fi

# 清空日志文件
> "${OUTPUT}"

echo "=========================================="
echo "Memory Peak Collection"
echo "=========================================="
echo "Model Dir: ${MODEL_DIR}"
echo "OM Model: ${OM_NAME}"
echo "Input File: ${INPUT_FILE}"
echo "Duration: ${DURATION}s"
echo "Interval: ${INTERVAL_US}us"
echo "Output: ${OUTPUT}"
echo "=========================================="
echo ""
echo "Start logging for ${DURATION}s..."

START_EPOCH=$(date +%s)
CNT=0

# 主循环
while :; do
    # 第100次循环时启动推理程序（后台运行）
    if [ "$CNT" -eq 100 ]; then
        echo "Starting inference at count ${CNT}..."
        cd "${MODEL_DIR}/out"
        if [ -f "./${EXECUTABLE}" ]; then
            # Use custom command if provided, otherwise use default
            if [ -n "$CUSTOM_CMD" ]; then
                echo "Using custom command: ${CUSTOM_CMD}"
                # Use sh -c to run custom command and get correct PID
                sh -c "${CUSTOM_CMD}" >> "${OUTPUT}" 2>&1 &
            else
                # Run inference and capture output (FPS info)
                ./${EXECUTABLE} --model ../model/${OM_NAME} --input ../data/${INPUT_FILE} >> "${OUTPUT}" 2>&1 &
            fi
            PID=$!
            echo "Inference PID: ${PID}"
        else
            echo "Error: Executable not found: ${MODEL_DIR}/out/${EXECUTABLE}"
        fi
        cd - > /dev/null
    fi
    
    # 时间检查
    NOW_EPOCH=$(date +%s)
    ELAPSED=$((NOW_EPOCH - START_EPOCH))
    
    if [ "$ELAPSED" -ge "$DURATION" ]; then
        echo "Duration reached, stopping..."
        break
    fi
    
    # 采集内存数据
    # MMZ内存（media-mem）
    grep used /proc/umap/media-mem >> "${OUTPUT}" 2>/dev/null
    
    # OS内存（MemAvailable）
    grep MemAvailable /proc/meminfo >> "${OUTPUT}" 2>/dev/null
    
    # 等待
    usleep ${INTERVAL_US}
    
    CNT=$((CNT + 1))
done

# 结束统计
# 等待推理进程结束（添加超时避免慢模型如VLM无限等待）
if [ -n "$PID" ]; then
    echo "Waiting for inference process (PID: ${PID})..."
    
    # 等待最多30秒
    WAIT_COUNT=0
    MAX_WAIT=30
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if ! kill -0 $PID 2>/dev/null; then
            # 进程已结束
            break
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    # 超时后强制kill
    if kill -0 $PID 2>/dev/null; then
        echo "Inference process still running after ${MAX_WAIT}s, killing..."
        kill -9 $PID 2>/dev/null
        wait $PID 2>/dev/null
        echo "Inference process killed"
    else
        echo "Inference completed"
    fi
fi

# Stop bspddrs
if [ -n "$BSPDDRS_PID" ]; then
    kill $BSPDDRS_PID 2>/dev/null
    wait $BSPDDRS_PID 2>/dev/null
    echo "Stopped bspddrs (PID: ${BSPDDRS_PID})"
    
    # Extract max DDR percentage
    if [ -f "${BSPDDRS_LOG}" ]; then
        MAX_DDR=$(grep -oP 'ddrc\d+\[\d+\.?\d*%\]' "${BSPDDRS_LOG}" | grep -oP '\d+\.?\d*' | sort -rn | head -1)
        echo "Max DDR Bandwidth: ${MAX_DDR}%"
        echo "DDR Bandwidth Log: ${BSPDDRS_LOG}"
    fi
fi

NOW_EPOCH=$(date +%s)
ELAPSED=$((NOW_EPOCH - START_EPOCH))

echo ""
echo "=========================================="
echo "Collection Finished"
echo "=========================================="
echo "Total Time: ${ELAPSED}s"
echo "Total Samples: ${CNT}"
echo "Avg Interval: $((ELAPSED * 1000 / CNT))ms"
echo "Data saved to: ${OUTPUT}"
if [ -n "$MAX_DDR" ]; then
    echo "Max DDR Bandwidth: ${MAX_DDR}%"
fi
echo ""
echo "Next step: Run on PC"
echo "  python test/main_flow.py --config <model_config> --platform <platform> --mode memory"
echo "=========================================="

exit 0