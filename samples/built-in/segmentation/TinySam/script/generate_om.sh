#!/bin/bash

PIP_SHOW_OUTPUT=$(pip show mindcmd 2>/dev/null)
PIP_EXIT_CODE=$?

if [ $PIP_EXIT_CODE -ne 0 ]; then
    echo "错误：未检测到mindcmd包，请先通过pip安装该包！" >&2
    exit 1
fi

MINDCMD_LOCATION=$(echo "$PIP_SHOW_OUTPUT" | awk '/^Location: / {print $2}')

INI_FILE="${MINDCMD_LOCATION}/mindcmd/mindcmd.ini"

if [ ! -f "$INI_FILE" ]; then
    echo "错误：文件 $INI_FILE 不存在！"
    exit 1
fi

sed -i 's/^TARGET_VERSION=.*/TARGET_VERSION=SS928V100/' "$INI_FILE"
echo "将TARGET_VERSION的值修改为SS928V100 $INI_FILE"

if ! grep -q "^n_loop_enable=1" "$INI_FILE"; then
    # 默认追加到文件末尾
    echo "" >> "$INI_FILE"
    echo "n_loop_enable=1" >> "$INI_FILE"
    echo "添加 n_loop_enable=1"
fi


ATC_PATH=$(which atc 2>/dev/null)

if [ -z "$ATC_PATH" ]; then
    echo "错误：未检测到atc工具，请设置atc运行环境！" >&2
    exit 1
fi

echo "检测到atc工具路径：$ATC_PATH"

ASCEND_TOOLKIT_DIR=$(dirname "$(dirname "$(dirname "$ATC_PATH")")")

sed -i "s#^CANN_INSTALL_PATH=.*#CANN_INSTALL_PATH=${ASCEND_TOOLKIT_DIR}#" "$INI_FILE"
echo "将CANN_INSTALL_PATH的值修改为$ASCEND_TOOLKIT_DIR"

mindcmd oneclick pytorch -m script.test_box.image_encoder --input_shape 1,3,448,448 -w ./script/tinysam_box_results/quant_model/image_encoder/image_encoder_quantized.pt --quant_config ./script/tinysam_box_results/quant_model/image_encoder/image_encoder_config.json --realquant
mindcmd oneclick pytorch -m script.test_box.prompt_encoder --input_shape "1,2;1,2" -w ./script/tinysam_box_results/quant_model/prompt_encoder/prompt_encoder_quantized.pt --quant_config ./script/tinysam_box_results/quant_model/prompt_encoder/prompt_encoder_config.json --realquant
mindcmd oneclick pytorch -m script.test_box.mask_decoder --input_shape "1,256,28,28;1,2,256" -w ./script/tinysam_box_results/quant_model/mask_decoder/mask_decoder_quantized.pt --quant_config ./script/tinysam_box_results/quant_model/mask_decoder/mask_decoder_config.json --realquant

TARGET_FILES=(
    "mask_decoder_deploy_model_release.om"
    "prompt_encoder_deploy_model_release.om"
    "image_encoder_deploy_model_release.om"
)

for file_name in "${TARGET_FILES[@]}"; do
    file_path=$(find ~/MindCmd-WorkSpace -type f -name "${file_name}" -print -quit)

    if [ -n "${file_path}" ]; then
        cp -f "${file_path}" ./model
        echo "成功复制：${file_name}"
    else
        echo "警告：未找到文件 ${file_name}"
    fi
done
