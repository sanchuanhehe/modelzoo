/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastspeech2_postprocess.h"
#include "fastspeech2_preprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace Infer {
namespace FastSpeech2NS {
    using namespace std;

    // 补充WAV文件相关常量（对齐Python代码）
    const int WAV_SAMPLING_RATE = 22050; // 采样率（和Python一致）
    const int WAV_BITS_PER_SAMPLE = 16; // 位深（16bit）
    const int WAV_CHANNELS = 1; // 单声道
    const float WAV_INT16_MAX = 32767.0f;

    // WAV文件头结构体（小端序）
    struct WavHeader {
        // RIFF Chunk
        char riff_id[4] = { 'R', 'I', 'F', 'F' };
        uint32_t riff_size; // 整个文件大小 - 8
        char riff_type[4] = { 'W', 'A', 'V', 'E' };

        // Format Chunk
        char fmt_id[4] = { 'f', 'm', 't', ' ' };
        uint32_t fmt_size = 16; // PCM格式固定为16
        uint16_t audio_format = 1; // 1=PCM（无压缩）
        uint16_t num_channels = WAV_CHANNELS; // 声道数
        uint32_t sample_rate = WAV_SAMPLING_RATE; // 采样率
        uint32_t byte_rate; // 字节率 = 采样率 * 声道数 * 位深/8
        uint16_t block_align; // 块对齐 = 声道数 * 位深/8
        uint16_t bits_per_sample = WAV_BITS_PER_SAMPLE; // 位深

        // Data Chunk
        char data_id[4] = { 'd', 'a', 't', 'a' };
        uint32_t data_size; // 音频数据字节数
    };

    // -------------------------- 对外接口函数 --------------------------
    bool FastSpeech2Postprocess(vector<string>& fileList, vector<TensorBuf>& outBufs, vector<TensorDesc>& outDescs)
    {
        LOG(INFO) << "====== 后处理启动 ======";
        if (outBufs.size() < 2 || fileList.size() != 1) {
            LOG(ERROR) << "size error: outBufs.size()=" << outBufs.size() << ", fileList.size()=" << fileList.size();
            return false;
        }

        auto cfg = ReadCfgFile("../data/cfg.txt");
        std::string saveWavDir = cfg["save_result_wav"];
        std::string saveBinDir = cfg["save_result_bin"];

        // 创建输出目录
        if (!CreateDirectoryRecursive(saveWavDir)) {
            LOG(ERROR) << "Create output txt directory failed: " << saveWavDir;
            return false;
        }

        if (!saveBinDir.empty() && !CreateDirectoryRecursive(saveBinDir)) {
            LOG(ERROR) << "Create result bin directory failed: " << saveBinDir;
            return false;
        }

        size_t wavIdx = 1;
        size_t melIdx = 0;
        std::string textId, text;
        std::string line = fileList[0];
        if (!ParseLine(line, textId, text)) {
            LOG(ERROR) << "Skip invalid line: " << line;
            return false;
        }

        // 保存原始输出bin
        if (!saveBinDir.empty()) {
            // 保存wave bin
            std::string waveBinPath = saveBinDir + "/" + textId + "_wave.bin";
            std::ofstream waveBinFile(waveBinPath, std::ios::binary);
            if (waveBinFile.is_open()) {
                waveBinFile.write(reinterpret_cast<char*>(outBufs[wavIdx].data.get()), outBufs[wavIdx].size);
                LOG(INFO) << "Saved wave output bin to: " << waveBinPath;
            } else {
                LOG(WARNING) << "Save wave output bin failed: " << waveBinPath;
            }

            // 保存mel bin
            std::string melBinPath = saveBinDir + "/" + textId + "_mel.bin";
            std::ofstream melBinFile(melBinPath, std::ios::binary);
            if (melBinFile.is_open()) {
                melBinFile.write(reinterpret_cast<char*>(outBufs[melIdx].data.get()), outBufs[melIdx].size);
                LOG(INFO) << "Saved mel output bin to: " << melBinPath;
            } else {
                LOG(WARNING) << "Save mel output bin failed: " << melBinPath;
            }
        }

        // 转换成音频格式输出
        if (!saveWavDir.empty()) {
            std::string wavPath = saveWavDir + "/" + textId + ".wav";
            std::ofstream wavFile(wavPath, std::ios::binary);
            if (wavFile.is_open()) {
                // 1. 读取原始wave数据（float32格式）
                float* waveFloatData = reinterpret_cast<float*>(outBufs[wavIdx].data.get());
                int waveDataLen = outBufs[wavIdx].size / sizeof(float); // float32数据长度
                LOG(INFO) << "Wave float data length: " << waveDataLen;

                // 2. 量化：float32 → int16（对齐Python：wave = (wave * INT16_MAX).astype(np.int16)）
                vector<int16_t> waveInt16Data;
                waveInt16Data.reserve(waveDataLen);
                for (int i = 0; i < waveDataLen; ++i) {
                    // 限幅（避免超出int16范围）
                    // ========== 新增：tanh计算（替代OM模型中移除的低精度tanh） ==========
                    float tanh_val = std::tanh(waveFloatData[i]);
                    // ===============================================================
                    float val = tanh_val * WAV_INT16_MAX; // 原代码是waveFloatData[i] * ...，改为tanh后的值
                    val = std::max(-WAV_INT16_MAX - 1.0f, std::min(val, WAV_INT16_MAX));
                    waveInt16Data.push_back(static_cast<int16_t>(val));
                }

                // 3. 构造WAV文件头
                WavHeader wavHeader;
                wavHeader.block_align = wavHeader.num_channels * wavHeader.bits_per_sample / 8;
                wavHeader.byte_rate = wavHeader.sample_rate * wavHeader.block_align;
                wavHeader.data_size = waveInt16Data.size() * sizeof(int16_t); // 音频数据字节数
                wavHeader.riff_size = 4 + (8 + wavHeader.fmt_size) + (8 + wavHeader.data_size); // 总大小-8

                // 4. 写入WAV文件（头 + 量化后的数据）
                // 注意：WAV头是小端序，x86/ARM架构默认小端，无需转换
                wavFile.write(reinterpret_cast<char*>(&wavHeader), sizeof(WavHeader));
                wavFile.write(reinterpret_cast<char*>(waveInt16Data.data()), wavHeader.data_size);

                wavFile.close();
                LOG(INFO) << "Saved WAV audio file to: " << wavPath;
            } else {
                LOG(WARNING) << "Save WAV audio file failed: " << wavPath;
            }
        }

        LOG(INFO) << "====== 后处理完成 ======";
        return true;
    }

}
} // namespace Infer