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

#include "facenet_postprocess.h"
#include "log.h"
#include "utils.h"
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace Infer {
namespace FaceNetNS {

    // 常量定义
    constexpr int FEATURE_DIMENSION = 512;
    constexpr size_t FLOAT_SIZE = sizeof(float);
    constexpr int BYTE_BIT_NUM = 8;
    constexpr int FP16_BIT_NUM = 16;

    std::string GetFileName(const std::string& path)
    {
        size_t sep = path.find_last_of("/\\");
        size_t start = (sep == std::string::npos) ? 0 : sep + 1;
        size_t end = path.find_last_of(".");
        return (end == std::string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
    }

    /**
     * 直接从特征数据保存txt文件
     */
    bool FeaturesToTxt(const float* features, const std::string& txtPath)
    {
        try {
            std::ofstream txtFile(txtPath);
            if (!txtFile.is_open()) {
                LOG(ERROR) << "Open txt file failed: " << txtPath;
                return false;
            }

            for (size_t i = 0; i < FEATURE_DIMENSION; ++i) {
                if (i > 0) {
                    txtFile << " ";
                }
                txtFile << features[i];
            }

            return true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Save txt file failed: " << txtPath << ", error: " << e.what();
            return false;
        }
    }

    /**
     * 人脸网络后处理主函数
     */
    bool FaceNetPostprocess(std::vector<std::string>& fileList,
        std::vector<TensorBuf>& outBufs,
        std::vector<TensorDesc>& outDescs)
    {
        LOG(INFO) << "====== Start FaceNet postprocessing ======";
        if (outBufs.size() != 1 || fileList.size() != 1) {
            LOG(ERROR) << "size error: outBufs.size()=" << outBufs.size() << ", fileList.size()=" << fileList.size();
            return false;
        }

        auto cfg = ReadCfgFile("../data/cfg.txt");
        std::string outputTxtDir = cfg["save_result_txt"];
        std::string saveBinDir = cfg["save_result_bin"];

        // 创建输出目录
        if (!CreateDirectoryRecursive(outputTxtDir)) {
            LOG(ERROR) << "Create output txt directory failed: " << outputTxtDir;
            return false;
        }

        if (!saveBinDir.empty() && !CreateDirectoryRecursive(saveBinDir)) {
            LOG(ERROR) << "Create result bin directory failed: " << saveBinDir;
            return false;
        }

        size_t idx = 0;
        const std::string& imgPath = fileList[idx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "Processing file: " << fileName;

        // 保存原始输出bin（保持原有配置支持：配置为空则不保存）
        if (!saveBinDir.empty()) {
            std::string binPath = saveBinDir + "/" + fileName + "_output.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            if (binFile.is_open()) {
                binFile.write(reinterpret_cast<char*>(outBufs[idx].data.get()),
                    outBufs[idx].size);
                LOG(INFO) << "Saved raw output bin to: " << binPath;
            } else {
                LOG(WARNING) << "Save raw output bin failed: " << binPath;
            }
        }

        // 获取推理结果数据并转换为FP32
        const float* featureData = nullptr;
        size_t dataSize = outBufs[idx].size;
        size_t typeSize = outDescs[idx].typeSize; // 16 or 32 bit
        size_t floatCount = dataSize / (typeSize / BYTE_BIT_NUM);
        std::vector<float> preds(floatCount);
        if (dataSize == 0 || outBufs[idx].data.get() == nullptr) {
            LOG(ERROR) << "Invalid output data for: " << fileName;
            return false;
        }
        if (typeSize == FP16_BIT_NUM) {
            const uint16_t* fp16Preds = reinterpret_cast<const uint16_t*>(outBufs[idx].data.get());
            for (size_t i = 0; i < floatCount; ++i) {
                preds[i] = HalfToFloat(fp16Preds[i]);
            }
            featureData = reinterpret_cast<const float*>(preds.data());
        } else {
            featureData = reinterpret_cast<const float*>(outBufs[idx].data.get());
        }

        // 保存txt文件
        std::string txtPath = outputTxtDir + "/" + fileName + ".txt";
        if (FeaturesToTxt(featureData, txtPath)) {
            LOG(INFO) << "Saved feature vector to txt: " << txtPath;
        } else {
            LOG(WARNING) << "Save feature vector to txt failed: " << txtPath;
        }

        LOG(INFO) << "====== FaceNet postprocessing completed ======";
        return true;
    }

} // namespace FaceNetNS
} // namespace Infer