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

#include "yolo11s_seg_postprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

using json = nlohmann::json;
namespace Infer {
namespace Yolo11sSegNS {
    using namespace std;

    constexpr int BYTE_BIT_NUM = 8;
    constexpr int FP16_BIT_NUM = 16;
    constexpr int FP32_BIT_NUM = 32;
    const float MinValue = 0.0f;
    const int Yolo80ClassCount = 80;
    const int NumBoxes = 8400;
    const int BoxDim = 116;
    const int MaskCoeffStartIdx = 84;
    const int ProtoChannels = 32;
    const int ProtoHeight = 160;
    const int ProtoWidth = 160;
    const float HalfValue = 2.0f;
    const int MinContourPoints = 3;
    const int ContourPointStep = 2;
    const float MaskThreshold = 0.0f; // 掩码阈值，与Python保持一致

    struct BBox {
        // 给所有POD成员设置默认值，vector会自动初始化
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f, score = 0.0f;
        int classId = -1;
        int cocoClassId = -1;
        vector<float> maskCoeffs; // 非POD类型，自动初始化
        float input_x1 = 0.0f, input_y1 = 0.0f, input_x2 = 0.0f, input_y2 = 0.0f;
    };

    const vector<int> Yolo80ToCoco90 = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    };

    string GetFileName(const string& path)
    {
        size_t sep = path.find_last_of("/\\");
        size_t start = (sep == string::npos) ? 0 : sep + 1;
        size_t end = path.find_last_of(".");
        return (end == string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
    }

    vector<BBox> BatchedNMS(vector<BBox>& boxes, float iouThres)
    {
        map<int, vector<BBox>> classBoxes;
        for (const auto& box : boxes) {
            classBoxes[box.classId].push_back(box);
        }

        vector<BBox> nmsResult;
        for (auto& entry : classBoxes) {
            vector<BBox>& clsBoxes = entry.second;
            sort(clsBoxes.begin(), clsBoxes.end(), [](const BBox& a, const BBox& b) {
                return a.score > b.score;
            });

            vector<bool> keep(clsBoxes.size(), true);
            for (size_t i = 0; i < clsBoxes.size(); ++i) {
                if (!keep[i])
                    continue;
                nmsResult.push_back(clsBoxes[i]);

                const BBox& boxI = clsBoxes[i];
                for (size_t j = i + 1; j < clsBoxes.size(); ++j) {
                    if (!keep[j])
                        continue;
                    const BBox& boxJ = clsBoxes[j];

                    float interLeft = max(boxI.x1, boxJ.x1);
                    float interTop = max(boxI.y1, boxJ.y1);
                    float interRight = min(boxI.x2, boxJ.x2);
                    float interBottom = min(boxI.y2, boxJ.y2);
                    float interArea = max(0.0f, interRight - interLeft) * max(0.0f, interBottom - interTop);

                    float areaI = (boxI.x2 - boxI.x1) * (boxI.y2 - boxI.y1);
                    float areaJ = (boxJ.x2 - boxJ.x1) * (boxJ.y2 - boxJ.y1);
                    float unionArea = areaI + areaJ - interArea;
                    float iou = unionArea > 0.0f ? interArea / unionArea : 0.0f;

                    if (iou > iouThres) {
                        keep[j] = false;
                    }
                }
            }
        }
        return nmsResult;
    }

    void CalculateScaleAndPad(int originalHeight, int originalWidth, int targetWidth, int targetHeight,
        float& scale, int& padWidth, int& padHeight)
    {
        scale = min(static_cast<float>(targetWidth) / originalWidth,
            static_cast<float>(targetHeight) / originalHeight);

        float newWidth = originalWidth * scale;
        float newHeight = originalHeight * scale;

        int deltaWidth = targetWidth - static_cast<int>(round(newWidth));
        int deltaHeight = targetHeight - static_cast<int>(round(newHeight));
        padWidth = deltaWidth / 2;
        padHeight = deltaHeight / 2;

        padWidth = static_cast<int>(round(padWidth - 0.1));
        padHeight = static_cast<int>(round(padHeight - 0.1));
    }

    void ScaleBoxes(vector<BBox>& boxes, int originalWidth, int originalHeight,
        float scale, int padWidth, int padHeight)
    {
        for (auto& box : boxes) {
            box.input_x1 = box.x1;
            box.input_y1 = box.y1;
            box.input_x2 = box.x2;
            box.input_y2 = box.y2;

            box.x1 = (box.x1 - padWidth) / scale;
            box.y1 = (box.y1 - padHeight) / scale;
            box.x2 = (box.x2 - padWidth) / scale;
            box.y2 = (box.y2 - padHeight) / scale;

            box.x1 = max(0.0f, min(box.x1, static_cast<float>(originalWidth)));
            box.y1 = max(0.0f, min(box.y1, static_cast<float>(originalHeight)));
            box.x2 = max(0.0f, min(box.x2, static_cast<float>(originalWidth)));
            box.y2 = max(0.0f, min(box.y2, static_cast<float>(originalHeight)));
        }
    }

    cv::Mat CropMask(const cv::Mat& mask, const BBox& box, int maskHeight, int maskWidth)
    {
        float x1 = box.x1;
        float y1 = box.y1;
        float x2 = box.x2;
        float y2 = box.y2;

        cv::Mat maskRegion = cv::Mat::zeros(mask.size(), CV_32FC1);
        for (int y = 0; y < maskHeight; y++) {
            for (int x = 0; x < maskWidth; x++) {
                if (x >= x1 && x < x2 && y >= y1 && y < y2) {
                    maskRegion.at<float>(y, x) = 1.0f;
                }
            }
        }

        cv::Mat croppedMask = mask.mul(maskRegion);
        return croppedMask;
    }

    cv::Mat ProcessMask(const float* protos, const vector<float>& maskCoeffs, const BBox& box,
        int targetWidth, int targetHeight)
    {
        cv::Mat protosMat(ProtoChannels, ProtoHeight * ProtoWidth, CV_32FC1, const_cast<float*>(protos));
        cv::Mat coeffsMat(1, ProtoChannels, CV_32FC1, const_cast<float*>(maskCoeffs.data()));

        cv::Mat maskMat;
        cv::gemm(coeffsMat, protosMat, 1.0, cv::noArray(), 0.0, maskMat);
        maskMat = maskMat.reshape(1, ProtoHeight);

        float widthRatio = static_cast<float>(ProtoWidth) / targetWidth;
        float heightRatio = static_cast<float>(ProtoHeight) / targetHeight;

        BBox downsampledBox;
        downsampledBox.x1 = box.input_x1 * widthRatio;
        downsampledBox.y1 = box.input_y1 * heightRatio;
        downsampledBox.x2 = box.input_x2 * widthRatio;
        downsampledBox.y2 = box.input_y2 * heightRatio;

        cv::Mat croppedMask = CropMask(maskMat, downsampledBox, ProtoHeight, ProtoWidth);
        if (croppedMask.empty()) {
            LOG(WARNING) << "ProcessMask - croppedMask is empty";
            return cv::Mat();
        }

        cv::Mat resizedMask;
        cv::resize(croppedMask, resizedMask, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);
        return resizedMask;
    }

    vector<vector<float>> MaskToCocoSegment(const BBox& box, const cv::Mat& maskBinary)
    {
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;

        cv::findContours(maskBinary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<vector<float>> validSegments;
        for (const auto& contour : contours) {
            if (contour.size() < MinContourPoints)
                continue;

            vector<float> seg;
            for (const auto& point : contour) {
                float x = max(box.x1, min(static_cast<float>(point.x), box.x2));
                float y = max(box.y1, min(static_cast<float>(point.y), box.y2));
                seg.push_back(x);
                seg.push_back(y);
            }

            if (seg.size() % ContourPointStep != 0) {
                seg.pop_back();
            }

            if (seg.size() >= MinContourPoints * ContourPointStep) {
                validSegments.push_back(seg);
            }
        }

        if (validSegments.empty()) {
            validSegments.push_back({});
        }
        return validSegments;
    }

    void SaveJsonResults(const vector<BBox>& results, const string& savePath, const string& fileName,
        const float* protos, int targetWidth, int targetHeight,
        float scale, int padWidth, int padHeight, int originalWidth, int originalHeight)
    {
        json jsonResults = json::array();
        int validCount = 0;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& box = results[i];

            if (box.cocoClassId == -1) {
                continue;
            }

            float bboxWidth = box.x2 - box.x1;
            float bboxHeight = box.y2 - box.y1;
            if (bboxWidth <= 0 || bboxHeight <= 0) {
                continue;
            }

            cv::Mat mask = ProcessMask(protos, box.maskCoeffs, box, targetWidth, targetHeight);
            if (mask.empty()) {
                continue;
            }

            cv::Rect roi(padWidth, padHeight,
                targetWidth - 2 * padWidth,
                targetHeight - 2 * padHeight);
            if (roi.width <= 0 || roi.height <= 0 || roi.x + roi.width > mask.cols || roi.y + roi.height > mask.rows) {
                LOG(WARNING) << "Invalid ROI for mask cropping";
                continue;
            }

            cv::Mat croppedMask = mask(roi).clone();
            if (croppedMask.empty()) {
                continue;
            }

            cv::Mat resizedMask;
            cv::resize(croppedMask, resizedMask, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_LINEAR);

            cv::Mat maskBinary;
            cv::threshold(resizedMask, maskBinary, MaskThreshold, 1.0f, cv::THRESH_BINARY);
            maskBinary.convertTo(maskBinary, CV_8UC1);

            auto segmentation = MaskToCocoSegment(box, maskBinary);
            if (segmentation == vector<vector<float>> { {} }) {
                continue;
            }

            json result;
            result["image_id"] = stoi(fileName);
            result["category_id"] = box.cocoClassId;
            result["bbox"] = { box.x1, box.y1, bboxWidth, bboxHeight };
            result["score"] = box.score;
            result["segmentation"] = segmentation;
            jsonResults.push_back(result);

            validCount++;
        }

        string jsonPath = savePath + "/" + fileName + ".json";
        ofstream jsonFile(jsonPath);
        if (jsonFile.is_open()) {
            jsonFile << jsonResults.dump(2);
            jsonFile.close();
            LOG(INFO) << "Saved " << validCount << " valid results to: " << jsonPath;
        } else {
            LOG(WARNING) << "Failed to open json file: " << jsonPath;
        }
    }

    bool saveBinFile(const string& savePath, const void* data, size_t dataSize)
    {
        ofstream binFile(savePath, ios::binary);
        if (!binFile.is_open()) {
            LOG(WARNING) << "无法打开bin文件: " << savePath;
            return false;
        }
        binFile.write(reinterpret_cast<const char*>(data), dataSize);
        binFile.close();
        return true;
    }

    const float* GetOutDataAsFP32(const TensorBuf& buf, const TensorDesc& desc, std::vector<float>& preds)
    {
        // 清空输出容器（确保数据干净）
        preds.clear();

        // 获取Tensor基础信息
        size_t typeSize = desc.typeSize;
        size_t floatCount = buf.size / (typeSize / BYTE_BIT_NUM);

        // 根据数据类型处理
        if (typeSize == FP16_BIT_NUM) {
            // FP16转FP32
            const uint16_t* fp16Preds = reinterpret_cast<const uint16_t*>(buf.data.get());
            preds.resize(floatCount);
            for (size_t i = 0; i < floatCount; ++i) {
                preds[i] = HalfToFloat(fp16Preds[i]);
            }
            return preds.data();
        } else if (typeSize == FP32_BIT_NUM) {
            // 直接返回FP32数据
            return reinterpret_cast<const float*>(buf.data.get());
        } else {
            LOG(ERROR) << "output unsupported desc.typeSize: " << typeSize;
            return nullptr;
        }
    }

    bool Yolov11sSegPostprocess(vector<string>& fileList,
        vector<TensorBuf>& outBufs,
        vector<TensorDesc>& outDescs)
    {
        LOG(INFO) << "====== Start Postprocessing ======";

        auto cfg = ReadCfgFile("../data/cfg.txt");
        float confThres = stof(cfg["conf_threshold"]);
        float nmsThres = stof(cfg["nms_threshold"]);
        string saveBin = cfg["save_result_bin"];
        string saveJson = cfg["save_result_json"];
        int targetWidth = stoi(cfg["img_width"]);
        int targetHeight = stoi(cfg["img_height"]);

        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "Failed to create bin directory: " << saveBin;
            return false;
        }
        if (!saveJson.empty() && !CreateDirectoryRecursive(saveJson)) {
            LOG(ERROR) << "Failed to create json directory: " << saveJson;
            return false;
        }

        if (fileList.size() * 2 != outBufs.size()) {
            LOG(ERROR) << "Mismatch between file count and output buffers";
            return false;
        }

        const string& imgPath = fileList[0];
        string fileName = GetFileName(imgPath);

        int detBufIdx = 0;
        int protoBufIdx = 1;
        // dpico核的输出顺序和dlite核不一样
        if (outDescs[detBufIdx].defaultStride != 0) {
            detBufIdx = 1;
            protoBufIdx = 0;
        }
        std::vector<float> detectionsPreds;
        std::vector<float> protosPreds;
        const float* detectionsData = GetOutDataAsFP32(outBufs[detBufIdx], outDescs[detBufIdx], detectionsPreds);
        const float* protosData = GetOutDataAsFP32(outBufs[protoBufIdx], outDescs[protoBufIdx], protosPreds);

        if (!detectionsData || !protosData) {
            LOG(WARNING) << "Null output buffer";
            return false;
        }

        if (!saveBin.empty()) {
            string binPath = saveBin + "/" + fileName + "_output_0.bin";
            saveBinFile(binPath, detectionsData, outBufs[detBufIdx].size);
            binPath = saveBin + "/" + fileName + "_output_1.bin";
            saveBinFile(binPath, protosData, outBufs[protoBufIdx].size);
        }

        vector<BBox> boxes;
        for (int j = 0; j < NumBoxes; ++j) {
            float cx = detectionsData[0 * NumBoxes + j];
            float cy = detectionsData[1 * NumBoxes + j];
            float w = detectionsData[2 * NumBoxes + j];
            float h = detectionsData[3 * NumBoxes + j];

            float maxClsScore = -INFINITY;
            int classId = 0;
            for (int c = 0; c < Yolo80ClassCount; ++c) {
                int k = 4 + c;
                float clsScore = detectionsData[k * NumBoxes + j];
                if (clsScore > maxClsScore) {
                    maxClsScore = clsScore;
                    classId = c;
                }
            }

            if (maxClsScore < confThres)
                continue;

            BBox box;
            box.x1 = cx - w / HalfValue;
            box.y1 = cy - h / HalfValue;
            box.x2 = cx + w / HalfValue;
            box.y2 = cy + h / HalfValue;
            box.score = maxClsScore;
            box.classId = classId;
            box.cocoClassId = -1;

            for (int k = MaskCoeffStartIdx; k < BoxDim; ++k) {
                box.maskCoeffs.push_back(detectionsData[k * NumBoxes + j]);
            }

            boxes.push_back(box);
        }

        LOG(INFO) << "Boxes before NMS: " << boxes.size();
        vector<BBox> nmsResult = BatchedNMS(boxes, nmsThres);
        LOG(INFO) << "Boxes after NMS: " << nmsResult.size();

        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(WARNING) << "Failed to read image: " << imgPath;
            return false;
        }
        int originalHeight = img.rows;
        int originalWidth = img.cols;

        float scale;
        int padWidth, padHeight;
        CalculateScaleAndPad(originalHeight, originalWidth, targetWidth, targetHeight, scale, padWidth, padHeight);

        ScaleBoxes(nmsResult, originalWidth, originalHeight, scale, padWidth, padHeight);

        const size_t cocoSize = Yolo80ToCoco90.size();
        for (auto& box : nmsResult) {
            if (box.classId >= 0 && static_cast<size_t>(box.classId) < cocoSize) {
                box.cocoClassId = Yolo80ToCoco90[box.classId];
            } else {
                box.cocoClassId = -1;
            }
        }

        if (!saveJson.empty()) {
            SaveJsonResults(nmsResult, saveJson, fileName, protosData, targetWidth, targetHeight,
                scale, padWidth, padHeight, originalWidth, originalHeight);
        }

        LOG(INFO) << "====== Postprocessing Completed ======";
        return true;
    }

} // namespace
} // namespace Infer