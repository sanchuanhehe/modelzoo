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

#include "yolo3_postprocess.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <map>

namespace Infer
{
using namespace std;
constexpr float CONF_THRES = 0.001f;
constexpr float IOU_THRES = 0.6f;
constexpr int BYTE_BIT_NUM = 8;
constexpr uint8_t SCALE_SIZE = 3;
constexpr uint8_t CLASS_NUM = 80;
constexpr uint8_t OUT_PARM_NUM = 85; /* x, y, w,h, obj , class(80) */
constexpr int IMG_SIZE = 640;
#ifdef SVP_ACL_PLATFORM
    const vector<uint32_t> EXPANDED_STRIDES = {32, 16, 8};
    const vector<uint32_t> H_SIZES = {20, 40, 80};
    const vector<uint32_t> W_SIZES = {20, 40, 80};
    const vector<vector<uint32_t>> ANCHOR_GRIDS = {
        {116, 90, 156, 198, 373, 326},
        {30, 61, 62, 45, 59, 119},
        {10, 13, 16, 30, 33, 23}};
#else
    const vector<uint32_t> EXPANDED_STRIDES = {8, 16, 32};
    const vector<uint32_t> H_SIZES = {80, 40, 20};
    const vector<uint32_t> W_SIZES{80, 40, 20};
    const vector<vector<uint32_t>> ANCHOR_GRIDS = {
        {10, 13, 16, 30, 33, 23},
        {30, 61, 62, 45, 59, 119},
        {116, 90, 156, 198, 373, 326}};
#endif

struct InferParam
    {
        string omModelPath;
        string aclConfigPath;
        string imglistPath;
        size_t loop{1};
    };

struct DetectionInnerParam
    {
        float *outData{nullptr};
        size_t detectIdx{0};
        size_t wStrideOffset{0};
        float scoreThr{0.0f};
        uint32_t outWidth{0};
        uint32_t chnStep{0};
        uint32_t outHeightIdx{0};
        uint32_t objScoreOffset{0};
    };

enum VaildBoxId
    {
        SCORE_IDX = 0,
        XCENTER_IDX = 1,
        YCENTER_IDX = 2,
        W_IDX = 3,
        H_IDX = 4,
        CLASS_ID_IDX = 5
    };

enum BoxValue
    {
        TOP_LEFT_X = 0,
        TOP_LEFT_Y = 1,
        BOTTOM_RIGHT_X = 2,
        BOTTOM_RIGHT_Y = 3,
        SCORE = 4,
        CLASS_ID = 5,
        BBOX_SIZE = 6
    };

std::vector<int> cocoCategoryIdsChange{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

static int ReadImgFileToBuf(const string &fileName, const TensorDesc &desc, TensorBuf &inBuf)
    {
        ifstream binFile(fileName, ifstream::binary);
        if (binFile.is_open() == false) {
            LOG(ERROR) << "open file " << fileName << " failed";
            return -1;
        }
        if (desc.defaultStride == 0) {
            binFile.read(static_cast<char *>(inBuf.GetRawPtr()), desc.defaultSize);
            return 0;
        }
        size_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        size_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

        for (size_t loop = 0; loop < loopTimes; loop++) {
            binFile.read((static_cast<char *>(inBuf.GetRawPtr()) + loop * desc.defaultStride), lineSize);
        }

        return 0;
    }

inline static float Sigmod(float a)
    {
        return 1.0f / (1.0f + exp(-a));
    }

static void GetMaxScoreAndIdx(uint32_t objScoreIdx, uint32_t chnStep, const float *outData, float &maxClassScore, uint32_t &maxClassIdx)
    {
        uint32_t classScoreIdx = objScoreIdx + chnStep;
        for (uint32_t c = 0; c < CLASS_NUM; c++) {
            float classScoreVal = outData[classScoreIdx];
            if (classScoreVal > maxClassScore) {
                maxClassScore = classScoreVal;
                maxClassIdx = c;
            }
            classScoreIdx += chnStep;
        }
    }

static void ProcessPerDectectionInner(const DetectionInnerParam &innerParam, const vector<float> &gridsX,
                                          const vector<float> &gridsY, vector<vector<float>> &vaildBox)
    {
        // 内容为 # [x1,y1,x2,y2,conf,class]
        float *outData = innerParam.outData;
        size_t wStrideOffset = innerParam.wStrideOffset;
        float scoreThr = innerParam.scoreThr;
        uint32_t chnStep = innerParam.chnStep;
        uint32_t outHeightIdx = innerParam.outHeightIdx;
        uint32_t objScoreOffset = innerParam.objScoreOffset;
        uint32_t offset = outHeightIdx * innerParam.wStrideOffset;
        for (uint32_t j = 0; j < innerParam.outWidth; j++)
        {
            for (uint32_t k = 0; k < SCALE_SIZE; k++)
            {
                offset = j + outHeightIdx * wStrideOffset + k * chnStep * OUT_PARM_NUM;
                uint32_t objScoreIdx = offset + objScoreOffset;
                float objScoreVal = Sigmod(outData[objScoreIdx]);
                
                float maxClassScore = -1000.0f;
                uint32_t maxClassIdx = 0;
               // GetMaxScoreAndIdx(objScoreIdx, chnStep, outData, maxClassScore, maxClassIdx);

                uint32_t classScoreIdx = objScoreIdx + chnStep;
                for (uint32_t c = 0; c < CLASS_NUM; c++)
                {
                    float classScoreVal = outData[classScoreIdx];
                    float confidenceScore = Sigmod(classScoreVal) * objScoreVal;
                    if (confidenceScore > scoreThr) {
                        uint32_t xCenterIdx = offset;
                        uint32_t yCenterIdx = xCenterIdx + chnStep;
                        uint32_t boxWidthIdx = yCenterIdx + chnStep;
                        uint32_t boxHeightIdx = boxWidthIdx + chnStep;
                       
                        float xCenter = (Sigmod(outData[xCenterIdx]) * 2 + gridsX[j]) * EXPANDED_STRIDES[innerParam.detectIdx];            // alg param
                        float yCenter = (Sigmod(outData[yCenterIdx]) * 2 + gridsX[outHeightIdx]) * EXPANDED_STRIDES[innerParam.detectIdx]; // alg param

                        float tmpValue = Sigmod(outData[boxWidthIdx]) * 2;
                        float boxWidth = tmpValue * tmpValue * ANCHOR_GRIDS[innerParam.detectIdx][(k << 1)];

                        tmpValue = Sigmod(outData[boxHeightIdx]) * 2;
                        float boxHeight = tmpValue * tmpValue * ANCHOR_GRIDS[innerParam.detectIdx][(k << 1) + 1];

                        vaildBox.push_back({confidenceScore, xCenter, yCenter, boxWidth, boxHeight, static_cast<float>(c)});
                    }
                    classScoreIdx += chnStep;
                }
                float confidenceScore = Sigmod(maxClassScore) * objScoreVal;
            }
        }
    }

static int ProcessPerDectection(size_t detectIdx, const TensorDesc &desc, TensorBuf &outBuf, vector<vector<float>> &vaildBox)
    {
        DetectionInnerParam innerParam;
        innerParam.scoreThr = CONF_THRES;
        innerParam.detectIdx = detectIdx;
        innerParam.outData = static_cast<float *>(outBuf.GetRawPtr());
        innerParam.wStrideOffset = outBuf.stride / (desc.typeSize / BYTE_BIT_NUM);
        uint32_t outHeight = desc.dims[desc.dimCount - 2];
        innerParam.outWidth = desc.dims[desc.dimCount - 1];
        innerParam.chnStep = outHeight * innerParam.wStrideOffset;

        vector<float> gridsX(W_SIZES[detectIdx]);
        vector<float> gridsY(H_SIZES[detectIdx]);

        for (uint32_t i = 0; i < H_SIZES[detectIdx]; i++) {
            gridsY[i] = i - 0.5;
        }

        for (uint32_t i = 0; i < W_SIZES[detectIdx]; i++) {
            gridsX[i] = i - 0.5;
        }

        innerParam.objScoreOffset = 4 * innerParam.chnStep; // 4

        for (uint32_t i = 0; i < outHeight; i++) {
            innerParam.outHeightIdx = i;
            ProcessPerDectectionInner(innerParam, gridsX, gridsY, vaildBox);
        }
        return SUCCESS;
    }

static float CalcIou(const vector<float> &box1, const vector<float> &box2)
    {
        float area1 = box1[6];
        float area2 = box2[6];

        float xx1 = max(box1[0], box2[0]);
        float yy1 = max(box1[1], box2[1]);
        float xx2 = min(box1[2], box2[2]);
        float yy2 = min(box1[3], box2[3]);

        float w = max(0.0f, xx2 - xx1 + 1);
        float h = max(0.0f, yy2 - yy1 + 1);

        float inter = w * h;
        float ovr = inter / (area1 + area2 - inter);
        return ovr;
    }

static void MulticlassNms(vector<vector<float>> &bboxes, const vector<vector<float>> &vaildBox, float nmsThr)
    {   
        // 按类别分组
        map<int, vector<vector<float>>> classBoxes;
        for (const auto& box : vaildBox) {
            classBoxes[box[CLASS_ID_IDX]].push_back(box);
        }
        
        for (map<int, vector<vector<float>>>::iterator it = classBoxes.begin(); it != classBoxes.end(); ++it) {
            vector<vector<float>>& clsBoxes = it->second;  // 只获取需要的value，不定义key变量
            sort(clsBoxes.begin(), clsBoxes.end(), [](const vector<float> &veci, const vector<float> &vecj)
                {
                if (veci[SCORE_IDX] > vecj[SCORE_IDX]) {
                    return true;
                }
                return false; });
            vector<bool> keep(clsBoxes.size(), true);
            for (size_t i = 0; i < clsBoxes.size(); ++i) {
                    if (!keep[i]) continue;
                    vector<float> item = clsBoxes[i];
                    float boxXCenter = item[XCENTER_IDX];
                    float boxYCenter = item[YCENTER_IDX];
                    float boxWidth = item[W_IDX];
                    float boxHeight = item[H_IDX];

                    float x1 = (boxXCenter - boxWidth / 2);
                    float y1 = (boxYCenter - boxHeight / 2);
                    float x2 = (boxXCenter + boxWidth / 2);
                    float y2 = (boxYCenter + boxHeight / 2);

                    float area = (x2 - x1 + 1) * (y2 - y1 + 1);


                    vector<float> bbox{x1, y1, x2, y2, item[SCORE_IDX], item[CLASS_ID_IDX], area};
                    bboxes.push_back(bbox);
                    for (size_t j = i+1; j < clsBoxes.size(); j++) {
                        if (!keep[j]) continue;
                        // 转换候选框坐标
                        vector<float> tmp = clsBoxes[j];
                        float xc2 = tmp[XCENTER_IDX];
                        float yc2 = tmp[YCENTER_IDX];
                        float w2 = tmp[W_IDX];
                        float h2 = tmp[H_IDX];
                        float x1_2 = std::round(xc2 - w2 / 2);
                        float y1_2 = std::round(yc2 - h2 / 2);
                        float x2_2 = std::round(xc2 + w2 / 2);
                        float y2_2 = std::round(yc2 + h2 / 2);
                        float area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1);
                        if (CalcIou(bbox,  {x1_2, y1_2, x2_2, y2_2, 0, 0, area2}) > nmsThr)
                        {
                            keep[j] = false;
                        }
                    }
                }
        }

    }

static bool Cmp(const vector<float> &veci, const vector<float> &vecj)
    {
        if (veci[CLASS_ID] < vecj[CLASS_ID]) {
            return true;
        } else if (veci[CLASS_ID] == vecj[CLASS_ID]) {
            return veci[SCORE] > vecj[SCORE];
        }
        return false;
    }

// 裁剪边界框到图像范围内
static void clipBoxes(vector<vector<float>> &bboxes, const cv::Size shape)
    {
        int height = shape.height;
        int width = shape.width;

        for (auto &box : bboxes) {
            box[TOP_LEFT_X] = std::max(0.0f, std::min(box[TOP_LEFT_X], static_cast<float>(width)));
            box[TOP_LEFT_Y] = std::max(0.0f, std::min(box[TOP_LEFT_Y], static_cast<float>(height)));
            box[BOTTOM_RIGHT_X] = std::max(0.0f, std::min(box[BOTTOM_RIGHT_X], static_cast<float>(width)));
            box[BOTTOM_RIGHT_Y] = std::max(0.0f, std::min(box[BOTTOM_RIGHT_Y], static_cast<float>(height)));
        }
    }

// 缩放边界框
static void scaleBoxes(vector<vector<float>> &bboxes, const string &filePath)
    {
        // img1_shape: (height, width) - 处理后图像的尺寸
        // img0_shape: (height, width) - 原始图像的尺寸
        // boxes: 边界框列表

        int h1 = IMG_SIZE;
        int w1 = IMG_SIZE;
        LOG(INFO) << "filePath: " << filePath;
        cv::Mat im0 = cv::imread(filePath);

        // 获取原始尺寸
        cv::Size shape = im0.size();
        int h0 = shape.height;
        int w0 = shape.width;

        // 计算增益 (gain)
        double gain = std::min(static_cast<double>(h1) / h0,
                               static_cast<double>(w1) / w0);

        LOG(INFO) << "gain: " << gain;
        // 计算填充 (pad)
        double pad_x = (w1 - w0 * gain) / 2.0f;
        double pad_y = (h1 - h0 * gain) / 2.0f;
        LOG(INFO) << "bboxes: " << bboxes.size();
        LOG(INFO) << "pad_x: " << pad_x << " pad_y:  " << pad_y;
        for (vector<float> &box : bboxes) {
            // 检查边界框大小
            if (box.size() < 4) {
                LOG(ERROR) << "Box " << " has insufficient elements: " << box.size();
                return;
            }
                        // 减去填充
            box[TOP_LEFT_X] -= pad_x;
            box[BOTTOM_RIGHT_X] -= pad_x;
            box[TOP_LEFT_Y] -= pad_y;
            box[BOTTOM_RIGHT_Y] -= pad_y;

            // 除以增益进行缩放
            box[TOP_LEFT_X] /= gain;
            box[BOTTOM_RIGHT_X] /= gain;
            box[TOP_LEFT_Y] /= gain;
            box[BOTTOM_RIGHT_Y] /= gain;

            box[TOP_LEFT_X] = std::max(0.0f, std::min(box[TOP_LEFT_X], static_cast<float>(w0)));
            box[TOP_LEFT_Y] = std::max(0.0f, std::min(box[TOP_LEFT_Y], static_cast<float>(h0)));
            box[BOTTOM_RIGHT_X] = std::max(0.0f, std::min(box[BOTTOM_RIGHT_X], static_cast<float>(w0)));
            box[BOTTOM_RIGHT_Y] = std::max(0.0f, std::min(box[BOTTOM_RIGHT_Y], static_cast<float>(h0)));

            box[TOP_LEFT_X] = std::round(box[TOP_LEFT_X]);
            box[BOTTOM_RIGHT_X] = std::round(box[BOTTOM_RIGHT_X]);
            box[TOP_LEFT_Y] = std::round(box[TOP_LEFT_Y]);
            box[BOTTOM_RIGHT_Y] = std::round(box[BOTTOM_RIGHT_Y]);

            std::vector<float> xywh_box(4);
            xywh_box[TOP_LEFT_X] = (box[TOP_LEFT_X] + box[BOTTOM_RIGHT_X]) / 2.0f; // x_center
            xywh_box[TOP_LEFT_Y] = (box[TOP_LEFT_Y] + box[BOTTOM_RIGHT_Y]) / 2.0f; // y_center
            xywh_box[BOTTOM_RIGHT_X] = box[BOTTOM_RIGHT_X] - box[TOP_LEFT_X];          // width
            xywh_box[BOTTOM_RIGHT_Y] = box[BOTTOM_RIGHT_Y] - box[TOP_LEFT_Y];          // height

            xywh_box[TOP_LEFT_X] -= xywh_box[BOTTOM_RIGHT_X] / 2.0f;
            xywh_box[TOP_LEFT_Y] -= xywh_box[BOTTOM_RIGHT_Y] / 2.0f;
            box[TOP_LEFT_X] = xywh_box[TOP_LEFT_X];
            box[BOTTOM_RIGHT_X] = xywh_box[BOTTOM_RIGHT_X];
            box[TOP_LEFT_Y] = xywh_box[TOP_LEFT_Y];
            box[BOTTOM_RIGHT_Y] = xywh_box[BOTTOM_RIGHT_Y];
        }
        return;
    }

static void SaveResult(const vector<vector<float>> &bboxes, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs, const string &filePath)
    {
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");
        string outputName = filePath.substr(start, end - start);
        string resultPath = "../out/result";
        string binPath = resultPath + "/bin";
        string txtPath = resultPath + "/txt";
        struct stat info; 
        if (stat(resultPath.c_str(), &info) != 0) {
            mkdir(resultPath.c_str(), 0777);
            INFO_LOG("create file success");
        }

        if (stat(txtPath.c_str(), &info) != 0) {
            mkdir(txtPath.c_str(), 0777);
        }

        if (stat(binPath.c_str(), &info) != 0) {
            mkdir(binPath.c_str(), 0777);
        }
        for (size_t j = 0; j < tensorBufs.size(); j++) {
#ifdef SVP_ACL_PLATFORM
            string outputFileName = binPath + outputName + "_" + to_string(2 - j) + ".bin";
#else
            string outputFileName = binPath + outputName + "_" + to_string(j) + ".bin";
#endif
            LOG(INFO) << " outputFileName: " << outputFileName;
            ofstream binfout(outputFileName, ios::out | ios::trunc);
            binfout.write(static_cast<char *>(tensorBufs[j].GetRawPtr()), tensorBufs[j].size);
            binfout.close();
        }

        string txtFile = txtPath + outputName + ".txt";
        ofstream txtfout(txtFile, std::ios::out);
        for (auto &box : bboxes) {
            txtfout << "Class " << cocoCategoryIdsChange[static_cast<int>(box[CLASS_ID])] << " | Score: " << box[SCORE]
                    << " | Box: [" << box[TOP_LEFT_X] << ", " << box[TOP_LEFT_Y] << ", "
                    << box[BOTTOM_RIGHT_X] << ", " << box[BOTTOM_RIGHT_Y] << "]\n";
        }
        txtfout.close();
    }

bool GetYolo3Box(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
    {
        if (tensorBufs.size() == 0 || fileList.size() == 0) {
            LOG(INFO) << " modelOuput.size() == 0 || inputFileList.size() == 0  NO OUT";
            return false;
        }
        vector<vector<float>> bboxes;
        vector<vector<float>> vaildBox;
        for (size_t i = 0; i < tensorBufs.size(); i++) {
            TensorDesc desc = tensorDescs[i];
            TensorBuf buf = tensorBufs[i];
            if (desc.defaultStride == 0) {
                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
            }
            ProcessPerDectection(i, desc, buf, vaildBox);
        }
        MulticlassNms(bboxes, vaildBox, IOU_THRES);
        sort(bboxes.begin(), bboxes.end(), Cmp);
        scaleBoxes(bboxes, fileList[0]);
        SaveResult(bboxes, tensorBufs, tensorDescs, fileList[0]);
        LOG(INFO) << "dump final data success " << fileList[0];
        return true;
    }
}