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
#include "dev_interface_adapter.h"

using namespace Infer;
using namespace std;
constexpr float CONF_THRES = 0.25f;
constexpr float IOU_THRES = 0.45f;
constexpr int BYTE_BIT_NUM = 8;
constexpr uint8_t SCALE_SIZE = 3;
constexpr uint8_t CLASS_NUM = 80;
constexpr uint8_t OUT_PARM_NUM = 85; /* x, y, w,h, obj , class(80) */
#ifdef MODEL_CUST_CONIFG
const vector<uint32_t> EXPANDED_STRIDES = { 32, 16, 8 };
const vector<uint32_t> H_SIZES = { 20, 40, 80 };
const vector<uint32_t> W_SIZES = { 20, 40, 80 };
const vector<vector<uint32_t>> ANCHOR_GRIDS = {
    {116, 90, 156, 198, 373, 326},
    {30, 61, 62, 45, 59, 119},
    {10, 13, 16, 30, 33, 23}
};
#else
const vector<uint32_t> EXPANDED_STRIDES = { 8, 16, 32 };
const vector<uint32_t> H_SIZES = { 80, 40, 20 };
const vector<uint32_t> W_SIZES { 80, 40, 20 };
const vector<vector<uint32_t>> ANCHOR_GRIDS = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};
#endif

struct InferParam {
    string omModelPath;
    string aclConfigPath;
    string imglistPath;
    size_t loop {1};
};

struct DetectionInnerParam {
    float *outData { nullptr };
    size_t detectIdx { 0 };
    size_t wStrideOffset { 0 };
    float scoreThr { 0.0f};
    uint32_t outWidth { 0 };
    uint32_t chnStep { 0 };
    uint32_t outHeightIdx { 0 };
    uint32_t objScoreOffset { 0 };
};

enum VaildBoxId {
    SCORE_IDX = 0,
    XCENTER_IDX = 1,
    YCENTER_IDX = 2,
    W_IDX = 3,
    H_IDX = 4,
    CLASS_ID_IDX = 5
};

enum BoxValue {
    TOP_LEFT_X = 0,
    TOP_LEFT_Y = 1,
    BOTTOM_RIGHT_X = 2,
    BOTTOM_RIGHT_Y = 3,
    SCORE = 4,
    CLASS_ID = 5,
    BBOX_SIZE = 6
};

static bool PathToRealPath(const string &path, string &realPath)
{
    if (path.empty()) {
        return false;
    }
    if (path.length() > PATH_MAX) {
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        return false;
    }
    realPath = tmpPath;
    return true;
}

static int ReadImglistFile(const string& fileName, vector<string>& imglists)
{
    string absPath;
    if (!PathToRealPath(fileName, absPath)) {
        LOG(ERROR) << "get realpath failed, file: " << fileName;
        return -1;
    }
    ifstream imglistFile(absPath, ios::in);
    if (imglistFile.is_open() == false) {
        LOG(ERROR) << "open file " << fileName << " failed";
        return -1;
    }
    char *pathPtr = strdup(absPath.c_str());
    if (pathPtr == nullptr) {
        LOG(ERROR) << "strdup failed";
        return -1;
    }
    string path(dirname(pathPtr));
    free(pathPtr);
    pathPtr = nullptr;
    string img;
    while (getline(imglistFile, img)) {
        if (!imglistFile.eof() && !imglistFile.good()) {
            return -1;
        }
        img.erase(remove(img.begin(), img.end(), '\n'), img.end());
        img.erase(remove(img.begin(), img.end(), '\r'), img.end());
        string imgFullPath = path + "/" + img;
        imglists.push_back(imgFullPath);
    }
    return 0;
}

static int ReadImgFileToBuf(const string& fileName, const TensorDesc& desc, TensorBuf& inBuf)
{
    ifstream binFile(fileName, ifstream::binary);
    if (binFile.is_open() == false) {
        LOG(ERROR) << "open file " << fileName << " failed";
        return -1;
    }
    if (desc.defaultStride == 0) {
        binFile.read(static_cast<char*>(inBuf.GetRawPtr()), desc.defaultSize);
        return 0;
    }
    size_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    size_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

    for (size_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char*>(inBuf.GetRawPtr()) + loop * desc.defaultStride), lineSize);
    }

    return 0;
}

inline static float Sigmod(float a)
{
    return 1.0f/ (1.0f + exp(-a));
}

static void GetMaxScoreAndIdx(uint32_t objScoreIdx, uint32_t chnStep, const float* outData, float& maxClassScore, uint32_t& maxClassIdx)
{
    uint32_t classScoreIdx = objScoreIdx + chnStep;
    for(uint32_t c = 0; c < CLASS_NUM; c++) {
        float classScoreVal = outData[classScoreIdx];
        if(classScoreVal > maxClassScore) {
            maxClassScore = classScoreVal;
            maxClassIdx = c;
        }
        classScoreIdx += chnStep;
    }
}

static void ProcessPerDectectionInner(const DetectionInnerParam& innerParam, const vector<float>& gridsX,
    const vector<float>& gridsY, vector<vector<float>>& vaildBox)
{
    // 内容为 # [x1,y1,x2,y2,conf,class]
    float *outData = innerParam.outData;
    size_t wStrideOffset = innerParam.wStrideOffset;
    float scoreThr = innerParam.scoreThr;
    uint32_t chnStep = innerParam.chnStep;
    uint32_t outHeightIdx  = innerParam.outHeightIdx;
    uint32_t objScoreOffset = innerParam.objScoreOffset;
    uint32_t offset = outHeightIdx * innerParam.wStrideOffset;
    for(uint32_t j = 0; j < innerParam.outWidth; j++) {
        for(uint32_t k = 0; k < SCALE_SIZE; k++) {
            offset = j + outHeightIdx * wStrideOffset + k * chnStep * OUT_PARM_NUM;
            uint32_t objScoreIdx = offset + objScoreOffset;
            float objScoreVal = Sigmod(outData[objScoreIdx]);
            if(objScoreVal <= scoreThr) {
                continue;
            }
            // max score
            float maxClassScore = -1000.0f;
            uint32_t maxClassIdx = 0;
            GetMaxScoreAndIdx(objScoreIdx, chnStep, outData, maxClassScore, maxClassIdx);

            float confidenceScore = Sigmod(maxClassScore) * objScoreVal;

            if(confidenceScore > scoreThr) {
                uint32_t xCenterIdx = offset;
                uint32_t yCenterIdx = xCenterIdx + chnStep;
                uint32_t boxWidthIdx = yCenterIdx + chnStep;
                uint32_t boxHeightIdx = boxWidthIdx + chnStep;
                //解码公式x_center = (bbox_params[0, 0, :, :].sigmoid() + grid_x) / 80 * 640  # 假设原图640x640 ；y_center = (bbox_params[0, 1, :, :].sigmoid() + grid_y) / 80 * 640
                /** 
                解码边界框
                    grid_y, grid_x = torch.meshgrid(torch.arange(80), torch.arange(80))
                    x_center = (torch.sigmoid(output[..., 0]) * 2 - 0.5 + grid_x) *640/ 80
                    y_center = (torch.sigmoid(output[..., 1]) * 2 - 0.5 + grid_y)  *640/ 80
                    width = (2 * torch.sigmoid(output[..., 2]))^2 * anchor_w  # 假设输入640x640
                    height = (2 * torch.sigmoid(output[..., 3]))^2 * anchor_h 
                */
                
                float xCenter = (Sigmod(outData[xCenterIdx]) * 2 + gridsX[j]) * EXPANDED_STRIDES[innerParam.detectIdx]; // alg param
                float yCenter = (Sigmod(outData[yCenterIdx]) * 2 + gridsX[outHeightIdx]) * EXPANDED_STRIDES[innerParam.detectIdx]; // alg param

                float tmpValue = Sigmod(outData[boxWidthIdx]) * 2;
                float boxWidth = tmpValue * tmpValue * ANCHOR_GRIDS[innerParam.detectIdx][(k << 1)];

                tmpValue = Sigmod(outData[boxHeightIdx]) * 2;
                float boxHeight = tmpValue * tmpValue * ANCHOR_GRIDS[innerParam.detectIdx][(k << 1) + 1];

                vaildBox.push_back({confidenceScore, xCenter, yCenter, boxWidth, boxHeight, static_cast<float>(maxClassIdx)});
            }
        }
    }
}

static int ProcessPerDectection(size_t detectIdx, const TensorDesc& desc, TensorBuf& outBuf, vector<vector<float>>& vaildBox)
{
    DetectionInnerParam innerParam;
    innerParam.scoreThr = CONF_THRES;
    innerParam.detectIdx = detectIdx;
    innerParam.outData = static_cast<float*>(outBuf.GetRawPtr());
    innerParam.wStrideOffset = outBuf.stride / (desc.typeSize / BYTE_BIT_NUM);
    uint32_t outHeight = desc.dims[desc.dimCount - 2];
    innerParam.outWidth = desc.dims[desc.dimCount - 1];
    innerParam.chnStep = outHeight * innerParam.wStrideOffset;

    vector<float> gridsX(W_SIZES[detectIdx]);
    vector<float> gridsY(H_SIZES[detectIdx]);

    for(uint32_t i = 0; i < H_SIZES[detectIdx]; i++) {
        gridsY[i] = i - 0.5;
    }

    for(uint32_t i = 0; i < W_SIZES[detectIdx]; i++) {
        gridsX[i] = i - 0.5;
    }

    innerParam.objScoreOffset = 4 * innerParam.chnStep; // 4

    for(uint32_t i = 0; i < outHeight; i++) {
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

static void MulticlassNms(vector<vector<float>>& bboxes, const vector<vector<float>>& vaildBox, float nmsThr)
{
    for(auto &item : vaildBox) {
        float boxXCenter = item[XCENTER_IDX];
        float boxYCenter = item[YCENTER_IDX];
        float boxWidth = item[W_IDX];
        float boxHeight = item[H_IDX];

        float x1 = (boxXCenter - boxWidth / 2);
        float y1 = (boxYCenter - boxHeight / 2);
        float x2 = (boxXCenter + boxWidth / 2);
        float y2 = (boxYCenter + boxHeight / 2);

        float area = (x2 - x1 + 1) * (y2 - y1 + 1);

        bool keep = true;

        vector<float> bbox {x1, y1, x2, y2, item[SCORE_IDX], item[CLASS_ID_IDX], area};
        for(size_t j = 0; j < bboxes.size(); j++) {
            if(CalcIou(bbox, bboxes[j]) > nmsThr) {
                keep = false;
                break;
            }
        }
        if(keep) {
            bboxes.push_back(bbox);
        }
    }
}

static bool Cmp(const vector<float>& veci, const vector<float>& vecj)
{
    if(veci[CLASS_ID] < vecj[CLASS_ID]) {
        return true;
    } else if(veci[CLASS_ID] == vecj[CLASS_ID]) {
        return veci[SCORE] > vecj[SCORE];
    }
    return false;
}

static void SaveResult(const vector<vector<float>> &bboxes, vector<TensorBuf> &outBufs, const string &filePath)
{
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find(".");
    string outputName = filePath.substr(start, end - start);
    string resultPath = filePath.substr(0, start) + "/../../out/result";
    string binPath = resultPath + "/bin";
    string txtPath = resultPath + "/txt";
    for (auto& path : {resultPath, binPath, txtPath}) {
        if (stat(path.c_str(), nullptr) != 0) {
            mkdir(path.c_str(), 0755);
        }
    }
    for (size_t j = 0; j < outBufs.size(); j++) {
#ifdef MODEL_CUST_CONIFG
        string outputFileName = binPath + outputName +  "_" + to_string(2 - j) + ".bin";
#else
        string outputFileName = binPath + outputName +  "_" + to_string(j) + ".bin";
#endif
        ofstream binfout(outputFileName, ios::out | ios::trunc);
        binfout.write(static_cast<char*>(outBufs[j].GetRawPtr()), outBufs[j].size);
        binfout.close();
    }

    string txtFile = txtPath + outputName + ".txt";
    ofstream txtfout(txtFile, std::ios::out);
    for (auto &box : bboxes) {
        txtfout << "Class " << static_cast<int>(box[CLASS_ID]) << " | Score: " << box[SCORE] 
            << " | Box: [" << box[TOP_LEFT_X] << ", " << box[TOP_LEFT_Y] << ", "
            << box[BOTTOM_RIGHT_X] << ", " << box[BOTTOM_RIGHT_Y] << "]\n";
    }
    txtfout.close();
}

static int YoloVPostProcess(const shared_ptr<MdlBase> &model, vector<TensorBuf> &outBufs, const string &filePath)
{
    vector<vector<float>> vaildBox;
    for (size_t i = 0; i < outBufs.size(); i++) {
        TensorDesc desc;
        model->GetOutTensorDescByIdx(i, desc);
        if (desc.defaultStride == 0) {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            outBufs[i].stride = desc.defaultStride;
        }
        ProcessPerDectection(i, desc, outBufs[i], vaildBox);
    }
    sort(vaildBox.begin(), vaildBox.end(), [](const vector<float>& veci, const vector<float>& vecj){
        if (veci[0] > vecj[0]){
            return true;
        }
        return false;
    });
    vector<vector<float>> bboxes;
    MulticlassNms(bboxes, vaildBox, IOU_THRES);
    sort(bboxes.begin(), bboxes.end(), Cmp);
    SaveResult(bboxes, outBufs, filePath);
    LOG(INFO) << "dump final data success " << filePath;
    return 0;
}

static bool ParseCmd(int argc, char *argv[], InferParam &inferParam)
{
    int opt;
    const char *optstring = "hm:a:i:l:";
    struct option longOptions[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, optstring, longOptions, NULL)) != -1) {
        switch (opt) {
            case 'm':
                if (!PathToRealPath(optarg, inferParam.omModelPath)) {
                    LOG(ERROR) << "parse model path error";
                    return false;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, inferParam.aclConfigPath)) {
                    LOG(ERROR) << "parse acl config path error";
                    return false;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, inferParam.imglistPath)) {
                    LOG(ERROR) << "parse image dir error";
                    return false;
                }
                break;
            case 'l':{
                char *endptr = nullptr;
                inferParam.loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    LOG(ERROR) << "incorrect input after -l/--loop, " << endptr;
                    return false;
                }
                break;
            }
            case '?':
                LOG(ERROR) << "incorrect config";
                return false;
            default:
                return false;
        }
    }
    return true;
}

static int ModelInfer(InferParam &inferParam)
{
    shared_ptr<MdlBase> model = MdlCreate();
    int ret = model->LoadModel(inferParam.omModelPath);
    if (ret != 0) {
        LOG(ERROR) << "load model failed";
        return ret;
    }
    vector<string> imglists;
    ret = ReadImglistFile(inferParam.imglistPath, imglists);
    if (ret != 0) {
        LOG(ERROR) << "read img list file failed";
        model->UnLoadModel();
        return ret;
    }
    vector<TensorBuf> inBufs, outBufs;
    TensorDesc desc;
    for (size_t i = 0; i < model->GetInTensorNum(); i++) {
        model->GetInTensorDescByIdx(i, desc);
        inBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }
    for (size_t i = 0; i < model->GetOutTensorNum(); i++) {
        model->GetOutTensorDescByIdx(i, desc);
        outBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    if (inBufs.size() != 1 || outBufs.size() != 3) { // yolov5s.om : 1 input, 3 outputs
        LOG(ERROR) << "input tensor num should be 1, output tensor num should be 3";
        model->UnLoadModel();
        return -1;
    }
    model->GetInTensorDescByIdx(0, desc);
    chrono::microseconds dur(0);
    for (size_t i = 0; i < imglists.size(); ++i) {
        ret = ReadImgFileToBuf(imglists[i], desc, inBufs[0]);
        if (ret != 0) {
            LOG(ERROR) << "read img file to buf failed";
            model->UnLoadModel();
            return -1;
        }
        auto start = chrono::high_resolution_clock::now();
        for (size_t j = 0; j < inferParam.loop; j++) {
            ret = model->Execute(inBufs, outBufs);
            if (ret != 0) {
                LOG(ERROR) << "execute inference failed";
                model->UnLoadModel();
                DevDeInit();
                return -1;
            }
        }
        auto end = chrono::high_resolution_clock::now();
        dur += chrono::duration_cast<chrono::microseconds>(end - start);
        YoloVPostProcess(model, outBufs, imglists[i]);
    }
    LOG(INFO) << "time: " << dur.count() << ", fps: " << 1000.0 * 1000.0 * (inferParam.loop * imglists.size()) / (float)dur.count();
    model->UnLoadModel();
    return 0;
}

int main(int argc, char *argv[])
{
    InferParam inferParam;
    if (!ParseCmd(argc, argv, inferParam)) {
        LOG(ERROR) << "fail to parse cmd";
        return -1;
    }
    if (inferParam.imglistPath.empty() || inferParam.omModelPath.empty()) {
        LOG(ERROR) << "imglistPath: " << inferParam.imglistPath << ", omModelPath:" << inferParam.omModelPath;
        return -1;
    }
    int ret = DevInit(inferParam.aclConfigPath);
    if (ret != 0) {
        LOG(ERROR) << "dev init failed";
        return -1;
    }
    ret = ModelInfer(inferParam);
    if (ret != 0) {
        LOG(ERROR) << "model infer failed";
        return -1;
    }
    DevDeInit();
    return 0;
}