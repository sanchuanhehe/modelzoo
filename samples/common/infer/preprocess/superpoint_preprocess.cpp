#include "superpoint_preprocess.h"
#include "log.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include "utils.h"
#include <nlohmann/json.hpp>
#include "PillowResize/PillowResize.hpp"

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
namespace Infer
{
    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;
        size_t lineSize = width;

        // 获取数据指针
        const float *srcData = chwImg.ptr<float>();

        for (int64_t loop = 0; loop < loopTimes; loop++) {
            // 添加循环边界检查

            float *destPtr = static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            // const float *srcPtr = chwData.data() + loop * width;
            const float *srcPtr = srcData + loop * width;
            // 检查指针有效性
            if (destPtr == nullptr || srcPtr == nullptr)
            {
                LOG(ERROR) << "Null pointer detected";
                return FAILED;
            }
            // 拷贝数据
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

    static cv::Mat HwcToChw(const cv::Mat &input)
    {
        // 确保输入是单通道
        if (input.channels() != 1) {
            std::cerr << "错误: 输入图像不是单通道灰度图" << std::endl;
            return cv::Mat();
        }

        int height = input.rows;
        int width = input.cols;

        // 创建 1xHxW 的矩阵（CHW格式）
        cv::Mat output(1, height * width, input.type());

        // 将HxW展平为1x(H*W)
        input.reshape(1, 1).copyTo(output);

        // 如果需要恢复形状为1xHxW（2D数组）
        cv::Mat chw_format = output.reshape(1, 1);

        return chw_format;
    }

    bool SuperPointPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
    {
        // 进度显示（简化版）
        LOG(INFO) << " SuperPointPreprocess ";
        for (size_t i = 0; i < fileList.size(); ++i) {

            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;

            cv::Mat img = cv::imread(fileList[0]);
            if (img.empty()) {
                throw std::runtime_error("Failed to load image: " + fileList[0]);
            }
            if (tensorDescs[i].defaultStride == 0) {
                tensorDescs[i].defaultStride = tensorDescs[i].dims[tensorDescs[i].dimCount - 1] * tensorDescs[i].typeSize / BYTE_BIT_NUM;
                tensorBufs[i].stride = tensorDescs[i].defaultStride;
            }
            // 计算缩放比例
            double targetH = 240;
            double targetW = 320;
            double scaleW = static_cast<double>(targetW) / img.cols;
            double scaleH = static_cast<double>(targetH) / img.rows;
            double scale = std::max(scaleW, scaleH);

            // 计算裁剪区域
            int cropHeight = static_cast<int>(targetH / scale);
            int cropWidth = static_cast<int>(targetW / scale);
            cv::Size cropSize = cv::Size(cropWidth, cropHeight);

            cv::Mat processedImage = img.clone();

            // 裁剪
            cv::Rect cropRect(0, 0, cropWidth, cropHeight);
            processedImage = processedImage(cropRect);

            cv::Size targetSize = cv::Size(targetW, targetH);
            // 缩放
            cv::resize(processedImage, processedImage, targetSize, 0, 0, cv::INTER_AREA);

            cv::Mat double_temp;
            processedImage.convertTo(double_temp, CV_64FC3); // 先转为double
            double_temp /= 255.0;                            // double精度除法
            double_temp.convertTo(processedImage, CV_32FC3); // 截断为float32
            cv::Mat grayImage;
            cv::cvtColor(processedImage, grayImage, cv::COLOR_BGR2GRAY);
            cv::Mat image = HwcToChw(grayImage);
            ReadImgFileToBuf(image, tensorDescs[i], tensorBufs[i]);
            LOG(INFO) << "read end: " << imgPath;
        }
        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}