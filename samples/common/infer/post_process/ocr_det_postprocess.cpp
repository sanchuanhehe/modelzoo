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

#include "ocr_det_postprocess.h"
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
#include <nlohmann/json.hpp>
#include <map>

using namespace std;
using json = nlohmann::json;

namespace Infer
{
    using namespace std;
    constexpr float CONF_THRES = 0.3f;
    constexpr float BOX_THRES = 0.6f;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr int MAX_CANDIDATES = 1000;
    constexpr float UNCLIP_RATIO = 1.5;
    constexpr int IMG_SIZE = 960;
    constexpr int MIN_SIZE = 3;
    constexpr int DOUBLE = 2;

    struct DetectionResult
    {
        std::vector<std::vector<cv::Point>> boxes;
        std::vector<float> scores;
    };

    static json PointsToJson(const std::vector<cv::Point> &points)
    {
        json points_array = json::array();
        for (const auto &point : points) {
            json point_array = json::array();
            point_array.push_back(point.x);
            point_array.push_back(point.y);
            points_array.push_back(point_array);
        }
        return points_array;
    }

    static json ConvertToJson(const std::vector<std::vector<cv::Point>> &dtBoxes,
                              const std::string &image_file)
    {
        json resultArray = json::array();
        json singleResult;

        // 设置图像名称
        singleResult["img_name"] = image_file;

        // 创建检测信息列表
        json detList = json::array();
        for (const auto &box : dtBoxes) {
            json detInfo;
            detInfo["points"] = PointsToJson(box);
            detInfo["text"] = "";
            detList.push_back(detInfo);
        }

        singleResult["pre"] = detList;
        resultArray.push_back(singleResult);

        return resultArray;
    }

    // 计算多边形面积
    static double PolygonArea(const std::vector<cv::Point> &contour)
    {
        double area = 0.0;
        int n = contour.size();
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            area += (double)contour[i].x * contour[j].y - (double)contour[j].x * contour[i].y;
        }
        return std::abs(area) / 2.0;
    }

    // 计算多边形周长
    static double PolygonLength(const std::vector<cv::Point> &contour)
    {
        double length = 0.0;
        int n = contour.size();
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            double dx = (double)contour[j].x - contour[i].x;
            double dy = (double)contour[j].y - contour[i].y;
            length += std::sqrt(dx * dx + dy * dy);
        }
        return length;
    }

    // 获取最小外接矩形框
    static std::pair<std::vector<cv::Point>, double> GetMiniBoxes(const std::vector<cv::Point> &contour)
    {
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        cv::Point2f vertices[4];
        rotated_rect.points(vertices);

        std::vector<cv::Point> points;
        for (int i = 0; i < 4; ++i) {
            points.push_back(cv::Point(round(vertices[i].x), round(vertices[i].y)));
        }

        // 排序点
        std::sort(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b)
                  { return a.x < b.x; });

        std::vector<cv::Point> box(4);
        if (points[1].y > points[0].y) {
            box[0] = points[0];
            box[3] = points[1];
        } else {
            box[0] = points[1];
            box[3] = points[0];
        }

        if (points[3].y > points[2].y) {
            box[1] = points[2];
            box[2] = points[3];
        } else {
            box[1] = points[3];
            box[2] = points[2];
        }

        double sside = std::min(rotated_rect.size.width, rotated_rect.size.height);
        return {box, sside};
    }

    // 计算两点间距离
    static double PointDistance(const cv::Point &p1, const cv::Point &p2)
    {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    // 计算单位法向量
    static cv::Point2f GetUnitNormal(const cv::Point &a, const cv::Point &b)
    {
        double dx = b.x - a.x;
        double dy = b.y - a.y;
        double length = std::sqrt(dx * dx + dy * dy);
        if (length == 0)
            return cv::Point2f(0, 0);
        return cv::Point2f(-dy / length, dx / length);
    }

    // 判断多边形方向（顺时针为true）
    static bool IsClockwise(const std::vector<cv::Point> &polygon)
    {
        double sum = 0.0;
        int n = polygon.size();
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            sum += (polygon[j].x - polygon[i].x) * (polygon[j].y + polygon[i].y);
        }
        return sum > 0;
    }

    // 实现多边形偏移
    static std::vector<std::vector<cv::Point>> Unclip(const std::vector<cv::Point> &box, float unclip_ratio)
    {
        if (box.size() < 3) {
            return {box};
        }

        // 1. 计算偏移距离（与Python完全一致）
        double area = PolygonArea(box);
        double length = PolygonLength(box);
        double distance = area * unclip_ratio / length;

        LOG(INFO) << "Area: " << area << ", Length: " << length << ", Distance: " << distance;

        // 2. 确定多边形方向并反转（pyclipper要求顺时针）
        std::vector<cv::Point> polygon = box;
        bool clockwise = IsClockwise(polygon);
        if (!clockwise) {
            std::reverse(polygon.begin(), polygon.end());
        }

        // 3. 创建偏移后的点集
        std::vector<cv::Point> result;
        int n = polygon.size();

        // 对于每条边，创建偏移点
        std::vector<cv::Point> offsets;
        for (int i = 0; i < n; i++) {
            cv::Point p1 = polygon[i];
            cv::Point p2 = polygon[(i + 1) % n];

            // 获取单位法向量（指向外部）
            cv::Point2f normal = GetUnitNormal(p1, p2);

            // 偏移点
            cv::Point offset1, offset2;
            offset1.x = static_cast<int>(p1.x + normal.x * distance);
            offset1.y = static_cast<int>(p1.y + normal.y * distance);
            offset2.x = static_cast<int>(p2.x + normal.x * distance);
            offset2.y = static_cast<int>(p2.y + normal.y * distance);

            offsets.push_back(offset1);
            offsets.push_back(offset2);
        }

        // 4. 处理圆角连接（JT_ROUND）
        for (int i = 0; i < n; i++) {
            int idx1 = i * DOUBLE;
            int idx2 = (i * DOUBLE + 1) % (DOUBLE * n);
            int idx3 = (i * DOUBLE + DOUBLE) % (DOUBLE * n);

            cv::Point p1 = offsets[idx1];
            cv::Point p2 = offsets[idx2];
            cv::Point p3 = offsets[idx3];

            // 添加第一个偏移点
            result.push_back(p1);

            // 计算圆角（添加插值点）
            int segments = 10; // 圆角细分段数
            cv::Point center = polygon[(i + 1) % n];

            for (int j = 1; j <= segments; j++) {
                double t = static_cast<double>(j) / segments;
                double angle1 = atan2(p2.y - center.y, p2.x - center.x);
                double angle2 = atan2(p3.y - center.y, p3.x - center.x);

                // 确保角度差在合理范围内
                if (angle1 < 0)
                    angle1 += DOUBLE * M_PI;
                if (angle2 < 0)
                    angle2 += DOUBLE * M_PI;

                double angle_diff = angle2 - angle1;
                if (angle_diff < 0)
                    angle_diff += DOUBLE * M_PI;
                if (angle_diff > M_PI)
                    angle_diff -= DOUBLE * M_PI;

                double current_angle = angle1 + angle_diff * t;
                cv::Point rounded_point;
                rounded_point.x = static_cast<int>(center.x + distance * cos(current_angle));
                rounded_point.y = static_cast<int>(center.y + distance * sin(current_angle));

                result.push_back(rounded_point);
            }
        }

        // 5. 移除距离太近的点（简化多边形）
        std::vector<cv::Point> simplified;
        for (size_t i = 0; i < result.size(); i++) {
            if (simplified.empty() ||
                PointDistance(result[i], simplified.back()) > 1.0) {
                simplified.push_back(result[i]);
            }
        }

        // 移除首尾重复点
        if (!simplified.empty() &&
            PointDistance(simplified.front(), simplified.back()) < 1.0) {
            simplified.pop_back();
        }

        // 如果方向被反转过，再反转回来
        if (!clockwise) {
            std::reverse(simplified.begin(), simplified.end());
        }
        return {simplified};
    }

    // 快速框评分
    static float BoxScoreFast(const cv::Mat &bitmap, const std::vector<cv::Point> &box)
    {
        LOG(INFO) << "BoxScoreFast";
    
        int h = bitmap.rows;
        int w = bitmap.cols;

        int xmin = w - 1, xmax = 0, ymin = h - 1, ymax = 0;
        for (const auto &point : box) {
            xmin = std::min(xmin, point.x);
            xmax = std::max(xmax, point.x);
            ymin = std::min(ymin, point.y);
            ymax = std::max(ymax, point.y);
        }

        xmin = std::max(0, std::min(xmin, w - 1));
        xmax = std::max(0, std::min(xmax, w - 1));
        ymin = std::max(0, std::min(ymin, h - 1));
        ymax = std::max(0, std::min(ymax, h - 1));

        cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        std::vector<cv::Point> adjusted_box;
        for (const auto &point : box) {
            adjusted_box.push_back(cv::Point2f(point.x - xmin, point.y - ymin));
        }

        std::vector<std::vector<cv::Point>> contours = {adjusted_box};
        cv::fillPoly(mask, contours, cv::Scalar(1));

        cv::Mat roi = bitmap(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1));
        cv::Scalar mean_val = cv::mean(roi, mask);
        return static_cast<float>(mean_val[0]);
    }

    static DetectionResult BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                                           vector<int> &shapeList)
    {
        DetectionResult result;
        LOG(INFO) << " BoxesFromBitmap ";
        cv::Mat bitmapUint8;
        bitmap.convertTo(bitmapUint8, CV_8UC1, 255.0); // 缩放并转换类型

        // 2. 定义轮廓和层级结构
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat hierarchy;

        // 3. 调用findContours，对应Python的cv2.RETR_LIST和cv2.CHAIN_APPROX_SIMPLE
        // OpenCV 3.x返回3个值：(img, contours, hierarchy)
        // OpenCV 4.x及以上返回2个值：(contours, hierarchy)
        cv::findContours(bitmapUint8, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        int num_contours = std::min(static_cast<int>(contours.size()), MAX_CANDIDATES);
       
        for (int index = 0; index < num_contours; ++index) {
            auto boxResult = GetMiniBoxes(contours[index]);
            auto points = std::get<0>(boxResult);
            auto sside = std::get<1>(boxResult);
            if (sside < MIN_SIZE) {
                continue;
            }

            float score;
            score = BoxScoreFast(pred, points);
           
            if (score < BOX_THRES) {
                continue;
            }
            
            auto expanded_boxes = Unclip(points, UNCLIP_RATIO);
            if (expanded_boxes.empty()) {
                continue;
            }
            auto expandedResult = GetMiniBoxes(expanded_boxes[0]);
            auto box = std::get<0>(expandedResult);
            auto newSside = std::get<1>(expandedResult);
            if (newSside < MIN_SIZE + 2) {
                continue;
            }

            // 坐标映射回原图
            double bitmapWidth = bitmap.cols;
            double bitmapHeight = bitmap.rows;
            int destWidth = shapeList[0];
            int destHeight = shapeList[1];
           
            for (auto &point : box) {
                point.x = std::max(0, std::min(destWidth, static_cast<int>(std::round(point.x /
                    (bitmapWidth - shapeList[2]) * destWidth))));
                point.y = std::max(0, std::min(destHeight, static_cast<int>(std::round(point.y /
                    (bitmapHeight - shapeList[3]) * destHeight))));
            }
            result.boxes.push_back(box);
            result.scores.push_back(score);
        }
        return result;
    }

    static void SaveResult(std::vector<std::vector<cv::Point>> bboxs, std::vector<TensorBuf> &tensorBufs,
        std::vector<TensorDesc> &tensorDescs, const string &filePath)
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
        for (size_t j = 0; j < tensorBufs.size(); j++)
        {
            string outputFileName = binPath + outputName + "_" + to_string(j) + ".bin";
            LOG(INFO) << " outputFileName: " << outputFileName;
            ofstream binfout(outputFileName, ios::out | ios::trunc);
            binfout.write(static_cast<char *>(tensorBufs[j].GetRawPtr()), tensorBufs[j].size);
            binfout.close();
        }

        string jsonFileNmae = txtPath + outputName + ".json";
        json jsonResult = ConvertToJson(bboxs, outputName.substr(1) + ".jpg");
        // 保存新文件
        ofstream jsonFile(jsonFileNmae);

        jsonFile << jsonResult.dump(4);
        jsonFile.close();
    }

    // 缩放边界框
    static void GetShapeInfo(vector<int> &shapeList, const string &filePath)
    {
        LOG(INFO) << "filePath: " << filePath;
        cv::Mat im0 = cv::imread(filePath);

        // 获取原始尺寸
        cv::Size shape = im0.size();
        int h0 = shape.height;
        int w0 = shape.width;

        // 计算缩放比例
        double r = std::min(static_cast<double>(IMG_SIZE) / shape.width,
                            static_cast<double>(IMG_SIZE) / shape.height);

        // 计算未填充的尺寸
        int resizeW = static_cast<int>(std::round(shape.width * r));
        int resizeH = static_cast<int>(std::round(shape.height * r));
        resizeW = std::max(static_cast<int>(std::round(resizeW / 32.0)) * 32, 32);
        resizeH = std::max(static_cast<int>(std::round(resizeH / 32.0)) * 32, 32);
        shapeList.push_back(w0);
        shapeList.push_back(h0);
        shapeList.push_back(IMG_SIZE - resizeW);
        shapeList.push_back(IMG_SIZE - resizeH);
    }

    static cv::Mat GetPreMat(TensorDesc desc, TensorBuf buf)
    {
        cv::Mat pre;
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        if (desc.dimCount < 2) {
            return pre;
        }
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }
        int64_t height = desc.dims[desc.dimCount - 2]; /* dims last dim is width */
        int64_t width = desc.dims[desc.dimCount - 1];  /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = buf.stride / dataSize;
        LOG(INFO) << "dataSize:  " << buf.size;
        if (width == strideElemNum) {
            LOG(INFO) << "no stride ";
            float *data_ptr = static_cast<float *>(buf.GetRawPtr());
            if (data_ptr == nullptr) {
                LOG(INFO) << "no data_ptr ";
            }
            cv::Mat mat(height, width, CV_32FC1, data_ptr);
            return mat;
        } else {
            float *temp = new float[loopTimes * width];
            for (int64_t loop = 0; loop < loopTimes; loop++) {
                float *srcPtr = static_cast<float *>(buf.GetRawPtr()) + loop * strideElemNum;
                float *destPtr = temp + loop * width;
                // 检查指针有效性
                if (destPtr == nullptr || srcPtr == nullptr) {
                    LOG(ERROR) << "Null pointer detected";
                    return pre;
                }
                // 拷贝数据
                memcpy(destPtr, srcPtr, width * dataSize);
            }
            cv::Mat mat(height, width, CV_32FC1, temp);
            return mat;
        }
    }

    static std::vector<cv::Point> OrderPointsClockwise(const std::vector<cv::Point> &pts)
    {
        if (pts.size() != 4) {
            throw std::invalid_argument("Input must contain exactly 4 points");
        }

        std::vector<cv::Point> rect(4);
        std::vector<float> sums;
        std::vector<float> diffs;

        // 步骤1: 计算每个点的x+y和
        for (const auto &pt : pts) {
            sums.push_back(pt.x + pt.y);
        }

        // 找到和最小和最大的索引
        auto minSumIter = std::min_element(sums.begin(), sums.end());
        auto maxSumIter = std::max_element(sums.begin(), sums.end());
        int minIdx = std::distance(sums.begin(), minSumIter);
        int maxIdx = std::distance(sums.begin(), maxSumIter);

        rect[0] = pts[minIdx]; // 左上角
        rect[2] = pts[maxIdx]; // 右下角

        // 步骤2: 找到剩余的两个点
        std::vector<cv::Point> tmp;
        for (int i = 0; i < 4; ++i) {
            if (i != minIdx && i != maxIdx) {
                tmp.push_back(pts[i]);
            }
        }

        // 步骤3: 计算剩余两个点的y-x差
        for (const auto &pt : tmp) {
            diffs.push_back(pt.y - pt.x);
        }

        // 找到差最小和最大的索引
        auto minDiffIter = std::min_element(diffs.begin(), diffs.end());
        auto maxDiffIter = std::max_element(diffs.begin(), diffs.end());
        int minDiffIdx = std::distance(diffs.begin(), minDiffIter);
        int maxDiffIdx = std::distance(diffs.begin(), maxDiffIter);

        rect[1] = tmp[minDiffIdx]; // 右上角
        rect[3] = tmp[maxDiffIdx]; // 左下角

        return rect;
    }

    std::vector<cv::Point> ClipDetRes(const std::vector<cv::Point> &points,
                                      int img_height, int img_width)
    {
        std::vector<cv::Point> clippedPoints = points;

        for (auto &point : clippedPoints) {
            point.x = std::max(0, std::min(point.x, img_width - 1));
            point.y = std::max(0, std::min(point.y, img_height - 1));
        }

        return clippedPoints;
    }

    // 过滤函数
    static std::vector<std::vector<cv::Point>> FilterTagDetRes(
        const std::vector<std::vector<cv::Point>> &dtBoxes,
        const float imgWidth, const float imgHeight)
    {

        std::vector<std::vector<cv::Point>> dtBoxesNew;

        for (const auto &box : dtBoxes) {
            std::vector<cv::Point> processedBox = box;

            // 如果box是列表形式，在C++中我们假设已经是vector格式
            // 所以不需要类型转换步骤

            // 按顺时针排序点
            processedBox = OrderPointsClockwise(processedBox);

            // 裁剪到图像边界
            processedBox = ClipDetRes(processedBox, imgHeight, imgWidth);

            // 计算矩形宽度和高度
            double rectWidth = cv::norm(processedBox[0] - processedBox[1]);
            double rectHeight = cv::norm(processedBox[0] - processedBox[3]);

            // 过滤太小的框
            if (rectWidth <= MIN_SIZE || rectHeight <= MIN_SIZE) {
                continue;
            }

            dtBoxesNew.push_back(processedBox);
        }

        return dtBoxesNew;
    }

    bool OcrDetPostprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs,
        std::vector<TensorDesc> &tensorDescs)
    {
        for (size_t i = 0; i < tensorBufs.size(); i++) {
            std::vector<std::vector<cv::Point>> bbox;
            TensorDesc desc = tensorDescs[i];
            TensorBuf buf = tensorBufs[i];
            if (desc.defaultStride == 0) {
                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
            }
            // 假设pred是4D张量 [batch, channel, height, width]
            // 这里简化处理，实际可能需要根据具体数据结构调整
            vector<int> shapeList;
            GetShapeInfo(shapeList, fileList[0]);
            cv::Mat pred = GetPreMat(desc, buf);
            cv::Mat segmentationMask;
            cv::threshold(pred, segmentationMask, CONF_THRES, 1, cv::THRESH_BINARY);
            segmentationMask.convertTo(segmentationMask, CV_8UC1); // 转为uchar类型
            cv::Mat binaryMap;
            binaryMap.push_back(segmentationMask);

            cv::Mat mask = binaryMap.clone();
            DetectionResult detResult;
            detResult = BoxesFromBitmap(pred, mask, shapeList);
            bbox = FilterTagDetRes(detResult.boxes, shapeList[0], shapeList[1]);

            SaveResult(bbox, tensorBufs, tensorDescs, fileList[0]);
            LOG(INFO) << "dump final data success " << fileList[0];
        }
        return true;
    }
}
