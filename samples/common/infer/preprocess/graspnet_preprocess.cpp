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

#include "graspnet_preprocess.h"
#include "utils.h"
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

namespace Infer {
namespace Grasp {

constexpr int BYTE_BIT_NUM = 8;
constexpr int COLOR_IDX = 0;
constexpr int DEPTH_IDX = 1;
constexpr int WORKSPACE_MASK_IDX = 2;
constexpr int POINT_DIMENSION = 3;
constexpr int INTRINSIC_SIZE = 3;
constexpr int INPUT_IMAGE_NUM = 3;

// 向量类
struct Vector3d {
    double x, y, z;

    Vector3d() : x(0), y(0), z(0) {}
    Vector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector3d operator+(const Vector3d& other) const
    {
        return Vector3d(x + other.x, y + other.y, z + other.z);
    }

    Vector3d operator-(const Vector3d& other) const
    {
        return Vector3d(x - other.x, y - other.y, z - other.z);
    }

    Vector3d operator*(double scalar) const
    {
        return Vector3d(x * scalar, y * scalar, z * scalar);
    }
};

struct CameraInfo {
    float width;  // 图像宽度（像素）
    float height; // 图像高度（像素）
    float fx;     // 相机在x轴方向的焦距（像素单位），计算公式：fx = f * sx，其中f是物理焦距，sx是像素尺寸
    float fy;     // 相机在y轴方向的焦距（像素单位），计算公式：fy = f * sy
    float cx;     // 主点在x轴方向的坐标（像素），通常接近width/2，表示光学中心在图像中的水平位置
    float cy;     // 主点在y轴方向的坐标（像素），通常接近height/2，表示光学中心在图像中的垂直位置
    float scale;  // 将深度图像像素值转换为真实世界距离的缩放因子，深度值除以这个值得到真实深度

    CameraInfo(float w, float h, float fxVal, float fyVal, float cxVal, float cyVal, float s) :
        width(w), height(h), fx(fxVal), fy(fyVal), cx(cxVal), cy(cyVal), scale(s) {}
};

class FloatCloudWrapper {
private:
    float* data_;
    size_t numPoints_;

public:
    // 构造函数: 从vector<Vector3d>转换
    FloatCloudWrapper(const std::vector<Vector3d>& cloud) : numPoints_(cloud.size())
    {
        data_ = new float[numPoints_ * POINT_DIMENSION];

        for (size_t i = 0; i < numPoints_; ++i) {
            data_[i * POINT_DIMENSION] = static_cast<float>(cloud[i].x);
            data_[i * POINT_DIMENSION + 1] = static_cast<float>(cloud[i].y);
            data_[i * POINT_DIMENSION + 2] = static_cast<float>(cloud[i].z);
        }
    }

    // 析构函数
    ~FloatCloudWrapper()
    {
        delete[] data_;
    }

    // 获取原始指针
    float* GetData() { return data_; }
    const float* GetData() const { return data_; }

    // 获取总float数
    size_t GetFloatCount() const { return numPoints_ * POINT_DIMENSION; }
};

struct PointsCloud {
    std::vector<Vector3d> points;
    std::vector<Vector3d> colors;
};

static struct CameraInfo LoadCameraParam(const std::string& metaPath, int imgWidth, int imgHeight)
{
    auto cfg = ReadCfgFile("../data/camera_params.txt");
    // 创建保存目录
    if (cfg.size() == 0) {
        throw std::runtime_error("Failed to load camera param, please run\
            'python3 ../script/graspnet_parse_cam_param.py' first");
    }
    double fx = std::stod(cfg["fx"]);
    double fy = std::stod(cfg["fy"]);
    double cx = std::stod(cfg["cx"]);
    double cy = std::stod(cfg["cy"]);
    double scale = std::stod(cfg["depth_scale_factor"]);

        // 创建相机信息
    CameraInfo camera(imgWidth, imgHeight, fx, fy, cx, cy, scale);
    LOG(INFO) << "Camera Info:";
    LOG(INFO) << "Resolution: " << camera.width << "x" << camera.height;
    LOG(INFO) << "Focal Length: (" << camera.fx << ", " << camera.fy << ")";
    LOG(INFO) << "Principal Point: (" << camera.cx << ", " << camera.cy << ")";
    LOG(INFO) << "Depth Scale: " << camera.scale;

    return camera;
}

static bool LoadImages(std::vector<std::string>& fileList, cv::Mat& color, cv::Mat& depth, cv::Mat& workspaceMask)
{
    if (fileList.size() != INPUT_IMAGE_NUM) {
        LOG(ERROR) << "Error: FileList size is " << fileList.size() << " , which need to be equal to 3";
        return false;
    }
    color = cv::imread(fileList[COLOR_IDX], cv::IMREAD_COLOR);
    depth = cv::imread(fileList[DEPTH_IDX], cv::IMREAD_ANYDEPTH);
    workspaceMask = cv::imread(fileList[WORKSPACE_MASK_IDX], cv::IMREAD_GRAYSCALE);
    LOG(INFO) << "Color Image Path: " << fileList[COLOR_IDX];
    LOG(INFO) << "Depth Image Path: " << fileList[DEPTH_IDX];
    LOG(INFO) << "WorkspaceMask Image Path: " << fileList[WORKSPACE_MASK_IDX];

    if (color.empty()) {
        LOG(ERROR) << "Error: Could not load color image from: " << fileList[COLOR_IDX];
        return false;
    }
    if (depth.empty()) {
        LOG(ERROR) << "Error: Could not load depth image from: " << fileList[DEPTH_IDX];
        return false;
    }
    if (workspaceMask.empty()) {
        LOG(ERROR) << "Error: Could not load workspace mask from: " << fileList[WORKSPACE_MASK_IDX];
        return false;
    }

    LOG(INFO) << "Successfully loaded images:";
    LOG(INFO) << "Color: " << color.cols << "x" << color.rows << " (" << color.channels() << " channels)";
    LOG(INFO) << "Depth: " << depth.cols << "x" << depth.rows << " (" << depth.channels() << " channels)";
    LOG(INFO) << "Mask: " << workspaceMask.cols << "x" << workspaceMask.rows;

    return true;
}

// 从深度图提取点云
static std::shared_ptr<PointsCloud> CreatePointCloudFromDepth(const cv::Mat& depth, const CameraInfo& camera)
{
    auto cloud = std::make_shared<PointsCloud>();

    int height = depth.rows;
    int width = depth.cols;

    std::vector<Vector3d> points;
    points.reserve(height * width);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float depthVal = depth.at<uint16_t>(y, x);
            float z = depthVal / camera.scale;
            float xCoord = (x - camera.cx) * z / camera.fx;
            float yCoord = (y - camera.cy) * z / camera.fy;
            
            points.push_back(Vector3d(xCoord, yCoord, z));
        }
    }

    cloud->points = points;
    return cloud;
}

static Vector3d ConvertColor(const cv::Vec3f& color)
{
    // OpenCV是BGR格式，转换为RGB
    return Vector3d(color[2], color[1], color[0]);
}

static void ApplyMask(const cv::Mat& depth, const cv::Mat& workspaceMask, const std::shared_ptr<PointsCloud>& cloud,
    const cv::Mat& color, std::vector<Vector3d>& cloudMasked, std::vector<Vector3d>& colorMasked)
{
    int height = depth.rows;
    int width = depth.cols;

    size_t pointIndex = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool maskValid = workspaceMask.at<uint8_t>(y, x) > 0;
            bool depthValid = depth.at<uint16_t>(y, x) > 0;

            if (maskValid && depthValid && pointIndex < cloud->points.size()) {
                // 添加点坐标
                const auto& point = cloud->points[pointIndex];
                cloudMasked.push_back(Vector3d(
                    static_cast<float>(point.x),
                    static_cast<float>(point.y), 
                    static_cast<float>(point.z)
                ));

                // 添加颜色
                cv::Vec3f colorPixel = color.at<cv::Vec3f>(y, x);
                colorMasked.push_back(ConvertColor(colorPixel));
            }
            pointIndex++;
        }
    }
}

static void LoadRandomData(std::vector<size_t>& indices, int numPoints, int isTest)
{
    std::ifstream infile("../data/random_idx.txt"); // 打开文件

    if (!infile.is_open() || isTest == 0) {
        LOG(INFO) << "Do not load random data";
        return;
    }
    LOG(INFO) << "Load test data success";

    for (int i = 0; i < numPoints; i++) {
        infile >> indices[i];
    }
    infile.close();
}

static void SamplePoints(const std::vector<Vector3d>& cloudMasked, 
    const std::vector<Vector3d>& colorMasked, int numPoints, std::vector<Vector3d>& cloudSampled,
    std::vector<Vector3d>& colorSampled, int isTest)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    if (cloudMasked.size() >= numPoints) {
        // 随机选择不重复的点
        std::vector<size_t> indices(cloudMasked.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        // 随机排列索引，然后取前numPoints个值
        std::shuffle(indices.begin(), indices.end(), gen);

        // isTest表示是否在点云采样时采用固定采样，1表示固定采样，0表示随机采样 
        // 验证模型精度时可将/data/cfg.txt中设置为is_test=1，并将结果与/data/use_for_precision_align.txt进行比对
        LoadRandomData(indices, numPoints, isTest);

        for (int i = 0; i < numPoints; ++i) {
            cloudSampled.push_back(cloudMasked[indices[i]]);
            colorSampled.push_back(colorMasked[indices[i]]);
        }
    } else {
        // 如果点数不够，先添加所有点，然后随机重复采样
        cloudSampled = cloudMasked;
        colorSampled = colorMasked;

        int remaining = numPoints - cloudMasked.size();
        std::uniform_int_distribution<size_t> dis(0, cloudMasked.size() - 1);

        for (int i = 0; i < remaining; ++i) {
            size_t idx = dis(gen);
            cloudSampled.push_back(cloudMasked[idx]);
            colorSampled.push_back(colorMasked[idx]);
        }
    }
}

static void SaveVectorToBinary(const std::vector<Vector3d>& points, const std::string& fileName)
{
    std::ofstream saveBin(fileName, std::ios::binary);
    if (!saveBin) {
        LOG(ERROR) << "Cannot open file for writing: " << fileName;
        return;
    }

    // 先写入点的数量
    size_t numPoints = points.size();
    saveBin.write(reinterpret_cast<const char*>(&numPoints), sizeof(size_t));

    // 写入所有点的数据
    saveBin.write(reinterpret_cast<const char*>(points.data()), 
        numPoints * sizeof(Vector3d));

    saveBin.close();
}

static void SavePointCloud(const std::string& saveBinDir, const std::vector<Vector3d>& points,
    const std::vector<Vector3d>& colors)
{
    std::string pointsFileName = saveBinDir + "cloud_points.bin";
    std::string colorFileName = saveBinDir + "cloud_color.bin";
    SaveVectorToBinary(points, pointsFileName);
    SaveVectorToBinary(colors, colorFileName);
}

static FloatCloudWrapper ProcessData(std::vector<std::string>& fileList, const CameraInfo& camera,
    int numPoints, const std::string& saveBinDir, int isTest)
{
    // 加载图像
    cv::Mat color, depth, workspaceMask;
    if (!LoadImages(fileList, color, depth, workspaceMask)) {
        throw std::runtime_error("Failed to load input images");
    }

    // 转换颜色图像到float类型并归一化
    cv::Mat colorFloat;
    color.convertTo(colorFloat, CV_32FC3, 1.0/255.0);

    // 生成点云
    auto cloudFull = CreatePointCloudFromDepth(depth, camera);
    if (!cloudFull || cloudFull->points.empty()) {
        throw std::runtime_error("Failed to generate point cloud from depth image");
    }
    LOG(INFO) << "Generated full point cloud with " << cloudFull->points.size() << " points";

    // 获取有效点
    std::vector<Vector3d> cloudMasked;
    std::vector<Vector3d> colorMasked;
    ApplyMask(depth, workspaceMask, cloudFull, colorFloat, cloudMasked, colorMasked);
    LOG(INFO) << "After masking: " << cloudMasked.size() << " points";

    // 采样点
    std::vector<Vector3d> cloudSampled;
    std::vector<Vector3d> colorSampled;
    SamplePoints(cloudMasked, colorMasked, numPoints, cloudSampled, colorSampled, isTest);
    LOG(INFO) << "After sampling: " << cloudSampled.size() << " points";

    // 保存点云
    SavePointCloud(saveBinDir, cloudMasked, colorMasked);

    // 准备结果
    FloatCloudWrapper wrapper(cloudSampled);

    return wrapper;
}

static Result CopyCloudWrapperToBuf(FloatCloudWrapper &wrapper, TensorDesc &desc, TensorBuf &inBuf)
{
    LOG(INFO) << "CopyCloudWrapperToBuf: desc.dimCount " << desc.dimCount;

    const size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
    const float *srcData = wrapper.GetData();
    if (wrapper.GetFloatCount() != desc.defaultSize / dataSize) {
        LOG(ERROR) << "Input points size invalid, wrapper.GetFloatCount = " << wrapper.GetFloatCount();
        return FAILED;
    }
    LOG(INFO) << "CopyCloudWrapperToBuf: desc.dimCount " << desc.dimCount;
    DevMemcpy(inBuf.GetRawPtr(), desc.defaultSize, srcData, desc.defaultSize);
    return SUCCESS;
}

bool GraspNetPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
{
    auto cfg = ReadCfgFile("../data/cfg.txt");
    int numPoints = std::stoi(cfg["num_points"]);
    int imgHeight = std::stoi(cfg["img_height"]);
    int imgWidth = std::stoi(cfg["img_width"]);
    int isTest = std::stoi(cfg["is_test"]);
    std::string saveBinDir = cfg["save_out_bin"];

    LOG(INFO) << "numPoints: " << numPoints;
    LOG(INFO) << "imgWidth: " << imgWidth << " imgHeight: " << imgHeight;

    // 创建保存目录
    if (!saveBinDir.empty() && !CreateDirectoryRecursive(saveBinDir)) {
        LOG(ERROR) << "Create preprocess bin dir failed";
        return false;
    }

    // 加载相机参数
    CameraInfo camera = LoadCameraParam("../data/camera_params.txt", imgWidth, imgHeight);

    // 处理数据
    FloatCloudWrapper floatCloud = ProcessData(fileList, camera, numPoints, saveBinDir, isTest);
    CopyCloudWrapperToBuf(floatCloud, inDescs[0], inBufs[0]);
    return true;
}

}
}