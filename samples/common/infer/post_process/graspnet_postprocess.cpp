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

#include "graspnet_postprocess.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <array>
#include <functional>
#include <unordered_map>

namespace Infer {
namespace Grasp {

constexpr int BYTE_BIT_NUM = 8;
constexpr int GRASP_ARRAY_LEN = 17;
constexpr int POINT_DIMENSION = 3;

struct Vector3i {
    int x_, y_, z_;

    Vector3i() : x_(0), y_(0), z_(0) {}
    Vector3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}

    bool operator==(const Vector3i& other) const
    {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }
};

// 3维向量
struct Vector3d {
    double x_, y_, z_;

    Vector3d() : x_(0), y_(0), z_(0) {}
    Vector3d(double x, double y, double z) : x_(x), y_(y), z_(z) {}

    Vector3d operator+(const Vector3d& other) const
    {
        return Vector3d(x_ + other.x_, y_ + other.y_, z_ + other.z_);
    }

    Vector3d operator-(const Vector3d& other) const
    {
        return Vector3d(x_ - other.x_, y_ - other.y_, z_ - other.z_);
    }

    Vector3d operator*(double scalar) const
    {
        return Vector3d(x_ * scalar, y_ * scalar, z_ * scalar);
    }

    Vector3d operator/(double scalar) const
    {
        return Vector3d(x_ / scalar, y_ / scalar, z_ / scalar);
    }

    Vector3d& operator+=(const Vector3d& other)
    {
        x_ += other.x_; y_ += other.y_; z_ += other.z_;
        return *this;
    }

    bool operator==(const Vector3d& other) const
    {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    double norm() const
    {
        return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
    }
};

// 3X3矩阵
struct Matrix3d {
    std::array<std::array<double, POINT_DIMENSION>, POINT_DIMENSION> data_;

    Matrix3d()
    {
        for (int i = 0; i < POINT_DIMENSION; ++i)
            for (int j = 0; j < POINT_DIMENSION; ++j)
                data_[i][j] = (i == j) ? 1.0 : 0.0;
    }

    double& operator()(int row, int col) { return data_[row][col]; }
    double operator()(int row, int col) const { return data_[row][col]; }

    Matrix3d operator*(const Matrix3d& other) const
    {
        Matrix3d result;
        for (int i = 0; i < POINT_DIMENSION; ++i) {
            for (int j = 0; j < POINT_DIMENSION; ++j) {
                result(i, j) = data_[i][0] * other(0, j) + 
                    data_[i][1] * other(1, j) + data_[i][2] * other(2, j);
            }
        }
        return result;
    }

    Vector3d operator*(const Vector3d& vec) const
    {
        return Vector3d(
            data_[0][0] * vec.x_ + data_[0][1] * vec.y_ + data_[0][2] * vec.z_,
            data_[1][0] * vec.x_ + data_[1][1] * vec.y_ + data_[1][2] * vec.z_,
            data_[2][0] * vec.x_ + data_[2][1] * vec.y_ + data_[2][2] * vec.z_);
    }

    Matrix3d transpose() const
    {
        Matrix3d result;
        for (int i = 0; i < POINT_DIMENSION; ++i)
            for (int j = 0; j < POINT_DIMENSION; ++j)
                result(i, j) = data_[j][i];
        return result;
    }

    double trace() const
    {
        return data_[0][0] + data_[1][1] + data_[2][2];
    }
};

// 动态大小的列向量
class VectorXd {
private:
    std::vector<double> data_;

public:
    VectorXd() = default;
    explicit VectorXd(size_t size) : data_(size, 0.0) {}
    explicit VectorXd(const std::vector<double>& data) : data_(data) {}

    size_t size() const { return data_.size(); }
    double& operator[](size_t i) { return data_[i]; }
    const double& operator[](size_t i) const { return data_[i]; }

    void resize(size_t size) { data_.resize(size, 0.0); }
    void push_back(double value) { data_.push_back(value); }

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }
};

// 动态大小的矩阵
class MatrixXd {
private:
    std::vector<double> data_;  // 连续内存存储
    size_t rows_, cols_;

public:
    MatrixXd() : rows_(0), cols_(0) {}
    MatrixXd(size_t rows, size_t cols) : rows_(rows), cols_(cols)
    {
        data_.resize(rows * cols, 0.0);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double& operator()(size_t row, size_t col)
    {
        return data_[row * cols_ + col];
    }

    const double& operator()(size_t row, size_t col) const
    {
        return data_[row * cols_ + col];
    }

    // 支持数组式访问的便捷方法
    double* operator[](size_t row) { return &data_[row * cols_]; }
    const double* operator[](size_t row) const { return &data_[row * cols_]; }

    void resize(size_t rows, size_t cols)
    {
        rows_ = rows; cols_ = cols;
        data_.resize(rows * cols, 0.0);
    }

    MatrixXd block(size_t startRow, size_t startCol, size_t blockRows, size_t blockCols) const
    {
        MatrixXd result(blockRows, blockCols);
        for (size_t i = 0; i < blockRows; ++i) {
            for (size_t j = 0; j < blockCols; ++j) {
                result(i, j) = (*this)(startRow + i, startCol + j);
            }
        }
        return result;
    }

    std::vector<double> col(size_t colIdx) const
    {
        std::vector<double> column(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            column[i] = (*this)(i, colIdx);
        }
        return column;
    }

    // 获取连续数据指针（便于批量操作）
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
};

class PointsCloud {
public:
    PointsCloud() {}
    ~PointsCloud() {}
    std::shared_ptr<PointsCloud> VoxelDownSample(double voxelSize) const;
private:
    Vector3d ComputeMinBound(const std::vector<Vector3d>& points) const;
    Vector3d ComputeMaxBound(const std::vector<Vector3d>& points) const;
public:
    std::vector<Vector3d> points_;
    std::vector<Vector3d> colors_;
};

class GraspGroup {
public:
    GraspGroup();
    explicit GraspGroup(const MatrixXd& graspArray);
    explicit GraspGroup(const std::string& filename);
    explicit GraspGroup(std::vector<TensorBuf>& outBufs, std::vector<TensorDesc>& outDescs);

    // 属性访问器
    VectorXd Scores() const;
    VectorXd Widths() const;
    VectorXd Heights() const;
    VectorXd Depths() const;
    std::vector<Matrix3d> RotationMatrices() const;
    MatrixXd Translations() const;
    VectorXd ObjectIds() const;

    // 功能方法
    int Size() const;
    bool Empty() const;

    void SortByScore(bool reverse = false);
    GraspGroup Nms(double translationThresh = 0.03, double rotationThresh = 30.0 / 180.0 * M_PI) const;

    // 操作符重载
    GraspGroup operator[](const std::vector<int>& indices) const;

    void Save(const std::string& filename) const;
    void LoadModelOutput(TensorBuf &outBufs, TensorDesc &outDescs);

private:
    MatrixXd graspGroupArray_;

    void ValidateIndices(const std::vector<int>& indices) const;
};

class ModelFreeCollisionDetector {
public:
    ModelFreeCollisionDetector(const MatrixXd& scenePoints, double voxelSize = 0.01);

    struct DetectionResult {
        std::vector<bool> collisionMask;
        std::vector<bool> emptyMask;
        std::vector<double> globalIou;
        std::vector<double> leftIou;
        std::vector<double> rightIou;
        std::vector<double> bottomIou;
        std::vector<double> shiftingIou;
    };

    DetectionResult Detect(const GraspGroup& graspGroup, double approachDist = 0.05, 
        double collisionThresh = 0.01, bool returnEmptyGrasp = false, double emptyThresh = 0.01,
        bool returnIous = false);

private:
    double fingerWidth_ = 0.01;
    double fingerLength_ = 0.06;
    double voxelSize_;
    MatrixXd scenePoints_;
};

GraspGroup::GraspGroup() : graspGroupArray_(MatrixXd(0, GRASP_ARRAY_LEN)) {}

GraspGroup::GraspGroup(const MatrixXd& graspArray) : graspGroupArray_(graspArray)
{
    if (graspArray.cols() != GRASP_ARRAY_LEN) {
        throw std::invalid_argument("Grasp array must have " + std::to_string(GRASP_ARRAY_LEN) + " columns");
    }
}

GraspGroup::GraspGroup(std::vector<TensorBuf>& outBufs, std::vector<TensorDesc>& outDescs)
{
    LoadModelOutput(outBufs[0], outDescs[0]);
}

VectorXd GraspGroup::Scores() const
{
    auto col = graspGroupArray_.col(0);
    return VectorXd(col);
}

VectorXd GraspGroup::Widths() const
{
    auto col = graspGroupArray_.col(1);
    return VectorXd(col);
}

VectorXd GraspGroup::Heights() const
{
    auto col = graspGroupArray_.col(2);
    return VectorXd(col);
}

VectorXd GraspGroup::Depths() const
{
    auto col = graspGroupArray_.col(POINT_DIMENSION);
    return VectorXd(col);
}

std::vector<Matrix3d> GraspGroup::RotationMatrices() const
{
    std::vector<Matrix3d> rotations;
    int n = Size();
    rotations.reserve(n);

    for (int i = 0; i < n; ++i) {
        Matrix3d rot;
        for (int j = 0; j < POINT_DIMENSION; ++j) {
            for (int k = 0; k < POINT_DIMENSION; ++k) {
                rot(j, k) = graspGroupArray_[i][4 + j * POINT_DIMENSION + k];
            }
        }
        rotations.push_back(rot);
    }
    return rotations;
}

MatrixXd GraspGroup::Translations() const 
{
    return graspGroupArray_.block(0, 13, Size(), POINT_DIMENSION);
}

VectorXd GraspGroup::ObjectIds() const
{
    auto col = graspGroupArray_.col(16);
    return VectorXd(col);
}

int GraspGroup::Size() const
{
    return graspGroupArray_.rows();
}

void GraspGroup::SortByScore(bool reverse)
{
    int n = Size();
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return reverse ? graspGroupArray_[i][0] < graspGroupArray_[j][0]
                      : graspGroupArray_[i][0] > graspGroupArray_[j][0];
    });

    MatrixXd sortedArray(n, GRASP_ARRAY_LEN);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GRASP_ARRAY_LEN; ++j) {
            sortedArray[i][j] = graspGroupArray_[indices[i]][j];
        }
    }
    graspGroupArray_ = sortedArray;
}

// 非极大值抑制（NMS） 算法
static MatrixXd NmsGrasp(const MatrixXd& graspGroupArray, double translationThresh,
    double rotationThresh)
{
    // 非极大值抑制实现
    int numGrasps = graspGroupArray.rows();
    if (numGrasps == 0) {
        return graspGroupArray;
    }
    // keep数组：用于标记每个抓取是否保留（初始全为true）
    std::vector<bool> keep(numGrasps, true);
    // indices数组：存储抓取的索引，用于排序
    std::vector<int> indices(numGrasps);
    for (int i = 0; i < numGrasps; ++i) {
        indices[i] = i;
    }
    // 按抓取分数降序排序
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return graspGroupArray[i][0] > graspGroupArray[j][0];
    });
    // 遍历排序后的抓取，剔除冗余的
    for (size_t i = 0; i < indices.size(); ++i) {
        // 如果当前抓取已被标记为剔除，跳过
        if (!keep[indices[i]]) continue;
        // 提取第i个抓取的平移向量（位置）：矩阵第13、14、15列
        Vector3d transI(
            graspGroupArray[indices[i]][13],
            graspGroupArray[indices[i]][14],
            graspGroupArray[indices[i]][15]
        );
        // 提取第i个抓取的旋转矩阵：矩阵第4-12列（3x3旋转矩阵）
        Matrix3d rotI;
        for (int j = 0; j < POINT_DIMENSION; ++j) {
            for (int k = 0; k < POINT_DIMENSION; ++k) {
                rotI(j, k) = graspGroupArray[indices[i]][4 + j * POINT_DIMENSION + k];
            }
        }
        // 遍历i之后的所有抓取（分数更低的）
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (!keep[indices[j]]) continue;
            // 提取第j个抓取的平移向量
            Vector3d transJ(
                graspGroupArray[indices[j]][13],
                graspGroupArray[indices[j]][14],
                graspGroupArray[indices[j]][15]
            );
            // 计算平移距离（欧氏距离）
            Vector3d diff = transI - transJ;
            double transDist = diff.norm();
            // 如果平移距离小于阈值，进一步检查旋转角度
            if (transDist < translationThresh) {
                Matrix3d rotJ;
                for (int k = 0; k < POINT_DIMENSION; ++k) {
                    for (int l = 0; l < POINT_DIMENSION; ++l) {
                        rotJ(k, l) = graspGroupArray[indices[j]][4 + k * POINT_DIMENSION + l];
                    }
                }
                // 计算旋转角度差异
                // rotI * rotJ^T 得到旋转差异矩阵，通过trace计算旋转角度
                Matrix3d rotDiff = rotI * rotJ.transpose();
                double trace = rotDiff.trace(); // 矩阵迹：对角线元素之和
                // 计算旋转角度（弧度）：利用旋转矩阵迹与角度的关系 tr(R) = 1 + 2cosθ
                double angle = std::acos(std::min(1.0, std::max(-1.0, (trace - 1) / 2)));
                // 旋转角度小于阈值 → 标记为剔除
                if (angle < rotationThresh) {
                    keep[indices[j]] = false;
                }
            }
        }
    }
    // 统计需要保留的抓取数量
    int count = std::count(keep.begin(), keep.end(), true);
    MatrixXd result(count, graspGroupArray.cols());
    // 遍历所有抓取，将保留的复制到结果矩阵
    int idx = 0;
    for (int i = 0; i < numGrasps; ++i) {
        if (keep[i]) {
            for (size_t j = 0; j < graspGroupArray.cols(); ++j) {
                result[idx][j] = graspGroupArray[i][j];
            }
            idx++;
        }
    }
    LOG(INFO) << " result " << result.rows() << "  " << result.cols();
    return result;
}

// 对多个抓取姿态按分数排序后，剔除位置和旋转角度都过于接近的冗余抓取（只保留分数最高的那个）,
// 最终输出去重后的抓取姿态集合，避免同一位置存在大量重复的有效抓取。
GraspGroup GraspGroup::Nms(double translationThresh, double rotationThresh) const
{
    MatrixXd resultArray = NmsGrasp(graspGroupArray_, translationThresh, rotationThresh);
    return GraspGroup(resultArray);
}

GraspGroup GraspGroup::operator[](const std::vector<int>& indices) const
{
    ValidateIndices(indices);

    MatrixXd resultArray(indices.size(), GRASP_ARRAY_LEN);
    for (size_t i = 0; i < indices.size(); ++i) {
        for (int j = 0; j < GRASP_ARRAY_LEN; ++j) {
            resultArray[i][j] = graspGroupArray_[indices[i]][j];
        }
    }
    return GraspGroup(resultArray);
}

void GraspGroup::Save(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Cannot open file: " << filename;
        return;
    }

    for (int i = 0; i < Size(); ++i) {
        for (int j = 0; j < GRASP_ARRAY_LEN; ++j) {
            file << graspGroupArray_[i][j];
            if (j < GRASP_ARRAY_LEN - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}

void GraspGroup::LoadModelOutput(TensorBuf &outBufs, TensorDesc &outDescs)
{
    LOG(INFO) << "LoadModelOutput: outDescs.dimCount " << outDescs.dimCount;
    std::vector<std::vector<float>> data;
    const size_t dataSize = outDescs.typeSize / BYTE_BIT_NUM;
    int64_t width = 1;
    int64_t loopTimes = 1;
    if (outDescs.dimCount != 0) {
        width = outDescs.dims[outDescs.dimCount - 1]; // 最后一维是width
        loopTimes = outDescs.defaultSize / dataSize / width;
    }
    LOG(INFO) << "LoadModelOutput: outDescs.defaultSize " << outDescs.defaultSize;
    float* outData = new float[outDescs.defaultSize];
    DevMemcpy(outData, outDescs.defaultSize, outBufs.GetRawPtr(), outDescs.defaultSize);

    for (int64_t loop = 0; loop < loopTimes; loop++) {
        std::vector<float> row;
        row.reserve(width);
        
        for (int64_t w = 0; w < width; w++) {
            row.push_back(outData[loop * width + w]);
        }
        int objectId = row[width - 1];
        if (objectId == -1) {
            data.push_back(row);
        }
    }
    delete[] outData;

    if (data.empty()) {
        graspGroupArray_ = MatrixXd(0, GRASP_ARRAY_LEN);
        return;
    }

    size_t rows = data.size();
    size_t cols = width;
    LOG(INFO) << "rows " << rows;
    // 验证列数
    if (width != GRASP_ARRAY_LEN) {
        LOG(ERROR) << "Invalid output width " << width;
        return;
    }
    // 验证所有行都有相同的列数
    for (size_t i = 1; i < rows; ++i) {
        if (data[i].size() != cols) {
            LOG(ERROR) << "Inconsistent number of columns in file";
            return;
        }
    }

    graspGroupArray_.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            graspGroupArray_[i][j] = data[i][j];
        }
    }
}

// 索引合法性校验
void GraspGroup::ValidateIndices(const std::vector<int>& indices) const
{
    for (int idx : indices) {
        if (idx < 0 || idx >= Size()) {
            LOG(ERROR) << "Index out of range";
            return;
        }
    }
}

// Vector3i向量哈希结构体
struct hashEigen {
    std::size_t operator()(const Vector3i &v) const
    {
        size_t hashSeed = 0;
        // 对 x/y/z 三个分量分别哈希，结合0x9e3779b9（黄金比例常数，减少哈希冲突）
        // 通过移位（<<6/>>2）和异或（^）混合哈希值，提升哈希分布的均匀性
        hashSeed ^= std::hash<int>()(v.x_) + 0x9e3779b9 +
            (hashSeed << 6) + (hashSeed >> 2);
        hashSeed ^= std::hash<int>()(v.y_) + 0x9e3779b9 +
            (hashSeed << 6) + (hashSeed >> 2);
        hashSeed ^= std::hash<int>()(v.z_) + 0x9e3779b9 +
            (hashSeed << 6) + (hashSeed >> 2);
        return hashSeed;
    }
};

// 累加同一个体素内的所有点云坐标，最终计算体素的平均点
struct AccumulatedPoint {
    void AddPoint(const PointsCloud &cloud, int index)
    {
        point_ = point_ + cloud.points_[index];
        numOfPoints_++;
    }

    Vector3d GetAveragePoint() const
    {
        return point_ / double(numOfPoints_);
    }

    int numOfPoints_ = 0;
    Vector3d point_ = Vector3d(0, 0, 0);
};

// 计算点云在 x/y/z 三个轴上的最小边界
Vector3d PointsCloud::ComputeMinBound(const std::vector<Vector3d>& points) const
{
    if (points.empty()) {
        return Vector3d(0.0, 0.0, 0.0);
    }
    Vector3d result = points[0];
    for (const auto& p : points) {
        result.x_ = std::min(result.x_, p.x_);
        result.y_ = std::min(result.y_, p.y_);
        result.z_ = std::min(result.z_, p.z_);
    }
    return result;
}

// 计算点云在 x/y/z 三个轴上的最大边界
Vector3d PointsCloud::ComputeMaxBound(const std::vector<Vector3d>& points) const
{
    if (points.empty()) {
        return Vector3d(0.0, 0.0, 0.0);
    }
    Vector3d result = points[0];
    for (const auto& p : points) {
        result.x_ = std::max(result.x_, p.x_);
        result.y_ = std::max(result.y_, p.y_);
        result.z_ = std::max(result.z_, p.z_);
    }
    return result;
}

// 体积像素下采样:降低点云密度，减少后续碰撞检测的计算量
std::shared_ptr<PointsCloud> PointsCloud::VoxelDownSample(double voxelSize) const
{
    auto output = std::make_shared<PointsCloud>();
    if (voxelSize <= 0.0) { // 非法体素大小校验
        LOG(INFO) << "voxelSize <= 0.";
    }
    // 1. 计算体素化的边界（扩大0.5个体素大小，避免点落在边界外）
    Vector3d voxelSize3(voxelSize, voxelSize, voxelSize);
    Vector3d voxelMinBound = ComputeMinBound(points_) - voxelSize3 * 0.5;
    Vector3d voxelMaxBound = ComputeMaxBound(points_) + voxelSize3 * 0.5;
    // 2. 校验体素大小是否过小（避免索引溢出）
    if (voxelSize * std::numeric_limits<int>::max() <
        std::max({voxelMaxBound.x_ - voxelMinBound.x_,
                  voxelMaxBound.y_ - voxelMinBound.y_,
                  voxelMaxBound.z_ - voxelMinBound.z_})) {
        LOG(INFO) << "voxelSize is too small.";
    }
    // 3. 将每个点映射到对应的体素索引
    std::unordered_map<Vector3i, AccumulatedPoint, hashEigen> voxelindexToAccpoint;
    Vector3d refCoord;
    for (int i = 0; i < (int)points_.size(); i++) {
        // 计算点在体素网格中的相对坐标
        refCoord = (points_[i] - voxelMinBound) / voxelSize;
        // 向下取整得到体素索引
        Vector3i voxelIndex(
            int(std::floor(refCoord.x_)),
            int(std::floor(refCoord.y_)),
            int(std::floor(refCoord.z_))
        );
        // 将点添加到对应体素的累加结构中
        voxelindexToAccpoint[voxelIndex].AddPoint(*this, i);
    }
    // 4. 遍历所有体素，取平均点作为降采样结果
    for (const auto& accpoint : voxelindexToAccpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
    }

    LOG(INFO) << "PointsCloud down sampled from " << (int)points_.size() <<
        " points to " << (int)output->points_.size() << " points";
    return output;
}

// 将输入的场景点云（Eigen 矩阵）转换为自定义点云格式，降采样后再转回矩阵存储，为碰撞检测做准备
ModelFreeCollisionDetector::ModelFreeCollisionDetector(const MatrixXd& scenePoints, double voxelSize) 
    : voxelSize_(voxelSize)
{
    auto cloud = std::make_shared<PointsCloud>();
    cloud->points_.reserve(scenePoints.rows());

    for (size_t i = 0; i < scenePoints.rows(); ++i) {
        cloud->points_.emplace_back(
            scenePoints[i][0],
            scenePoints[i][1],
            scenePoints[i][2]
        );
    }
    // 对点云进行体素降采样
    auto downsampledCloud = cloud->VoxelDownSample(voxelSize);
    scenePoints_.resize(downsampledCloud->points_.size(), POINT_DIMENSION);

    for (size_t i = 0; i < downsampledCloud->points_.size(); ++i) {
        scenePoints_[i][0] = downsampledCloud->points_[i].x_;
        scenePoints_[i][1] = downsampledCloud->points_[i].y_;
        scenePoints_[i][2] = downsampledCloud->points_[i].z_;
    }
}

ModelFreeCollisionDetector::DetectionResult ModelFreeCollisionDetector::Detect(
    const GraspGroup& graspGroup, double approachDist, double collisionThresh,
    bool returnEmptyGrasp, double emptyThresh, bool returnIous)
{
    DetectionResult result;
    int numGrasps = graspGroup.Size();
    int numPoints = scenePoints_.rows();
    // 预处理：统一趋近距离和抓取参数（高度/宽度的一半）
    approachDist = std::max(approachDist, fingerWidth_);
    const MatrixXd& T = graspGroup.Translations(); // 抓取中心平移向量
    std::vector<Matrix3d> R = graspGroup.RotationMatrices();// 抓取旋转矩阵
    VectorXd heights = graspGroup.Heights(); // 抓取高度
    VectorXd depths = graspGroup.Depths(); // 抓取深度
    VectorXd widths = graspGroup.Widths(); // 抓取宽度

    double voxelSizeCubed = std::pow(voxelSize_, POINT_DIMENSION);
    std::vector<double> halfHeights(numGrasps);
    std::vector<double> halfWidths(numGrasps);
    for (int g = 0; g < numGrasps; ++g) {
        halfHeights[g] = heights[g] / 2.0;
        halfWidths[g] = widths[g] / 2.0;
    }

    // 初始化结果
    result.collisionMask.resize(numGrasps, false);
    if (returnEmptyGrasp) result.emptyMask.resize(numGrasps, false);
    if (returnIous) {
        result.globalIou.resize(numGrasps, 0.0);
        result.leftIou.resize(numGrasps, 0.0);
        result.rightIou.resize(numGrasps, 0.0);
        result.bottomIou.resize(numGrasps, 0.0);
        result.shiftingIou.resize(numGrasps, 0.0);
    }

    // 将点云数据拷贝到数组，减少矩阵访问开销
    std::vector<double> sceneX(numPoints), sceneY(numPoints), sceneZ(numPoints);
    for (int p = 0; p < numPoints; ++p) {
        sceneX[p] = scenePoints_[p][0];
        sceneY[p] = scenePoints_[p][1];
        sceneZ[p] = scenePoints_[p][2];
    }

    // 并行处理每个抓取姿态（OpenMP加速，数量>10时并行）
    #pragma omp parallel for if(numGrasps > 10)
    for (int g = 0; g < numGrasps; ++g) {
        // 抓取姿态参数
        const Matrix3d& rot = R[g];
        double tx = T[g][0], ty = T[g][1], tz = T[g][2];
        double height = heights[g];
        double depth = depths[g];
        double width = widths[g];
        double hh = halfHeights[g];
        double hw = halfWidths[g];

        // 预计算旋转矩阵元素
        double r00 = rot(0,0), r01 = rot(0,1), r02 = rot(0,2);
        double r10 = rot(1,0), r11 = rot(1,1), r12 = rot(1,2);
        double r20 = rot(2,0), r21 = rot(2,1), r22 = rot(2,2);

        // 定义抓取区域的边界（局部坐标系）
        double xMin = depth - fingerLength_;
        double xMax = depth;
        double yLeftMin = -(hw + fingerWidth_);
        double yLeftMax = -hw;
        double yRightMin = hw;
        double yRightMax = hw + fingerWidth_;
        double xBottomMin = depth - fingerLength_ - fingerWidth_;
        double xBottomMax = depth - fingerLength_;
        double xShiftingMin = depth - fingerLength_ - fingerWidth_ - approachDist;
        double xShiftingMax = depth - fingerLength_ - fingerWidth_;

        // 统计不同区域的点数量
        int leftCount = 0, rightCount = 0, bottomCount = 0, shiftingCount = 0;
        int innerCount = 0;

        // 遍历所有场景点，判断是否在抓取区域内
        for (int p = 0; p < numPoints; ++p) {
            // 将场景点从世界坐标系转换到抓取局部坐标系（平移+旋转）
            double px = sceneX[p] - tx;
            double py = sceneY[p] - ty;
            double pz = sceneZ[p] - tz;

            // 旋转：应用旋转矩阵
            double x = r00 * px + r10 * py + r20 * pz;
            double y = r01 * px + r11 * py + r21 * pz;
            double z = r02 * px + r12 * py + r22 * pz;

            // 检查高度条件（z轴），不满足则跳过
            bool mask1 = (z > -hh) && (z < hh);
            if (!mask1) continue;

            // 检查不同区域的边界条件
            bool mask2 = (x > xMin) && (x < xMax); // 深度范围
            bool mask3 = (y > yLeftMin);
            bool mask4 = (y < yLeftMax); // 左手指区域
            bool mask5 = (y < yRightMax);
            bool mask6 = (y > yRightMin); // 右手指区域
            bool mask7 = (x <= xBottomMax) && (x > xBottomMin); // 底部区域
            bool mask8 = (x <= xShiftingMax) && (x > xShiftingMin); // 趋近区域

            // 统计各区域的点数量
            if (mask2 && mask3 && mask4) leftCount++;
            if (mask2 && mask5 && mask6) rightCount++;
            if (mask3 && mask5 && mask7) bottomCount++;
            if (mask3 && mask5 && mask8) shiftingCount++;
            if (mask2 && !mask4 && !mask6) innerCount++;
        }

        // 计算各区域的体积（归一化到体素数量）
        double leftRightVolume = height * fingerLength_ * fingerWidth_ / voxelSizeCubed;
        double bottomVolume = height * (width + 2 * fingerWidth_) * fingerWidth_ / voxelSizeCubed;
        double shiftingVolume = height * (width + 2 * fingerWidth_) * approachDist / voxelSizeCubed;
        double volume = leftRightVolume * 2 + bottomVolume + shiftingVolume;
        // 计算全局IoU（交并比），判断是否碰撞
        int globalCount = leftCount + rightCount + bottomCount + shiftingCount;
        double globalIou = globalCount / (volume + 1e-6);
        result.collisionMask[g] = (globalIou > collisionThresh);
        // 返回各区域IoU和空抓取判断
        if (returnIous) {
            result.globalIou[g] = globalIou;
            result.leftIou[g] = leftCount / (leftRightVolume + 1e-6);
            result.rightIou[g] = rightCount / (leftRightVolume + 1e-6);
            result.bottomIou[g] = bottomCount / (bottomVolume + 1e-6);
            result.shiftingIou[g] = shiftingCount / (shiftingVolume + 1e-6);
        }

        if (returnEmptyGrasp) {
            double innerVolume = height * fingerLength_ * width / voxelSizeCubed;
            result.emptyMask[g] = (innerCount / innerVolume < emptyThresh); // 内部点少则为空抓取
        }
    }

    return result;
}

static GraspGroup CollisionDetection(const GraspGroup& graspGroup, 
    const std::shared_ptr<PointsCloud>& cloud, double voxelSize, double collisionThresh)
{
    // 转换点云为Eigen矩阵
    MatrixXd scenePoints(cloud->points_.size(), POINT_DIMENSION);
    for (size_t i = 0; i < cloud->points_.size(); ++i) {
        scenePoints[i][0] = cloud->points_[i].x_;
        scenePoints[i][1] = cloud->points_[i].y_;
        scenePoints[i][2] = cloud->points_[i].z_;
    }

    ModelFreeCollisionDetector detector(scenePoints, voxelSize);
    auto result = detector.Detect(graspGroup, 0.05, collisionThresh);

    std::vector<int> keepIndices;
    for (size_t i = 0; i < result.collisionMask.size(); ++i) {
        if (!result.collisionMask[i]) {
            keepIndices.push_back(i);
        }
    }

    return graspGroup[keepIndices];
}

static void VisGrasps(GraspGroup& graspGroup, const std::shared_ptr<PointsCloud>& cloud)
{
    auto sortedGrasps = graspGroup;
    sortedGrasps.Nms();
    sortedGrasps.SortByScore();

    // 取score排名前50的抓取姿态
    int numToShow = std::min(50, sortedGrasps.Size());
    std::vector<int> indices(numToShow);
    for (int i = 0; i < numToShow; ++i) {
        indices[i] = i;
    }

    graspGroup = sortedGrasps[indices];
    for (int i = 0; i < 10; ++i) {
        LOG(INFO) << "score:" << graspGroup.Scores()[i] << " " <<
            "width:" << graspGroup.Widths()[i] << " " <<
            "height:" << graspGroup.Heights()[i] << " " <<
            "depth:" << graspGroup.Depths()[i] << " " <<
            "translation: " << graspGroup.Translations()[i][0] << " " <<
            graspGroup.Translations()[i][1] << " " << graspGroup.Translations()[i][2];
        LOG(INFO) << "rotation:";
        auto rot = graspGroup.RotationMatrices()[i];
        for (int r = 0; r < POINT_DIMENSION; ++r) {
            LOG(INFO) << rot(r, 0) << " " << rot(r, 1) << " " << rot(r, 2);
        }
        LOG(INFO) << "object id:" << graspGroup.ObjectIds()[i];
    }
}

static void DataPostprocess(GraspGroup& graspGroupArray, const std::shared_ptr<PointsCloud>& cloud,
    float voxelSize, float collisionThresh)
{
    if (collisionThresh > 0) {
        graspGroupArray = CollisionDetection(graspGroupArray, cloud, voxelSize, collisionThresh);
    }

    VisGrasps(graspGroupArray, cloud);
}

static std::vector<Vector3d> LoadVectorFromBinary(const std::string& filename)
{
    std::ifstream saveBin(filename, std::ios::binary);
    if (!saveBin) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    size_t numPoints;
    //  读取点的数量
    saveBin.read(reinterpret_cast<char*>(&numPoints), sizeof(size_t));

    //  创建vector并读取数据
    std::vector<Vector3d> points(numPoints);
    saveBin.read(reinterpret_cast<char*>(points.data()), numPoints * sizeof(Vector3d));

    saveBin.close();
    return points;
}

static std::shared_ptr<PointsCloud> LoadCloud(const std::string& saveBinDir)
{
    auto cloud = std::make_shared<PointsCloud>();
    std::vector<Vector3d> points;
    std::vector<Vector3d> colors;
    points = LoadVectorFromBinary(saveBinDir + "cloud_points.bin");
    colors = LoadVectorFromBinary(saveBinDir + "cloud_color.bin");
    if (points.size() != colors.size()) {
        throw std::runtime_error("Cloud size not match, points size: " +
            std::to_string(points.size()) + "colors size" + std::to_string(colors.size()));
    }
    cloud->points_ = points;
    cloud->colors_ = colors;
    return cloud;
}

std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ?
           path.substr(start) : path.substr(start, end - start);
}

bool GraspNetPostprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& outBufs,
    std::vector<TensorDesc>& outDescs)
{
    auto cfg = ReadCfgFile("../data/cfg.txt");
    float voxelSize = std::stof(cfg["voxel_size"]);
    float collisionThresh = std::stof(cfg["collision_thresh"]);
    LOG(INFO) << "voxelSize: " << voxelSize << " collisionThresh: " << collisionThresh;
    std::string saveBinDir = cfg["save_out_bin"];
    std::string saveTxtDir = cfg["save_out_txt"];

    // 创建保存目录
    if (!saveBinDir.empty() && !CreateDirectoryRecursive(saveBinDir)) {
        LOG(ERROR) << "Create preprocess bin dir failed";
        return false;
    }
    if (!saveTxtDir.empty() && !CreateDirectoryRecursive(saveTxtDir)) {
        LOG(ERROR) << "Create preprocess txt dir failed";
        return false;
    }

    auto cloud = LoadCloud(saveBinDir);
    GraspGroup graspGroup(outBufs, outDescs);
    DataPostprocess(graspGroup, cloud, voxelSize, collisionThresh);
    const std::string& imgPath = fileList[0];
    std::string fileName = GetFileName(imgPath);
    graspGroup.Save(saveTxtDir + fileName + "_sorted_grasp_group.txt");
    return true;
}

}
}