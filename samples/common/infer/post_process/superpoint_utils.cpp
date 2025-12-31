#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "superpoint_utils.h"

std::string ExtractFilename(const std::string &filePath) {
    // 找到最后一个路径分隔符（处理'/'和'\\'）
    size_t sepPos = filePath.find_last_of("/\\");
    std::string filenameWithExt;
    
    if (sepPos != std::string::npos) {
        filenameWithExt = filePath.substr(sepPos + 1);
    } else {
        filenameWithExt = filePath; // 没有路径分隔符，整个字符串就是文件名
    }
    
    // 分离扩展名
    size_t dotPos = filenameWithExt.find_last_of('.');
    std::string filename = (dotPos != std::string::npos) ? 
                          filenameWithExt.substr(0, dotPos) : 
                          filenameWithExt;
    return filename;
}

std::vector<float> SuperpointPreprocess(const std::string& imageFile)
{
    cv::Mat bgr = cv::imread(imageFile);
    if (bgr.empty())
        throw std::runtime_error("cannot read image: " + imageFile);

    const int imgHeight = bgr.rows;
    const int imgWidth = bgr.cols;
    if (imgHeight != 240 || imgWidth != 320) {
            throw std::runtime_error("当前图片尺寸（宽"+ std::to_string(imgWidth) + ",高" + std::to_string(imgHeight)
                + "）不支持，请输出宽320，高240 的jpg图片");
    }

    /* 1. 灰度 + 归一化 [0,1] */
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    /* 2. 保存预处理后的图像为二进制文件 */
    std::string binSavePath = ExtractFilename(imageFile) + "_preprocessed.bin";
    if (!binSavePath.empty()) {
        // 打开文件流，以二进制写入模式打开
        std::ofstream ofs(binSavePath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("无法打开二进制文件进行写入: " + binSavePath);
        }
        
        // 写入实际的浮点数据
        size_t dataSize = imgHeight * imgWidth * sizeof(float);
        ofs.write(reinterpret_cast<const char*>(gray.ptr<float>()), dataSize);
        // 检查写入是否成功
        if (!ofs) {
            throw std::runtime_error("写入二进制文件失败: " + binSavePath);
        }
        std::cout<< "写入二进制文件成功: " + binSavePath << std::endl;
    }

    /* 3. 返回 NCHW 向量 */
    std::vector<float> buffer(imgHeight * imgWidth);
    std::memcpy(buffer.data(), gray.ptr<float>(), buffer.size() * sizeof(float));

    return buffer;
}

// Softmax函数
static void Softmax2dInplace(std::vector<float>& data, int batch, int channels, int height, int width)
{
    for (int b = 0; b < batch; ++b) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::vector<float> channelValues(channels);
                float maxVal = -1e9;
                // 求出65个值中的最大值
                for (int c = 0; c < channels; ++c) {
                    int idx = b * channels * height * width + c * height * width + y * width + x;
                    channelValues[c] = data[idx];
                    maxVal = std::max(maxVal, channelValues[c]);
                }
                // 求出exp， 
                float sumExp = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    channelValues[c] = std::exp(channelValues[c] - maxVal);
                    sumExp += channelValues[c];
                }
                // 使得所有exp 累加和为 1， 即总概率为1
                for (int c = 0; c < channels; ++c) {
                    int idx = b * channels * height * width + c * height * width + y * width + x;
                    data[idx] = channelValues[c] / sumExp;
                }
            }
        }
    }
}

// Depth-to-Space操作
static std::vector<float> DepthToSpace(const std::vector<float>& input,
    int batch, int channels, int height, int width, int blockSize)
{
    int blockSq = blockSize * blockSize;
    int outC = channels / blockSq;
    int outH = height * blockSize;
    int outW = width * blockSize;
    
    std::vector<float> output(batch * outC * outH * outW, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int inIdx = b * channels * height * width + c * height * width + y * width + x;
                    float val = input[inIdx];
                    
                    int blockNum = c % blockSq;
                    int blockRow = blockNum / blockSize;
                    int blockCol = blockNum % blockSize;

                    int outChannel = c / blockSq;
                    int outY = y * blockSize + blockRow;
                    int outX = x * blockSize + blockCol;

                    int outIdx = b * outC * outH * outW + outChannel * outH * outW + outY * outW + outX;
                    output[outIdx] = val;
                }
            }
        }
    }
    return output;
}

// NMS (非极大值抑制)
static std::vector<KeyPoint> NmsFast(const std::vector<KeyPoint>& corners, int H, int W, int nmsDist)
{
    std::vector<KeyPoint> keepPoints;
    if (corners.empty()) return keepPoints;

    std::vector<KeyPoint> sortedCorners = corners;
    std::sort(sortedCorners.begin(), sortedCorners.end(), [](const KeyPoint& a, const KeyPoint& b) {
        return a.score > b.score;
    });

    std::vector<std::vector<int>> grid(H, std::vector<int>(W, 0));
    std::vector<std::vector<int>> inds(H, std::vector<int>(W, -1));

    for (size_t i = 0; i < sortedCorners.size(); ++i) {
        int x = static_cast<int>(std::round(sortedCorners[i].x));
        int y = static_cast<int>(std::round(sortedCorners[i].y));
        if (x >= 0 && x < W && y >= 0 && y < H) {
            grid[y][x] = 1;
            inds[y][x] = static_cast<int>(i);
        }
    }

    int pad = nmsDist;
    std::vector<std::vector<int>> paddedGrid(H + 2 * pad, std::vector<int>(W + 2 * pad, 0));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            paddedGrid[y + pad][x + pad] = grid[y][x];
        }
    }

    for (size_t i = 0; i < sortedCorners.size(); ++i) {
        int x = static_cast<int>(std::round(sortedCorners[i].x));
        int y = static_cast<int>(std::round(sortedCorners[i].y));
        if (x < 0 || x >= W || y < 0 || y >= H) continue;
        
        int px = x + pad;
        int py = y + pad;
        if (paddedGrid[py][px] != 1) {
            continue;
        }
        for (int dy = -pad; dy <= pad; ++dy) {
            for (int dx = -pad; dx <= pad; ++dx) {
                int ny = py + dy;
                int nx = px + dx;
                if (ny >= 0 && ny < (int)paddedGrid.size() && nx >=0 && nx < (int)paddedGrid[0].size()) {
                    paddedGrid[ny][nx] = 0;
                }
            }
        }
        paddedGrid[py][px] = -1;
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (paddedGrid[y + pad][x + pad] != -1) {
                continue;
            }
            int idx = inds[y][x];
            if (idx != -1) {
                keepPoints.push_back(sortedCorners[idx]);
            }
        }
    }
    return keepPoints;
}

// 提取初始关键点
static std::vector<KeyPoint> ExtractInitialKeypoints(const std::vector<float>& heatmap, int H, int W,
    float confThresh, int borderRemove)
{
    std::vector<KeyPoint> pts;
    int startX = std::max(0, borderRemove);
    int endX = std::min(W, W - borderRemove);
    int startY = std::max(0, borderRemove);
    int endY = std::min(H, H - borderRemove);

    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            float score = heatmap[y * W + x];
            if (score >= confThresh) {
                pts.push_back({static_cast<float>(x), static_cast<float>(y), score, {}});
            }
        }
    }
    return pts;
}

// 双线性插值
static float BilinearInterpolate(const std::vector<float>& img, int H, int W, float y, float x) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || x1 >= W || y0 < 0 || y1 >= H) {
        return 0.0f;
    }

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * W + x0];
    float v10 = img[y0 * W + x1];
    float v01 = img[y1 * W + x0];
    float v11 = img[y1 * W + x1];

    return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + 
           (1 - dx) * dy * v01 + dx * dy * v11;
}

// 提取点周围的补丁
static std::vector<std::vector<float>> ExtractPatches(const std::vector<float>& heatmap, int H, int W,
    const std::vector<KeyPoint>& pts, int patchSize)
{
    int padSize = patchSize / 2;
    std::vector<std::vector<float>> patches;
    patches.reserve(pts.size());

    for (const auto& p : pts) {
        std::vector<float> patch(patchSize * patchSize, 0.0f);
        for (int dy = -padSize; dy <= padSize; ++dy) {
            for (int dx = -padSize; dx <= padSize; ++dx) {
                float x = p.x + dx;
                float y = p.y + dy;
                patch[(dy + padSize) * patchSize + (dx + padSize)] = BilinearInterpolate(heatmap, H, W, y, x);
            }
        }
        patches.push_back(patch);
    }
    return patches;
}

// 补丁归一化
static void NormalizePatches(std::vector<std::vector<float>>& patches)
{
    for (auto& patch : patches) {
        float sum = std::accumulate(patch.begin(), patch.end(), 0.0f) + 1e-6f;
        for (float& val : patch) {
            val /= sum;
        }
    }
}

// 亚像素级精确化
static std::vector<KeyPoint> SubpixelRefinement(const std::vector<float>& heatmap, int H, int W,
    const std::vector<KeyPoint>& pts, int patchSize)
{
    std::vector<KeyPoint> refinedPts = pts;
    auto patches = ExtractPatches(heatmap, H, W, pts, patchSize);
    NormalizePatches(patches);

    int halfPatch = patchSize / 2;
    for (size_t i = 0; i < patches.size(); ++i) {
        const auto& patch = patches[i];
        float sumExp = 0.0f;
        std::vector<float> expVals(patch.size());
        
        for (size_t j = 0; j < patch.size(); ++j) {
            expVals[j] = std::exp(patch[j] + 1e-6f);
            sumExp += expVals[j];
        }

        float dx = 0.0f, dy = 0.0f;
        for (int y = 0; y < patchSize; ++y) {
            for (int x = 0; x < patchSize; ++x) {
                float weight = expVals[y * patchSize + x] / sumExp;
                dx += static_cast<float>(x) * weight;
                dy += static_cast<float>(y) * weight;
            }
        }
        
        refinedPts[i].x += (dx - static_cast<float>(halfPatch));
        refinedPts[i].y += (dy - static_cast<float>(halfPatch));
    }
    return refinedPts;
}

// 提取描述子
static std::vector<KeyPoint> ExtractDescriptors(const std::vector<float>& coarseDesc, int descC,
    int descH, int descW, const std::vector<KeyPoint>& pts, int cellSize)
{
    std::vector<KeyPoint> ptsWithDesc = pts;

    float imgHeight = static_cast<float>(descH * cellSize);
    float imgWidth = static_cast<float>(descW * cellSize);

    for (size_t i = 0; i < pts.size(); ++i) {
        float xNorm = (pts[i].x / (imgWidth - 1.0f)) * 2.0f - 1.0f;
        float yNorm = (pts[i].y / (imgHeight - 1.0f)) * 2.0f - 1.0f;

        float xSamp = 0.5f * (xNorm + 1.0f) * static_cast<float>(descW - 1);
        float ySamp = 0.5f * (yNorm + 1.0f) * static_cast<float>(descH - 1);

        int x0 = static_cast<int>(std::floor(xSamp));
        int y0 = static_cast<int>(std::floor(ySamp));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        std::vector<float> desc(descC, 0.0f);
        if (!(x0 < 0 || x1 >= descW || y0 < 0 || y1 >= descH)) {
            float dx = xSamp - static_cast<float>(x0);
            float dy = ySamp - static_cast<float>(y0);

            for (int c = 0; c < descC; ++c) {
                int idx00 = c * descH * descW + y0 * descW + x0;
                int idx10 = c * descH * descW + y0 * descW + x1;
                int idx01 = c * descH * descW + y1 * descW + x0;
                int idx11 = c * descH * descW + y1 * descW + x1;
                
                float v00 = coarseDesc[idx00];
                float v10 = coarseDesc[idx10];
                float v01 = coarseDesc[idx01];
                float v11 = coarseDesc[idx11];
                
                desc[c] = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + 
                          (1 - dx) * dy * v01 + dx * dy * v11;
            }
            
            float norm = std::sqrt(std::inner_product(desc.begin(), desc.end(), desc.begin(), 0.0f) + 1e-6f);
            for (float& v : desc) {
                v /= norm;
            }
        }
        
        ptsWithDesc[i].descriptor = desc;
    }
    return ptsWithDesc;
}

// 处理关键点和描述子的函数
std::vector<KeyPoint> ProcessKeypoints(const std::vector<float> &inputSemiData, const std::vector<float> &inputDescData)
{
    int nmsDist = 4;
    float confThresh = 0.015f;
    int borderRemove = 4;
    int cellSize = 8;
    int patchSize = 5;

    // 生成热力图
    int B = 1, semiC = 65, semiH = 30, semiW = 40;

     // 解决const引用冲突：创建semiData的非const副本
    std::vector<float> semiData = inputSemiData;
    std::vector<float> descData = inputDescData;
    Softmax2dInplace(semiData, B, semiC, semiH, semiW);
    
    // 移除dust channel
    std::vector<float> nodustData(B * (semiC - 1) * semiH * semiW);
    for (int i = 0; i < B; ++i) {
        for (int c = 0; c < semiC - 1; ++c) {
            for (int y = 0; y < semiH; ++y) {
                for (int x = 0; x < semiW; ++x) {
                    int srcIdx = i * semiC * semiH * semiW + c * semiH * semiW + y * semiW + x;
                    int dstIdx = i * (semiC - 1) * semiH * semiW + c * semiH * semiW + y * semiW + x;
                    nodustData[dstIdx] = semiData[srcIdx];
                }
            }
        }
    }
    
    // 执行depth to space操作
    auto heatmap = DepthToSpace(nodustData, B, semiC - 1, semiH, semiW, cellSize);
    int heatH = semiH * cellSize;
    int heatW = semiW * cellSize;
    
    // 提取单通道热力图数据
    std::vector<float> heatmapSingle(heatH * heatW);
    for (int i = 0; i < heatH * heatW; ++i) {
        heatmapSingle[i] = heatmap[i];
    }

    // 检测关键点
    auto initialPts = ExtractInitialKeypoints(heatmapSingle, heatH, heatW, confThresh, borderRemove);
    std::cout << "Initial keypoints detected: " << initialPts.size() << std::endl;
    
    if (initialPts.empty()) {
        throw std::runtime_error("No keypoints detected");
    }
    
    // 应用NMS
    auto nmsPts = NmsFast(initialPts, heatH, heatW, nmsDist);
    std::cout << "Keypoints after NMS: " << nmsPts.size() << std::endl;
    
    if (nmsPts.empty()) {
        throw std::runtime_error("No keypoints remaining after NMS");
    }

    // 亚像素级精确化
    auto refinedPts = SubpixelRefinement(heatmapSingle, heatH, heatW, nmsPts, patchSize);
    std::cout << "Keypoints after subpixel refinement: " << refinedPts.size() << std::endl;

    // 提取描述符
    int descC = 256, descH = 30, descW = 40;
    auto ptsWithDescriptors = ExtractDescriptors(descData, descC, descH, descW, refinedPts, cellSize);
    std::cout << "Extracted " << ptsWithDescriptors.size() << " descriptors" << std::endl;

    return ptsWithDescriptors;
}

// 可视化关键点的函数
void VisualizeKeypoints(const std::vector<KeyPoint>& keypoints, const std::string& imagePath, 
    const std::string& outputPath) 
{
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
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
    
    cv::Size targetSize = cv::Size(targetW,targetH);
    // 缩放
    cv::resize(processedImage, processedImage, targetSize, 0, 0, cv::INTER_AREA);
    
    // 绘制关键点
    for (const auto& p : keypoints) {
        cv::circle(processedImage, cv::Point(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y))), 
                  2, cv::Scalar(0, 255, 0), -1);
    }
    
    // 保存结果
    cv::imwrite(outputPath, processedImage);
    std::cout << "Result saved to: " << outputPath << std::endl;
}