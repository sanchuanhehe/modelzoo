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


#include "hrnet_postprocess.h"
#include "log.h"
#include <fstream>
#include <sys/stat.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <map>

namespace Infer {
using namespace std;
using namespace cv;

class Munkres {
public:
    std::vector<std::pair<int, int>> Compute(const std::vector<std::vector<float>>& costMatrix) {
        int n = costMatrix.size();
        int m = n > 0 ? costMatrix[0].size() : 0;

        // 初始化辅助数组
        std::vector<int> u(n + 1, 0);
        std::vector<int> v(m + 1, 0);
        std::vector<int> p(m + 1, 0);  // 匹配结果（列→行）
        std::vector<int> way(m + 1, 0); // 路径记录

        // 复制代价矩阵
        std::vector<std::vector<float>> matrix = costMatrix;

        for (int i = 1; i <= n; ++i) {
            p[0] = i;
            int minv = std::numeric_limits<int>::max();
            std::vector<int> minvals(m + 1, std::numeric_limits<int>::max());
            std::vector<int> used(m + 1, 0);

            int j0 = 0;
            do {
                used[j0] = 1;
                int i0 = p[j0];
                int delta = std::numeric_limits<int>::max();
                int j1 = 0;

                // 计算当前行的最小代价
                for (int j = 1; j <= m; ++j) {
                    if (used[j] == 0) {
                        float curFloat = matrix[i0 - 1][j - 1] - u[i0] - v[j];
                        int cur = static_cast<int>(curFloat);
                        if (cur < minvals[j]) {
                            minvals[j] = cur;
                            way[j] = j0;
                        }
                        if (minvals[j] < delta) {
                            delta = minvals[j];
                            j1 = j;
                        }
                    }
                }

                // 更新势能
                for (int j = 0; j <= m; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minvals[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            // 调整匹配
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }

        // 整理匹配结果（行→列）
        std::vector<std::pair<int, int>> result;
        for (int j = 1; j <= m; ++j) {
            if (p[j] != 0) {
                result.emplace_back(p[j] - 1, j - 1);
            }
        }

        return result;
    }
};

class Params {
public:
    int NumJoints;
    int MaxNumPeople;
    float DetectionThreshold;
    float TagThreshold;
    bool UseDetectionVal;
    bool IgnoreTooMuch;
    std::vector<int> JointOrder;

    Params() {
        // 默认适配COCO数据集（17关节）
        NumJoints = 17;
        MaxNumPeople = 30;
        DetectionThreshold = 0.1f;
        TagThreshold = 1.0f;
        UseDetectionVal = true;
        IgnoreTooMuch = true;
        // 关节顺序
        JointOrder = {0,1,2,3,4,5,6,11,12,7,8,9,10,13,14,15,16};
    }

    Params(int numJoints_, int maxNumPeople_, float detThresh_, float tagThresh_, 
           bool useDetVal_, bool ignoreTooMuch_, const std::vector<int>& jointOrder_) {
        NumJoints = numJoints_;
        MaxNumPeople = maxNumPeople_;
        DetectionThreshold = detThresh_;
        TagThreshold = tagThresh_;
        UseDetectionVal = useDetVal_;
        IgnoreTooMuch = ignoreTooMuch_;
        JointOrder = jointOrder_;
    }
};

std::vector<std::pair<int, int>> PyMaxMatch(const std::vector<std::vector<float>>& scores) {
    Munkres munkres;
    return munkres.Compute(scores);
}

std::vector<cv::Mat> MatchByTag(const cv::Mat& tagK, const cv::Mat& locK, const cv::Mat& valK, const Params& params) {
    int numJoints = params.NumJoints;
    cv::Mat defaultMat = cv::Mat::zeros(numJoints, 3 + tagK.cols, CV_32F);

    std::map<float, cv::Mat> jointDict;  // key: tag值, value: 人物关节矩阵
    std::map<float, std::vector<cv::Mat>> tagDict;

    for (int i = 0; i < numJoints; ++i) {
        int idx = params.JointOrder[i];

        // 提取当前关节的Tag和关节信息
        cv::Mat tags = tagK.row(idx);
        cv::Mat locs = locK.row(idx);
        cv::Mat vals = valK.row(idx);

        // 拼接关节信息: (x,y,val,tag...)
        cv::Mat joints;
        cv::hconcat(locs, vals.reshape(1, vals.rows), joints);
        cv::hconcat(joints, tags, joints);

        // 过滤低置信度关节
        cv::Mat mask;
        cv::compare(joints.col(2), params.DetectionThreshold, mask, cv::CMP_GT);
        std::vector<int> validIndices;
        for (int r = 0; r < joints.rows; ++r) {
            if (mask.at<uchar>(r) == 255) {
                validIndices.push_back(r);
            }
        }

        if (validIndices.empty()) continue;

        // 提取有效关节和Tag
        cv::Mat validJoints, validTags;
        for (int r : validIndices) {
            validJoints.push_back(joints.row(r));
            validTags.push_back(tags.row(r));
        }
        validJoints = validJoints.reshape(joints.cols, validIndices.size());
        validTags = validTags.reshape(tags.cols, validIndices.size());

        if (i == 0 || jointDict.empty()) {
            // 第一个关节，初始化人物
            for (int r = 0; r < validJoints.rows; ++r) {
                float key = validTags.at<float>(r, 0);
                cv::Mat person = defaultMat.clone();
                person.row(idx) = validJoints.row(r);
                jointDict[key] = person;
                tagDict[key].push_back(validTags.row(r));
            }
        } else {
            // 已有人物，计算Tag相似度并匹配
            std::vector<float> groupedKeys;
            for (auto& pair : jointDict) {
                groupedKeys.push_back(pair.first);
                if (groupedKeys.size() >= static_cast<size_t>(params.MaxNumPeople)) break;
            }

            if (params.IgnoreTooMuch && groupedKeys.size() == static_cast<size_t>(params.MaxNumPeople)) continue;

            // 计算已分组人物的平均Tag
            std::vector<cv::Mat> groupedTags;
            for (float key : groupedKeys) {
                cv::Mat avgTag = cv::Mat::zeros(1, validTags.cols, CV_32F);
                for (const cv::Mat& t : tagDict[key]) {
                    avgTag += t;
                }
                avgTag /= (float)tagDict[key].size();
                groupedTags.push_back(avgTag);
            }

            // 计算代价矩阵: 新关节Tag与已分组Tag的距离（转为Vector）
            int numAdded = validJoints.rows;
            int numGrouped = groupedKeys.size();
            std::vector<std::vector<float>> diffNormed(numAdded, std::vector<float>(numGrouped, 0.0f));
            cv::Mat diffSavedMat = cv::Mat::zeros(numAdded, numGrouped, CV_32F);

            for (int r = 0; r < numAdded; ++r) {
                cv::Mat tag = validTags.row(r);
                for (int c = 0; c < numGrouped; ++c) {
                    cv::Mat diff = tag - groupedTags[c];
                    float norm = (float)cv::norm(diff, cv::NORM_L2);
                    diffNormed[r][c] = norm;
                    diffSavedMat.at<float>(r, c) = norm;
                }
            }

            // 若使用检测值加权
            if (params.UseDetectionVal) {
                for (int r = 0; r < numAdded; ++r) {
                    float val = validJoints.at<float>(r, 2);
                    for (int c = 0; c < numGrouped; ++c) {
                        diffNormed[r][c] = std::round(diffNormed[r][c]) * 100 - val;
                    }
                }
            }

            // 扩展代价矩阵（当新关节数 > 已分组人数）
            if (numAdded > numGrouped) {
                for (int r = 0; r < numAdded; ++r) {
                    diffNormed[r].resize(numAdded, 1e10f);
                }
            }

            // 匈牙利算法匹配（Vector版）
            auto pairs = PyMaxMatch(diffNormed);

            // 处理匹配结果
            for (auto& pair : pairs) {
                int row = pair.first;
                int col = pair.second;

                if (row < numAdded && col < numGrouped && diffSavedMat.at<float>(row, col) < params.TagThreshold) {
                    // 匹配到已有人物
                    float key = groupedKeys[col];
                    jointDict[key].row(idx) = validJoints.row(row);
                    tagDict[key].push_back(validTags.row(row));
                } else if (row < numAdded) {
                    // 新增人物
                    float key = validTags.at<float>(row, 0);
                    cv::Mat person = defaultMat.clone();
                    person.row(idx) = validJoints.row(row);
                    jointDict[key] = person;
                    tagDict[key].push_back(validTags.row(row));
                }
            }
        }
    }

    // 转换为输出格式
    std::vector<cv::Mat> ans;
    for (auto& pair : jointDict) {
        ans.push_back(pair.second);
    }
    return ans;
}

// 热力图解析器（仅支持单张图）
class HeatmapParser {
private:
    Params mParams;
    bool mTagPerJoint;
    int mNmsKernel;
    int mNmsPadding;

    // NMS（私有函数）
    cv::Mat Nms(const cv::Mat& det) {
        CV_Assert(det.dims == 3); // [numJoints, H, W]（单张图的热力图）
        int numJoints = det.size[0];
        int H = det.size[1];
        int W = det.size[2];

        // 修复：多维Mat的zeros创建方式
        int ndims = det.dims;
        const int* sz = det.size.p;
        cv::Mat maxm = cv::Mat::zeros(ndims, sz, CV_32F);
        
        cv::Mat detMat = det.reshape(1, numJoints); // [numJoints, H*W]

        // 手动实现MaxPool2d (kernel=3, stride=1, padding=1)
        for (int j = 0; j < numJoints; ++j) {
            cv::Mat jointHeatmap = detMat.row(j).reshape(1, H);
            cv::Mat poolHeatmap = cv::Mat::zeros(H, W, CV_32F);

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    // 计算池化窗口范围
                    int yStart = std::max(0, y - mNmsPadding);
                    int yEnd = std::min(H-1, y + mNmsPadding);
                    int xStart = std::max(0, x - mNmsPadding);
                    int xEnd = std::min(W-1, x + mNmsPadding);

                    // 窗口内最大值
                    cv::Rect roi(xStart, yStart, xEnd - xStart + 1, yEnd - yStart + 1);
                    cv::Mat window = jointHeatmap(roi);
                    double maxVal;
                    cv::minMaxLoc(window, nullptr, &maxVal);
                    poolHeatmap.at<float>(y, x) = (float)maxVal;
                }
            }

            const float eps = 1e-6;
            // 2. 生成mask：|poolHeatmap - jointHeatmap| < eps 视为相等（即最大值位置）
            cv::Mat diff;
            cv::absdiff(poolHeatmap, jointHeatmap, diff); // 计算差值的绝对值
            cv::Mat mask = (diff < eps); // 生成CV_8UC1类型的mask（0/255）

            // 3. 转换mask类型并归一化
            cv::Mat maskFloat;
            mask.convertTo(maskFloat, jointHeatmap.type());
            maskFloat /= 255.0f;

            // 4. 执行乘法，保留最大值位置
            jointHeatmap = jointHeatmap.mul(maskFloat);

            poolHeatmap.copyTo(maxm.reshape(1, numJoints).row(j));
        }

        return maxm;
    }

    // Top-K提取
    void TopK(const cv::Mat& det, const cv::Mat& tag, cv::Mat& tagK, cv::Mat& locK, cv::Mat& valK) {
        CV_Assert(det.dims == 3 && tag.dims == 4); // det:[numJoints,H,W], tag:[numJoints,H,W,tagDim]
        // 单张图：det维度 [numJoints, H, W]，tag维度 [numJoints, H, W, tagDim]
        int numJoints = det.size[0];
        int H = det.size[1];
        int W = det.size[2];
        int tagDim = tag.size[3];

        // NMS处理
        cv::Mat detNms = Nms(det);

        // Reshape为[numJoints, H*W]
        cv::Mat detFlat = detNms.reshape(1, numJoints); // [numJoints, H*W]

        // Top-K提取
        int topK = mParams.MaxNumPeople;
        valK = cv::Mat::zeros(numJoints * topK, 1, CV_32F);
        cv::Mat indMat = cv::Mat::zeros(numJoints * topK, 1, CV_32S);

        for (int j = 0; j < numJoints; ++j) {
            cv::Mat jointFlat = detFlat.row(j).reshape(1, H*W); // [1, H*W]
            std::vector<std::pair<float, int>> valIdx;

            // 生成值-索引对
            for (int i = 0; i < H*W; ++i) {
                valIdx.emplace_back(jointFlat.at<float>(i), i);
            }

            // 降序排序
            std::sort(valIdx.begin(), valIdx.end(), 
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                });

            // 取Top-K
            for (int k = 0; k < topK; ++k) {
                int idx = static_cast<size_t>(k) < valIdx.size() ? valIdx[k].second : 0;
                float val = static_cast<size_t>(k) < valIdx.size() ? valIdx[k].first : 0.0f;
                valK.at<float>(j * topK + k, 0) = val;
                indMat.at<int>(j * topK + k, 0) = idx;
            }
        }

        // 提取Tag和坐标
        tagK = cv::Mat::zeros(numJoints * topK, tagDim, CV_32F);
        locK = cv::Mat::zeros(numJoints * topK, 2, CV_32S);

        for (int j = 0; j < numJoints; ++j) {
            for (int k = 0; k < topK; ++k) {
                int idx = indMat.at<int>(j * topK + k, 0);
                int x = idx % W;
                int y = idx / W;

                // 存储坐标
                locK.at<int>(j * topK + k, 0) = x;
                locK.at<int>(j * topK + k, 1) = y;

                // 存储Tag - 修复：多维Mat的索引方式
                cv::Mat tagVal;
                if (!mTagPerJoint) {
                    // Tag共享，扩展到所有关节 - 构造Range数组索引4维Mat
                    cv::Range ranges[] = {
                        cv::Range(0, 1),      // numJoints维度
                        cv::Range(y, y+1),    // H维度
                        cv::Range(x, x+1),    // W维度
                        cv::Range::all()      // tagDim维度
                    };
                    tagVal = tag(ranges).reshape(1, tagDim);
                } else {
                    // 构造Range数组索引4维Mat
                    cv::Range ranges[] = {
                        cv::Range(j, j+1),    // numJoints维度
                        cv::Range(y, y+1),    // H维度
                        cv::Range(x, x+1),    // W维度
                        cv::Range::all()      // tagDim维度
                    };
                    tagVal = tag(ranges).reshape(1, tagDim);
                }
                tagVal.copyTo(tagK.row(j * topK + k));
            }
        }

        // Reshape输出（适配MatchByTag输入）
        tagK = tagK.reshape(tagDim, numJoints);
        locK = locK.reshape(2, numJoints);
        valK = valK.reshape(1, numJoints);
    }

    // 调整坐标
    void Adjust(std::vector<cv::Mat>& ans, const cv::Mat& det) {
        int numJoints = det.size[0];
        int H = det.size[1];
        int W = det.size[2];

        for (size_t p = 0; p < ans.size(); ++p) {
            cv::Mat& person = ans[p];
            for (int j = 0; j < numJoints; ++j) {
                float val = person.at<float>(j, 2);
                if (val > 0) {
                    float y = person.at<float>(j, 0);
                    float x = person.at<float>(j, 1);
                    int xx = (int)x;
                    int yy = (int)y;

                    cv::Mat jointDet = det.row(j).reshape(1, H); // [H, W]
                    float right = (yy+1 < W) ? jointDet.at<float>(xx, yy+1) : 0;
                    float left = (yy-1 >= 0) ? jointDet.at<float>(xx, yy-1) : 0;
                    float down = (xx+1 < H) ? jointDet.at<float>(xx+1, yy) : 0;
                    float up = (xx-1 >= 0) ? jointDet.at<float>(xx-1, yy) : 0;

                    // 微调y
                    if (right > left) y += 0.25f;
                    else y -= 0.25f;

                    // 微调x
                    if (down > up) x += 0.25f;
                    else x -= 0.25f;

                    // 偏移0.5
                    person.at<float>(j, 0) = y + 0.5f;
                    person.at<float>(j, 1) = x + 0.5f;
                }
            }
        }
    }

    cv::Mat Refine(const cv::Mat& det, const cv::Mat& tag, cv::Mat keypoints) {
        CV_Assert(det.dims == 2 && tag.dims == 4); // det: [numJoints, H*W], tag: [numJoints, H, W, tagDim]
        int numJoints = det.rows;
        int H = tag.size[1];
        int W = tag.size[2];
        int tagDim = tag.size[3];

        // 计算平均Tag
        std::vector<cv::Mat> tags;
        for (int j = 0; j < numJoints; ++j) {
            float val = keypoints.at<float>(j, 2);
            if (val > 0) {
                int x = (int)keypoints.at<float>(j, 0);
                int y = (int)keypoints.at<float>(j, 1);
                cv::Range ranges[] = {
                    cv::Range(j, j+1),    // numJoints维度
                    cv::Range(y, y+1),    // H维度
                    cv::Range(x, x+1),    // W维度
                    cv::Range::all()      // tagDim维度
                };
                tags.push_back(tag(ranges).reshape(1, tagDim));
            }
        }

        cv::Mat prevTag = cv::Mat::zeros(1, tagDim, CV_32F);
        for (const cv::Mat& t : tags) prevTag += t;
        if (!tags.empty()) {
            prevTag /= (float)tags.size();
        }

        cv::Mat ans = cv::Mat::zeros(numJoints, 3, CV_32F);
        for (int j = 0; j < numJoints; ++j) {
            cv::Mat jointDet = det.row(j).reshape(1, H); // [H, W]
            
            // 修复：多维Mat索引tag(j)
            cv::Range ranges_j[] = {
                cv::Range(j, j+1),    // numJoints维度
                cv::Range::all(),     // H维度
                cv::Range::all(),     // W维度
                cv::Range::all()      // tagDim维度
            };
            cv::Mat jointTagMat = tag(ranges_j);
            cv::Mat jointTag = jointTagMat.reshape(tagDim, H*W); // [H*W, tagDim]

            // 计算Tag距离
            cv::Mat diff = jointTag - prevTag;
            cv::Mat tt;
            cv::norm(diff, tt, cv::NORM_L2, 1); // 每行的L2距离
            tt = tt.reshape(1, H); // [H, W]

            // 代价: det - round(tt)
            cv::Mat cost = jointDet - cv::Mat(tt.size(), CV_32F, cv::Scalar(0));
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    cost.at<float>(y, x) = jointDet.at<float>(y, x) - std::round(tt.at<float>(y, x));
                }
            }

            // 找最大值位置
            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(cost, nullptr, &maxVal, nullptr, &maxLoc);
            int xx = maxLoc.x;
            int yy = maxLoc.y;
            float val = jointDet.at<float>(yy, xx);

            // 偏移0.5
            float x = (float)xx + 0.5f;
            float y = (float)yy + 0.5f;

            // 微调
            float right = (xx+1 < W) ? jointDet.at<float>(yy, xx+1) : 0;
            float left = (xx-1 >= 0) ? jointDet.at<float>(yy, xx-1) : 0;
            float down = (yy+1 < H) ? jointDet.at<float>(yy+1, xx) : 0;
            float up = (yy-1 >= 0) ? jointDet.at<float>(yy-1, xx) : 0;

            if (right > left) x += 0.25f;
            else x -= 0.25f;

            if (down > up) y += 0.25f;
            else y -= 0.25f;

            ans.at<float>(j, 0) = x;
            ans.at<float>(j, 1) = y;
            ans.at<float>(j, 2) = val;
        }

        for (int j = 0; j < numJoints; ++j) {
            if (ans.at<float>(j, 2) > 0 && keypoints.at<float>(j, 2) == 0) {
                keypoints.at<float>(j, 0) = ans.at<float>(j, 0);
                keypoints.at<float>(j, 1) = ans.at<float>(j, 1);
                keypoints.at<float>(j, 2) = ans.at<float>(j, 2);
            }
        }

        return keypoints;
    }

public:
    HeatmapParser() {
        mParams = Params();
        mTagPerJoint = false;
        mNmsKernel = 3;
        mNmsPadding = 1;
    }

    HeatmapParser(const Params& params_, bool tagPerJoint_, int nmsKernel_ = 3, int nmsPadding_ = 1) {
        mParams = params_;
        mTagPerJoint = tagPerJoint_;
        mNmsKernel = nmsKernel_;
        mNmsPadding = nmsPadding_;
    }

    std::vector<cv::Mat> Match(const cv::Mat& tagK, const cv::Mat& locK, const cv::Mat& valK) {
        return MatchByTag(tagK, locK, valK, mParams);
    }

    std::pair<std::vector<cv::Mat>, std::vector<float>> Parse(const cv::Mat& det, const cv::Mat& tag, 
                                                              bool adjust = true, bool refine = true) {
        // Step1: Top-K提取（单张图）
        cv::Mat tagK, locK, valK;
        TopK(det, tag, tagK, locK, valK);

        // Step2: Tag匹配（单张图）
        std::vector<cv::Mat> ans = Match(tagK, locK, valK);

        // Step3: 坐标调整（单张图）
        if (adjust) {
            this->Adjust(ans, det);
        }

        // Step4: 计算平均置信度
        std::vector<float> scores;
        for (const cv::Mat& person : ans) {
            cv::Mat vals = person.col(2);
            scores.push_back((float)cv::mean(vals)[0]);
        }

        // Step5: 关键点细化（单张图）
        if (refine) {
            cv::Mat detNumpy = det.reshape(1, det.size[0]); // [numJoints, H*W]
            cv::Mat tagNumpy = tag; // [numJoints, H, W, tagDim]

            if (!mTagPerJoint) {
                // 扩展Tag到所有关节 - 修复4维Mat创建
                int sz_exp[] = {mParams.NumJoints, tagNumpy.size[1], tagNumpy.size[2], tagNumpy.size[3]};
                cv::Mat tagExpanded = cv::Mat::zeros(4, sz_exp, CV_32F);
                
                for (int j = 0; j < mParams.NumJoints; ++j) {
                    // 修复：多维Mat索引tagExpanded(j)
                    cv::Range ranges_exp[] = {
                        cv::Range(j, j+1),    // numJoints维度
                        cv::Range::all(),     // H维度
                        cv::Range::all(),     // W维度
                        cv::Range::all()      // tagDim维度
                    };
                    tagNumpy.copyTo(tagExpanded(ranges_exp));
                }
                tagNumpy = tagExpanded;
            }

            for (size_t p = 0; p < ans.size(); ++p) {
                ans[p] = Refine(detNumpy, tagNumpy, ans[p]);
            }
        }

        return {ans, scores};
    }
};

static Mat GetAffineMatrix(const Point2f& center, float scaleW, float scaleH, const Size& outputSize) {
    float dstW = outputSize.width;
    float dstH = outputSize.height;
    float srcW = scaleW * 200.0f;  // 基于人体尺度的原始宽度
    float srcH = scaleH * 200.0f;  // 基于人体尺度的原始高度

    // 缩放+平移矩阵（保证中心对齐）
    Mat trans = (Mat_<float>(2, 3) <<
        dstW / srcW, 0, dstW * (-center.x / srcW + 0.5f),
        0, dstH / srcH, dstH * (-center.y / srcH + 0.5f)
    );
    return trans;
}

std::tuple<Size, Point2f, std::pair<float, float>> GetScaleSize(const Mat& image) {
    int h = image.rows;
    int w = image.cols;
    
    // 图像中心（四舍五入）
    Point2f center(static_cast<int>(w / 2.0f + 0.5f), static_cast<int>(h / 2.0f + 0.5f));

    // 最小输入尺寸（短边），确保是64的倍数
    const int minInputSize = 512;
    int wResized, hResized;
    float scaleW, scaleH;

    if (w < h) {  // 短边为宽
        wResized = minInputSize;
        // 按比例+64倍数约束
        hResized = static_cast<int>(minInputSize / (float)w * h + 63) / 64 * 64;
        scaleW = w / 200.0f;
        scaleH = (hResized / (float)wResized) * (w / 200.0f);
    } else {  // 短边为高
        hResized = minInputSize;
        wResized = static_cast<int>(minInputSize / (float)h * w + 63) / 64 * 64;
        scaleH = h / 200.0f;
        scaleW = (wResized / (float)hResized) * (h / 200.0f);
    }

    return {Size(wResized, hResized), center, {scaleW, scaleH}};
}

std::pair<Mat, Mat> GetMultiStageOutputs(const std::vector<Mat>& outputs, const Size& sizeProjected) {
    // 插值到最后一个输出的尺寸
    Mat output;
    cv::resize(outputs[0], output, Size(outputs.back().cols, outputs.back().rows), 0, 0, cv::INTER_LINEAR);

    const int offsetFeat = 17;
    // 分离通道：热图和标签
    std::vector<Mat> outputChannels;
    cv::split(output, outputChannels);
    
    std::vector<Mat> heatmapChannels1, tagChannels;
    for (int i = 0; i < offsetFeat; i++) {
        heatmapChannels1.push_back(outputChannels[i]);
    }
    for (size_t i = offsetFeat; i < outputChannels.size(); i++) {
        tagChannels.push_back(outputChannels[i]);
    }

    Mat heatmapChannels;
    cv::merge(heatmapChannels1, heatmapChannels);
    Mat heatmaps = (heatmapChannels + outputs.back()) / 2.0f;;

    // 标签合并
    Mat tags;
    cv::merge(tagChannels, tags);

    // 插值到目标尺寸
    cv::resize(heatmaps, heatmaps, sizeProjected, 0, 0, cv::INTER_LINEAR);
    cv::resize(tags, tags, sizeProjected, 0, 0, cv::INTER_LINEAR);

    return {heatmaps, tags};
}

Mat ReadChwBinToMat(const std::string& binPath, const std::vector<int>& shape) {
    std::ifstream file(binPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开bin文件: " + binPath);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取所有数据
    std::vector<float> data(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);

    // shape格式：[batch, channel, height, width] → 转换为OpenCV Mat (height, width, channel)
    int batch = shape[0];
    int channel = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    Mat chwMat(channel, height * width, CV_32FC1);
    std::memcpy(chwMat.data, data.data(), height * width * channel * sizeof(float));

    std::vector<cv::Mat> channels;
    for (int c = 0; c < channel; ++c) {
        // 获取第c个通道的 (H, W) 数据
        cv::Mat channel_mat(height, width, CV_32FC1, chwMat.ptr<float>(c));
        channels.push_back(channel_mat.clone()); // 可能需要clone以确保数据连续
    }
    Mat result;
    merge(channels, result); // 将C个单通道Mat合并成一个 (H, W, C) 的多通道Mat

    return result;
}

bool HrnetPostprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    LOG(INFO) << fileList[0] << " save result";
    if (tensorBufs.size() != 2 || fileList.size() != 1) {
        return false;
    }
    static std::string resultPath = "./result";
    static std::string binPath = "./result/bin";
    static std::string txtPath = "./result/txt";
    struct stat info;
    if (stat(resultPath.c_str(), &info) != 0) {
        mkdir(resultPath.c_str(), 0777);
    }
    if (stat(binPath.c_str(), &info) != 0) {
        mkdir(binPath.c_str(), 0777);
    }
    if (stat(txtPath.c_str(), &info) != 0) {
        mkdir(txtPath.c_str(), 0777);
    }
    size_t start = fileList[0].find_last_of("/");
    size_t end = fileList[0].find_last_of(".");
    std::string fileName = fileList[0].substr(start , end-start);

    for (size_t i = 0; i < tensorBufs.size(); ++i) {
        std::string outputPath = binPath + "/" + fileName + "_" + to_string(i) + ".bin";
        LOG(INFO) << outputPath;
        std::ofstream ofs(outputPath, std::ios::out | std::ios::trunc);
        if (!ofs) {
            LOG(ERROR) << "Failed to open file: " << outputPath;
            continue;
        }
        ofs.write(reinterpret_cast<char*>(tensorBufs[i].GetRawPtr()), tensorBufs[i].size);
    }

    Mat image = imread(fileList[0]);
    if (image.empty()) {
        return false;
    }
    // 计算缩放尺寸、中心、缩放因子
    Size outputSize;
    Point2f center;
    std::pair<float, float> scalePair;
    std::tie(outputSize, center, scalePair) = GetScaleSize(image);
    float scaleW = scalePair.first;
    float scaleH = scalePair.second;

    // 计算仿射变换矩阵
    Mat trans = GetAffineMatrix(center, scaleW, scaleH, outputSize);

    std::vector<Mat> outputs;
    for (int i = 0; i < 2; i++) {
        std::string binPathResult = binPath + fileName + "_" + std::to_string(i) + ".bin";
        // 计算bin文件的shape: [1, 17*(2-i), 128*(i+1), 192*(i+1)]
        std::vector<int> shape = {
            1, 
            17 * (2 - i), 
            128 * (i + 1), 
            192 * (i + 1)
        };
        Mat output = ReadChwBinToMat(binPathResult, shape);
        outputs.push_back(output);
    }
    Mat invTrans;
    cv::invertAffineTransform(trans, invTrans);
    // 多阶段输出处理
    Mat heatmaps;
    Mat tags;
    std::tie(heatmaps, tags) = GetMultiStageOutputs(outputs, outputSize);

    cv::Mat reshapedMat = heatmaps.reshape(1, 17 * outputSize.height); 
    if (!reshapedMat.isContinuous()) {
        reshapedMat = reshapedMat.clone();
    }
    int detDims[] = {17, outputSize.height, outputSize.width};
    cv::Mat detMat(3, detDims, CV_32F, reshapedMat.data);
    detMat = detMat.clone();

    reshapedMat = tags.reshape(1, 17 * outputSize.height);
    if (!reshapedMat.isContinuous()) {
        reshapedMat = reshapedMat.clone();
    }
    int tag_sizes[] = {17, outputSize.height, outputSize.width, 1};
    cv::Mat tagMat(4, tag_sizes, CV_32F, reshapedMat.data);
    tagMat = tagMat.clone();
    return true;
}
}