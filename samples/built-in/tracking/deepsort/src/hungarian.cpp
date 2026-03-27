/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#include "hungarian.h"
#include <cmath>
#include <limits>

// 这是用于 DeepSORT 在 NPU 板端执行的一种基于贪心法则保持快速执行的二分图匹配版本
// 以应对标准的 O(N^3) 级匈牙利算法在特征成本矩阵庞大时引起的性能剧烈衰退
double HungarianAlgorithm::Solve(std::vector<std::vector<double>> &distMatrix,
                                 std::vector<int> &assignment)
{
    int nRows = distMatrix.size();
    if (nRows == 0)
        return 0.0;
    int nCols = distMatrix[0].size();
    if (nCols == 0)
        return 0.0;

    assignment.clear();
    assignment.assign(nRows, -1);

    double cost = 0.0;
    std::vector<bool> colCovered(nCols, false);
    std::vector<bool> rowCovered(nRows, false);

    // 基于最小局部成本的贪心快速匹配
    for (int r = 0; r < nRows; r++) {
        double minVal = std::numeric_limits<double>::max();
        int minCol = -1;
        for (int c = 0; c < nCols; c++) {
            if (!colCovered[c] && distMatrix[r][c] < minVal) {
                minVal = distMatrix[r][c];
                minCol = c;
            }
        }
        if (minCol != -1) {
            assignment[r] = minCol;
            colCovered[minCol] = true;
            rowCovered[r] = true;
            cost += minVal;
        }
    }

    return cost;
}
