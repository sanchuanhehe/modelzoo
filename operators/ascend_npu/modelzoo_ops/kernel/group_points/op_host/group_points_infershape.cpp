/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "group_points_tiling.h"
#include "register/op_def_registry.h"

static constexpr uint32_t BATCH_IDX = 0;
static constexpr uint32_t CHANNEL_IDX = 2;
static constexpr uint32_t NUM_POINT_IDX = 1;
static constexpr uint32_t NUM_SAMPLE_IDX = 2;

static constexpr uint32_t OUTPUT_SHAPE_DIM = 2;

using namespace ge;
namespace ops {
static graphStatus GroupPointsInferShape(gert::InferShapeContext* inferContext)
{
    gert::Shape* outShape = inferContext->GetOutputShape(0);
    if (outShape == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *featuresShape = inferContext->GetInputShape(0);
    const gert::Shape *indexShape = inferContext->GetInputShape(1);

    auto batch = featuresShape->GetDim(BATCH_IDX);
    auto channel = featuresShape->GetDim(CHANNEL_IDX);
    auto numPoints = indexShape->GetDim(NUM_POINT_IDX);
    auto numSample = indexShape->GetDim(NUM_SAMPLE_IDX);

    outShape->SetDimNum(OUTPUT_SHAPE_DIM);
    *outShape = {batch * numPoints * numSample, channel};
    return GRAPH_SUCCESS;
}

static graphStatus GroupPointsInferDataType(gert::InferDataTypeContext* inferContext)
{
    inferContext->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GroupPoints).InferShape(GroupPointsInferShape).InferDataType(GroupPointsInferDataType);
}