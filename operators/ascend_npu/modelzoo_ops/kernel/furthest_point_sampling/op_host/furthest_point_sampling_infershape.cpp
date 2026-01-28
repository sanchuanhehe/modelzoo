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

#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

using namespace ge;
namespace ops {
static constexpr int64_t POINTS_DIM = 3;
static constexpr int64_t OUT_DIM = 2;

static graphStatus FurthestPointSamplingInferShape(gert::InferShapeContext *inferContext)
{
    gert::Shape *outShape = inferContext->GetOutputShape(0);
    if (outShape == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *inputShape = inferContext->GetInputShape(0);
    const gert::RuntimeAttrs *attr = inferContext->GetAttrs();
    if (inputShape == nullptr || attr == nullptr || attr->GetAttrPointer<int32_t>(0) == nullptr) {
        return GRAPH_FAILED;
    }

    uint32_t nSample = *(attr->GetAttrPointer<int32_t>(0));
    uint32_t b = inputShape->GetDim(0);

    outShape->SetDimNum(OUT_DIM);
    *outShape = {b, nSample};

    return GRAPH_SUCCESS;
}

static graphStatus FurthestPointSamplingInferDataType(gert::InferDataTypeContext *inferContext)
{
    inferContext->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(FurthestPointSampling).InferShape(FurthestPointSamplingInferShape).InferDataType(FurthestPointSamplingInferDataType);
}