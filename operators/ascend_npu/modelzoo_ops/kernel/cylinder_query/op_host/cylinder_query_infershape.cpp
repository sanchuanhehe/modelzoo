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

static constexpr uint32_t GROUP_IDX_SHAPE_DIM = 3;

static graphStatus CylinderQueryInferShape(gert::InferShapeContext* inferContext)
{
    gert::Shape *outShape = inferContext->GetOutputShape(0);
    if (outShape == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape *seedXyzShape = inferContext->GetInputShape(0);
    const gert::Shape *pointXyzShape = inferContext->GetInputShape(1);

    auto b = seedXyzShape->GetDim(0);
    auto numQueries = seedXyzShape->GetDim(1);
    auto numPoints = pointXyzShape->GetDim(1);

    outShape->SetDimNum(GROUP_IDX_SHAPE_DIM);
    *outShape = {b, numQueries, numPoints};

    return GRAPH_SUCCESS;
}

static graphStatus CylinderQueryInferDataType(gert::InferDataTypeContext* inferContext)
{
    inferContext->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CylinderQuery).InferShape(CylinderQueryInferShape).InferDataType(CylinderQueryInferDataType);
}