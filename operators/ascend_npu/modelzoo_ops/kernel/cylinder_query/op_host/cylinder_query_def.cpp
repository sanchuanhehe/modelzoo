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
#include "cylinder_query_tiling.h"

using namespace ge;
namespace ops {
class CylinderQuery : public OpDef {
public:
    explicit CylinderQuery(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND};

        this->Attr("radius").AttrType(REQUIRED).Float();
        this->Attr("h_min").AttrType(REQUIRED).Float();
        this->Attr("h_max").AttrType(REQUIRED).Float();
        this->Attr("nsample").AttrType(REQUIRED).Int();
        this->Input("seed_xyz")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous()
            .ParamType(REQUIRED);
        this->Input("point_xyz")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous()
            .ParamType(REQUIRED);
        this->Input("rot_mat")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous()
            .ParamType(REQUIRED);
        this->Input("default_idx")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous()
            .ParamType(REQUIRED);

        this->Output("out")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .ParamType(REQUIRED);

        this->AICore().SetTiling(optiling::TilingFunctionForCylinderQuery);
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(CylinderQuery);
}  // namespace ops