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

#include "furthest_point_sampling_tiling.h"

namespace ops {
class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT};
        const std::vector<ge::DataType> intTypes = {ge::DT_INT32};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND};

        this->Attr("nsample").AttrType(REQUIRED).Int();
        this->Input("xyz_trans")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .ParamType(REQUIRED);
        this->Input("dist")
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .ParamType(REQUIRED);
        this->Output("out_idx")
            .DataType(intTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .ParamType(REQUIRED);
        this->AICore().SetTiling(optiling::TilingFunctionForFurthestPointSampling);
        OpAICoreConfig opCfg;
        opCfg.DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicFormatFlag(true)
            .DynamicCompileStaticFlag(true);
        this->AICore().AddConfig("ascend310p", opCfg);
    }
};

OP_ADD(FurthestPointSampling);
}