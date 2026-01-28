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

#include "proto/onnx/ge_onnx.pb.h"
#include "register/register.h"
#include "graph/operator.h"

using namespace ge;

namespace domi {

static const int ATTR_NUM = 1;

Status ParseOnnxInputParamsFurthestPointSampling(const Message *opMsg, Operator &opDest)
{
    const onnx::NodeProto *node = reinterpret_cast<const onnx::NodeProto *>(opMsg);
    if (node == nullptr) {
        return FAILED;
    }

    int nSample = 1;

    int attr_num = 0;
    for (const auto& attr: node->attribute()) {
        if (attr.name() == "nsample" && attr.type() == onnx::AttributeProto::INT) {
            nSample = attr.i();
            attr_num++;
        }
    }

    if (attr_num != ATTR_NUM) {
        return FAILED;
    }

    opDest.SetAttr("nsample", nSample);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("FurthestPointSampling")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::FurthestPointSampling"),
        ge::AscendString("ai.onnx::11::FurthestPointSampling"),
        ge::AscendString("ai.onnx::12::FurthestPointSampling"),
        ge::AscendString("ai.onnx::13::FurthestPointSampling"),
        ge::AscendString("ai.onnx::14::FurthestPointSampling"),
        ge::AscendString("ai.onnx::15::FurthestPointSampling"),
        ge::AscendString("ai.onnx::16::FurthestPointSampling"),
        ge::AscendString("ai.onnx::17::FurthestPointSampling"),
        ge::AscendString("ai.onnx::18::FurthestPointSampling")})
    .ParseParamsFn(ParseOnnxInputParamsFurthestPointSampling)
    .ImplyType(ImplyType::TVM);
} // domi

