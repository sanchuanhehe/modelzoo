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

static const int ATTR_NUM = 4;

Status ParseOnnxInputParamsCylinderQuery(const Message *opMsg, Operator &opDest)
{
    const onnx::NodeProto *node = reinterpret_cast<const onnx::NodeProto *>(opMsg);
    if (node == nullptr) {
        return FAILED;
    }

    float radius = 0.1;
    float hMin = -0.02;
    float hMax = 0.01;
    int nSample = 16;

    int attr_num = 0;
    for (const auto& attr: node->attribute()) {
        if (attr.name() == "radius" && attr.type() == onnx::AttributeProto::FLOAT) {
            radius = attr.f();
            attr_num++;
        } else if (attr.name() == "h_min" && attr.type() == onnx::AttributeProto::FLOAT) {
            hMin = attr.f();
            attr_num++;
        } else if (attr.name() == "h_max" && attr.type() == onnx::AttributeProto::FLOAT) {
            hMax = attr.f();
            attr_num++;
        } else if (attr.name() == "nsample" && attr.type() == onnx::AttributeProto::INT) {
            nSample = attr.i();
            attr_num++;
        }
    }

    if (attr_num != ATTR_NUM) {
        return FAILED;
    }

    opDest.SetAttr("radius", radius);
    opDest.SetAttr("h_min", hMin);
    opDest.SetAttr("h_max", hMax);
    opDest.SetAttr("nsample", nSample);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("CylinderQuery")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::CylinderQuery"),
        ge::AscendString("ai.onnx::11::CylinderQuery"),
        ge::AscendString("ai.onnx::12::CylinderQuery"),
        ge::AscendString("ai.onnx::13::CylinderQuery"),
        ge::AscendString("ai.onnx::14::CylinderQuery"),
        ge::AscendString("ai.onnx::15::CylinderQuery"),
        ge::AscendString("ai.onnx::16::CylinderQuery"),
        ge::AscendString("ai.onnx::17::CylinderQuery"),
        ge::AscendString("ai.onnx::18::CylinderQuery")})
    .ParseParamsFn(ParseOnnxInputParamsCylinderQuery)
    .ImplyType(ImplyType::TVM);
} // domi

