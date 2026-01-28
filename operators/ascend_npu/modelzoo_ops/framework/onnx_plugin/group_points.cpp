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
static const int ATTR_NUM = 0;

Status ParseOnnxInputParamsGroupPoints(const Message *opMsg, Operator &opDest)
{
    const onnx::NodeProto *node = reinterpret_cast<const onnx::NodeProto *>(opMsg);
    if (node == nullptr) {
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("GroupPoints")
    .FrameworkType(ONNX)
    .OriginOpType({
        ge::AscendString("npu::1::GroupPoints"),
        ge::AscendString("ai.onnx::11::GroupPoints"),
        ge::AscendString("ai.onnx::12::GroupPoints"),
        ge::AscendString("ai.onnx::13::GroupPoints"),
        ge::AscendString("ai.onnx::14::GroupPoints"),
        ge::AscendString("ai.onnx::15::GroupPoints"),
        ge::AscendString("ai.onnx::16::GroupPoints"),
        ge::AscendString("ai.onnx::17::GroupPoints"),
        ge::AscendString("ai.onnx::18::GroupPoints")})
    .ParseParamsFn(ParseOnnxInputParamsGroupPoints)
    .ImplyType(ImplyType::TVM);
} // domi

