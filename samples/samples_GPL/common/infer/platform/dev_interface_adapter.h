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

#pragma once
#include <stdint.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "log.h"

namespace Infer {
#define MAX_TENSOR_DIM 128

enum TensorType { 
   TENSOR_TYPE_UNDEFINED = -1,
   TENSOR_TYPE_FLOAT = 0, 
   TENSOR_TYPE_FLOAT16 = 1, 
   TENSOR_TYPE_INT8 = 2, 
   TENSOR_TYPE_INT32 = 3, 
   TENSOR_TYPE_UINT8 = 4, 
   TENSOR_TYPE_INT16 = 6, 
   TENSOR_TYPE_UINT16 = 7, 
   TENSOR_TYPE_UINT32 = 8, 
   TENSOR_TYPE_INT64 = 9, 
   TENSOR_TYPE_UINT64 = 10, 
   TENSOR_TYPE_DOUBLE = 11, 
   TENSOR_TYPE_BOOL = 12, 
   TENSOR_TYPE_STRING = 13,
   TENSOR_TYPE_COMPLEX64 = 16, 
   TENSOR_TYPE_COMPLEX128 = 17,
   TENSOR_TYPE_INT10 = 100,
   TENSOR_TYPE_UINT10 = 101,
   TENSOR_TYPE_INT12 = 102,
   TENSOR_TYPE_UINT12 = 103,
   TENSOR_TYPE_INT14 = 104,
   TENSOR_TYPE_UINT14 = 105,
   TENSOR_TYPE_INT24 = 106,
   TENSOR_TYPE_UINT24 = 107
};

enum TensorFormat {
   TENSOR_Format_UNDEFINED = -1, 
   TENSOR_Format_NCHW = 0, 
   TENSOR_Format_NHWC = 1, 
   TENSOR_Format_ND = 2, 
   TENSOR_Format_NC1HWC0 = 3, 
   TENSOR_Format_FRACTAL_Z = 4, 
   TENSOR_Format_NC1HWC0_C04 = 12, 
   TENSOR_Format_NDHWC = 27, 
   TENSOR_Format_FRACTAL_NZ = 29, 
   TENSOR_Format_NCDHW = 30, 
   TENSOR_Format_NDC1HWC0 = 32, 
   TENSOR_Format_FRACTAL_Z_3D = 33
};

struct TensorDesc {
   size_t dimCount = 0;
   size_t dims[MAX_TENSOR_DIM] = {0};
   TensorType type = TENSOR_TYPE_UNDEFINED;
   size_t typeSize = 0; // 单位bit
   TensorFormat format = TENSOR_Format_UNDEFINED;
   size_t defaultStride = 0;
   size_t defaultSize = 0;
};

struct TensorBuf {
   std::shared_ptr<void> data;
   size_t size = 0;
   size_t stride = 0;
   TensorBuf():TensorBuf(0, 0) {}
   TensorBuf(size_t dataSize, size_t dataStride);
   TensorBuf(void* externalData, size_t dataSize, size_t dataStride);
   TensorBuf DeepCopy();
   void ResetData(void* newData = nullptr);
   void* GetRawPtr() const { return data.get(); }
};

enum RunMode {
   Sync,
   Async
};

int32_t DevInit(const std::string& configPath = "");
int32_t DevDeInit();

int32_t DevMalloc(void **devPtr, size_t size);
int32_t DevFlush(void *devPtr, size_t size);
int32_t DevFree(void *devPtr);
int32_t DevMemcpy(void *dst, size_t destMax, const void *src, size_t count);

class MdlBase {
public:
   virtual ~MdlBase() = default;
   virtual int32_t LoadModel(const std::string& modelPath) = 0;
   virtual int32_t LoadModel(const char* modelBuf, size_t modelSize) = 0;
   virtual int32_t UnLoadModel() = 0;

   virtual size_t GetInTensorNum() = 0;
   virtual int32_t GetInTensorDescByIdx(size_t index, TensorDesc& tensorDesc) = 0;
   virtual int32_t GetInTensorDescByName(const std::string& name, TensorDesc& tensorDesc) = 0;
   virtual size_t GetOutTensorNum() = 0;
   virtual int32_t GetOutTensorDescByIdx(size_t index, TensorDesc& tensorDesc) = 0;
   virtual int32_t GetOutTensorDescByName(const std::string& name, TensorDesc& tensorDesc) = 0;

   virtual int32_t Execute(std::vector<TensorBuf>& inBufs, std::vector<TensorBuf>& outBufs,
      RunMode runMode = RunMode::Sync) = 0;
   virtual int32_t Wait() = 0;

protected:
   void PrintModelDesc()
   {
      TensorDesc tensorDesc;
      for (size_t i = 0; i < GetInTensorNum(); i++) {
         GetInTensorDescByIdx(i, tensorDesc);
         std::string dimsStr = "[";
         for (size_t j = 0; j < tensorDesc.dimCount; j++) {
               if (j + 1 == tensorDesc.dimCount) {
                  dimsStr =  dimsStr + std::to_string(tensorDesc.dims[j]) + "]";
                  break;
               }
               dimsStr =  dimsStr + std::to_string(tensorDesc.dims[j]) + "*";
         }
         LOG(INFO) << "input tensor " << i << " info : dims = " << dimsStr << ", dataType = " << tensorDesc.type
               << ",  dataSize = " << tensorDesc.typeSize << "bits, dataFormat = " << tensorDesc.format <<
               ", stride = " <<  tensorDesc.defaultStride << ", size = " << tensorDesc.defaultSize;
      }

      for (size_t i = 0; i < GetOutTensorNum(); i++) {
         GetOutTensorDescByIdx(i, tensorDesc);
         std::string dimsStr = "[";
         for (size_t j = 0; j < tensorDesc.dimCount; j++) {
               if (j + 1 == tensorDesc.dimCount) {
                  dimsStr =  dimsStr + std::to_string(tensorDesc.dims[j]) + "]";
                  break;
               }
               dimsStr =  dimsStr + std::to_string(tensorDesc.dims[j]) + "*";
         }
         LOG(INFO) << "output tensor " << i << " info : dims = " << dimsStr << ", dataType = " << tensorDesc.type
               << ",  dataSize = " << tensorDesc.typeSize << "bits, dataFormat = " << tensorDesc.format <<
               ", stride = " <<  tensorDesc.defaultStride << ", size = " << tensorDesc.defaultSize;
      }
   }
};

std::shared_ptr<MdlBase> MdlCreate();
}
