 /*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd
 * This file is part of [Hispark/modelzoo].
 *
 * [Hispark/modelzoo] is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, version 3 of the License only.
 *
 * [Hispark/modelzoo] is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with [Hispark/modelzoo].  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef MODEL_PROCESS_H
#define MODEL_PROCESS_H

#include <cstdint>
#include <iostream>
#include <vector>
#include "utils.h"
#include "acl/svp_acl.h"

class ModelProcess {
public:
    /**
    * Function: Class constructor
    * Description: Initialize a ModelProcess instance with default resource states
    */
    ModelProcess();

    /**
    * Function: Class destructor
    * Description: Release all allocated resources and destroy the ModelProcess instance
    */
    ~ModelProcess();

    /**
    * Function: Load model file into memory and initialize model resources
    * Input: modelPath - full file path of the target model file (including file name and extension)
    * Return: Result status code (success if the code indicates normal execution, failure otherwise)
    */
    Result LoadModelFromFileWithMem(const std::string& modelPath);

    /**
    * Function: Unload loaded model and release associated memory
    * Description: Clear model-related resources without returning any status
    */
    void Unload();

    /**
    * Function: Initialize input dataset for model inference
    * Return: Result status code (success/failure of initialization)
    */
    Result InitInput();

    /**
    * Function: Create model description structure
    * Description: Build and initialize the model descriptor to store model metadata
    * Return: Result status code (success/failure of descriptor creation)
    */
    Result CreateDesc();

    /**
    * Function: Destroy model description structure
    * Description: Release memory occupied by the model descriptor and clear related data
    */
    void DestroyDesc();

    /**
    * Function: Create input buffer for model inference
    * Input: inputDataBuffer - pointer to the host/device buffer containing input data
    * Input: bufferSize - total byte size of the inputDataBuffer
    * Input: stride - alignment step size for input data processing
    * Return: Result status code (success/failure of input buffer creation)
    */
    Result CreateInput(void *inputDataBuffer, size_t bufferSize, int stride);

    /**
    * Function: Create input buffer from a specified file
    * Input: filePath - full path of the file containing input data
    * Return: Result status code (success/failure of buffer creation)
    */
    Result CreateInputBuf(const std::string& filePath);

    /**
    * Function: Create task buffer and work buffer for model execution
    * Return: Result status code (success/failure of buffer allocation)
    */
    Result CreateTaskBufAndWorkBuf();

    /**
    * Function: Release all resources related to model input
    * Description: Free input buffers and clear input dataset configuration
    */
    void DestroyInput();

    /**
    * Function: Allocate output buffer to store model inference results
    * Return: Result status code (success/failure of output buffer creation)
    */
    Result CreateOutput();

    /**
    * Function: Release all resources related to model output
    * Description: Free output buffers and clear inference result data
    */
    void DestroyOutput();

    /**
    * Function: Execute model inference with prepared input data
    * Return: Result status code (success/failure of inference execution)
    */
    Result Execute();

    /**
    * Function: Export model inference output results to a file
    * Description: Write output buffer data to a local file (read-only operation, no state change)
    */
    void DumpModelOutputResult() const;

    /**
    * Function: Retrieve and display model inference output results
    * Description: Extract data from output buffer and present it (read-only operation, no state change)
    */
    void OutputModelResult() const;

    /**
    * Function: Create dedicated buffer for specified input/output index
    * Input: index - target index of the input/output buffer (starting from 0)
    * Return: Result status code (success/failure of buffer creation)
    */
    Result CreateBuf(int index);

    // Result GetInputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const;

    /**
    * Function: Get stride parameters of specified output index
    * Input: index - target index of the output buffer (starting from 0)
    * Output: bufSize - byte size of the output buffer (output parameter)
    * Output: stride - alignment step size of the output data (output parameter)
    * Output: dims - dimension information of the output model I/O (output parameter)
    * Return: Result status code (success/failure of parameter retrieval)
    */
    Result GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const;

    /**
    * Function: Get data size of specified input index
    * Input: index - target index of the input buffer (starting from 0)
    * Return: Total byte size of the input buffer at the specified index
    */
    size_t GetInputDataSize(int index) const;

    /**
    * Function: Get data size of specified output index
    * Input: index - target index of the output buffer (starting from 0)
    * Return: Total byte size of the output buffer at the specified index
    */
    size_t GetOutputDataSize(int index) const;

    /**
    * Function: Get the total number of model input nodes
    * Return: Count of input nodes (non-negative integer)
    */
    size_t GetInputNum() const;

    /**
    * Function: Get the total number of model output nodes
    * Return: Count of output nodes
    */
    size_t GetOutputNum() const;

    /**
    * Function: Get stride parameters of specified input index
    * Input: index - target index of the input buffer (starting from 0)
    * Output: buf_size - byte size of the input buffer (output parameter)
    * Output: stride - alignment step size of the input data (output parameter)
    * Output: dims - dimension information of the input model I/O (output parameter)
    * Return: Result status code (success/failure of parameter retrieval)
    */
    Result GetInputStrideParam(int index, size_t& buf_size, size_t& stride, svp_acl_mdl_io_dims& dims) const;

    /**
    * Function: Create model input buffer from raw data
    * Input: data - pointer to the raw input data buffer
    * Input: data_size - total byte size of the raw data buffer
    * Return: Result status code (success/failure of input buffer creation)
    */
    Result CreateInputFromData(const void* data, size_t data_size); 

    /**
    * Function: Create model input buffers from multiple raw data sources
    * Input: input_datas - vector of pointers to multiple raw input data buffers
    * Input: input_sizes - vector of byte sizes corresponding to each input data buffer
    * Return: Result status code (success/failure of multi-input buffer creation)
    */
    Result CreateInputFromData(const std::vector<const void*>& input_datas, 
                               const std::vector<size_t>& input_sizes);

    /**
    * Function: Extract a compact output tensor without stride padding
    * Input: index - output index
    * Output: packed - compact output bytes
    * Output: dims - logical dimensions reported by the model
    * Output: elem_type - worker element type used by the binary protocol
    * Return: Result status code (success/failure of extraction)
    */
    Result GetPackedOutputData(
        size_t index,
        std::vector<uint8_t>& packed,
        std::vector<int64_t>& dims,
        uint32_t& elem_type) const;
    
    /**
    * Function: Release all model-related resources
    * Description: Unified release of input/output buffers, model descriptor, task/work buffers, etc.
    */
    void DestroyResource();

private:
    void WriteOutput(const std::string& outputFileName, size_t index) const;

    Result ClearOutputStrideInvalidBuf(std::vector<int8_t>& buffer, size_t index) const;

    uint32_t executeNum_ { 0 };
    uint32_t modelId_ { 0 };
    size_t modelMemSize_ { 0 };
    size_t modelWeightSize_ { 0 };
    void *modelMemPtr_ { nullptr };
    void *modelWeightPtr_ { nullptr };
    bool loadFlag_ { false };
    svp_acl_mdl_desc *modelDesc_ { nullptr };
    svp_acl_mdl_dataset *input_ { nullptr };
    svp_acl_mdl_dataset *output_ { nullptr };
};

#endif // MODEL_PROCESS_H
