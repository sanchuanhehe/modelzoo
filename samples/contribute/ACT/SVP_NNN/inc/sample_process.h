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

#ifndef SAMPLE_PROCESS_H
#define SAMPLE_PROCESS_H

#include <cstdint>
#include <vector> 
#include "utils.h"
#include "acl/svp_acl.h"
#include "model_process.h"
#include "worker_types.h"

/**
* Class: SampleProcess
* Description: Core processing class for model inference sample, manages input data, model loading and resource lifecycle
*/
class SampleProcess {
public:
    /**
    * Function: Class constructor
    * Description: Initialize a SampleProcess instance with default values for all member variables
    */
    SampleProcess();

    /**
    * Function: Class destructor
    * Description: Release allocated resources and reset member variable states when instance is destroyed
    */
    ~SampleProcess();

    /**
    * Function: Initialize system and model inference related resources
    * Description: Set up device context, stream and basic runtime environment
    * Return: Result status code (success if the code indicates normal execution, failure otherwise)
    */
    Result InitResource();

    /**
    * Function: Execute the full process of sample inference
    * Description: Orchestrate input data preparation, model loading and inference execution
    * Return: Result status code (success/failure of the entire inference process)
    */
    Result Process();
    InferenceResponse ProcessOnce(
        const std::vector<const void*>& input_datas,
        const std::vector<size_t>& input_sizes,
        uint32_t request_id);
    // void DestroyResource();

    /**
    * Function: Set the file path of the input data source
    * Input: path - full file path of the input data file (including file name and extension)
    */
    void SetInputPath(const std::string& path);
    std::string input_path_;  // Member variable: Stores the full file path of the input data for inference

    const char* input_data_ = nullptr;  // Member variable: Pointer to the binary input data buffer (nullptr by default)
    size_t input_data_size_ = 0;        // Member variable: Total byte size of the binary input data buffer (0 by default)

    /**
    * Function: Set multiple input data sources for multi-input model inference
    * Input: input_datas - Vector of pointers to multiple binary input data buffers
    * Input: input_sizes - Vector of byte sizes corresponding to each input data buffer
    */
    void SetInputDatas(const std::vector<const void*>& input_datas, 
                       const std::vector<size_t>& input_sizes);
    std::vector<const void*> input_datas_;  // Member variable: Stores pointers to multiple input data buffers
    std::vector<size_t> input_sizes_;       // Member variable: Stores byte sizes of corresponding multiple input data buffers

    /**
    * Function: Load pre-configured model file into memory
    * Description: Initialize model resources and prepare for inference execution
    * Return: Result status code (success/failure of model loading)
    */
    Result LoadModel();

    /**
    * Function: Release all allocated resources
    * Description: Unified release of device context, stream, model resources and input/output buffers
    */
    void DestroyResource();

private:
    int32_t deviceId_ { 0 };  // Member variable: ID of the target computing device (default to 0)
    svp_acl_rt_context context_ { nullptr };  // Member variable: Runtime context of the computing device (nullptr by default)
    svp_acl_rt_stream stream_ { nullptr };    // Member variable: Runtime stream for asynchronous execution (nullptr by default)

    bool isInited_ = false;  // Member variable: Flag indicating if resource initialization is completed (false by default)
    ModelProcess modelProcess_;  // Member variable: Instance of ModelProcess class (used for model management, not local variable)
    bool isModelLoaded_ = false;  // Member variable: Flag indicating if model loading is completed (false by default)
};

/**
* Function: Friend function to set input file path for SampleProcess instance
* Description: Directly access the input_path_ member variable of SampleProcess (no need to add new Set method)
* Input: sample - Reference to the target SampleProcess instance
* Input: path - Full file path to be set for input data
*/
inline void set_input_path(SampleProcess& sample, const std::string& path) {
    sample.input_path_ = path;  // Assign input file path to the member variable of the SampleProcess instance
}

#endif // SAMPLE_PROCESS_H
