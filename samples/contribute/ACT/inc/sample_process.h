/**
* @file sample_process.h
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef SAMPLE_PROCESS_H
#define SAMPLE_PROCESS_H

#include <vector> 
#include "utils.h"
#include "acl/svp_acl.h"
#include "model_process.h"

/**
* SampleProcess
*/
class SampleProcess {
public:
    /**
    * @brief Constructor
    */
    SampleProcess();

    /**
    * @brief Destructor
    */
    ~SampleProcess();

    /**
    * @brief init reousce
    * @return result
    */
    Result InitResource();

    /**
    * @brief sample process
    * @return result
    */
    Result Process();
    // void DestroyResource();

    void SetInputPath(const std::string& path);
    std::string input_path_;  // 用于存储输入文件路径

    const char* input_data_ = nullptr;  // 二进制数据指针
    size_t input_data_size_ = 0;        // 数据大小

    void SetInputDatas(const std::vector<const void*>& input_datas, 
                       const std::vector<size_t>& input_sizes);
    std::vector<const void*> input_datas_;
    std::vector<size_t> input_sizes_;

    // 新增：声明 LoadModel 方法
    Result LoadModel();
    void DestroyResource();

private:
    // 新增：存储从Python传递的当前文件路径
    int32_t deviceId_ { 0 };
    svp_acl_rt_context context_ { nullptr };
    svp_acl_rt_stream stream_ { nullptr };

    bool isInited_ = false;
    ModelProcess modelProcess_;  // 改为成员变量，而非局部变量
    bool isModelLoaded_ = false;
};

// 新增友元函数（仅头文件，不修改cpp），用于传递路径（不新增类方法）
inline void set_input_path(SampleProcess& sample, const std::string& path) {
    sample.input_path_ = path;  // 直接访问私有input_path_，无需新增Set方法
}

#endif // SAMPLE_PROCESS_H
