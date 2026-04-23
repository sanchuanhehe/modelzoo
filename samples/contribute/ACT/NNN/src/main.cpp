/**
* @file main.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <fstream>
#include <iostream>
#include "sample_process.h"
#include "utils.h"
#include <vector>
#include <string>
using namespace std;

int main() {
    // 初始化推理环境（只执行一次）
    SampleProcess sample;
    if (sample.InitResource() != SUCCESS) {
        cerr << "Init resource failed" << endl;
        return -1;
    }

    // 加载模型（只执行一次）
    if (sample.LoadModel() != SUCCESS) {
        cerr << "Load model failed" << endl;
        sample.DestroyResource();
        return -1;
    }

    // 循环处理多次输入
    while (true) {
        vector<const void*> input_datas;
        vector<size_t> input_sizes;
        const int INPUT_COUNT = 3;

        // 读取输入数据（保持原有逻辑）
        bool readSuccess = true;
        for (int i = 0; i < INPUT_COUNT; ++i) {
            uint32_t data_size;
            cin.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            if (!cin.good()) {
                cerr << "Read input " << i << " size failed" << endl;
                readSuccess = false;
                break;
            }

            void* data = nullptr;
            aclError ret = aclrtMalloc(&data, data_size, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_SUCCESS || data == nullptr) {
                cerr << "Malloc buffer for input " << i << " failed" << endl;
                readSuccess = false;
                break;
            }

            cin.read(reinterpret_cast<char*>(data), data_size);
            if (!cin.good()) {
                cerr << "Read input " << i << " data failed" << endl;
                aclrtFree(data);
                readSuccess = false;
                break;
            }

            input_datas.push_back(data);
            input_sizes.push_back(data_size);
        }

        // 检查是否读取失败（比如到达输入末尾）
        if (!readSuccess) {
            // 释放已分配的内存
            for (auto ptr : input_datas) aclrtFree(const_cast<void*>(ptr));
            break;
        }

        // 设置输入并执行推理
        sample.SetInputDatas(input_datas, input_sizes);
        if (sample.Process() != SUCCESS) {
            cerr << "Inference failed" << endl;
        } else {
            cout << "3-input inference success" << endl;  // 注意这里修正了原代码的数字错误（5->3）
        }

        // 释放当前批次的输入内存
        for (auto data : input_datas) aclrtFree(const_cast<void*>(data));
    }

    // 最后释放所有资源
    sample.DestroyResource();
    return 0;
}
