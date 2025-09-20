#include "svp_nnn/svp_nnn.h"
#include "dev_interface_adapter.h"
#include <iostream>
#include <thread>
#include <chrono>
#include "log.h"

constexpr int MAX_THREAD_NUM = 4;

int testThread(std::shared_ptr<Infer::SvpAclMdl> model, Infer::RunMode runMode, int index)
{
    int ret;
    void *ptr = nullptr;
    std::vector<Infer::TensorBuf> inBuf, outBuf;
    Infer::TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t  outputNum = model->GetInTensorNum();

    for (size_t i = 0; i < inputNum; i++) {
        ret =  model->GetInTensorDescByIdx(i, desc);
        inBuf.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    for (size_t i = 0; i < outputNum; i++) {
        ret =  model->GetOutTensorDescByIdx(i, desc);
        outBuf.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    ret = model->Execute(inBuf, outBuf, runMode);
    if (ret != SUCCESS) {
        ERROR_LOG("execute model failed");
        return -1;
    }
    
    if (runMode == Infer::RunMode::Async) {
        ret = model->Wait();
        if (ret != SUCCESS) {
            ERROR_LOG("wait failed");
            return -1;
        }
        INFO_LOG("thread %d async execute success", index);
    } else {
        INFO_LOG("thread %d sync execute success", index);
    }

    return 0;
}

int main()
{
    int ret;
    std::thread syncT[MAX_THREAD_NUM];
    std::thread asyncT[MAX_THREAD_NUM];

    ret = Infer::DevInit("");
    if (ret != SUCCESS) {
        ERROR_LOG("dev init failed");
        return -1;
    }
       
    std::shared_ptr<Infer::SvpAclMdl> model = std::make_shared<Infer::SvpAclMdl>();
    ret = model->LoadModel("/mnt/workspace/modelzoo/samples/built-in/classification/mobileNetV2/model/mobileNetV2_original.om");
    if (ret != SUCCESS) {
        ERROR_LOG("load model failed");
        return -1;
    }
    

    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        syncT[i] = std::thread(testThread, model, Infer::RunMode::Sync, i);
        asyncT[i] = std::thread(testThread, model, Infer::RunMode::Async, i);
    }

    for (int i = 0; i < MAX_THREAD_NUM; i++) {
        syncT[i].join();
        asyncT[i].join();
    }

    (void)model->UnLoadModel();
    (void)Infer::DevDeInit();

    return 0;
}