# TinySam模型-推理指导


- [概述](#ZH-CN_TOPIC_0000001172161501)

  - [输入输出数据](#section4622531142820)
  - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型推理](#ZH-CN_TOPIC_0000001126281700)

  - [快速开始(推荐)](#section4622531142816)
  - [安装依赖](#section183221994410)
  - [准备数据集](#section183221994411)
  - [模型转化](#section741711594517)
  - [精度&性能评估](#section741711594518)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)


    ******


# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

TinySAM通过全阶段知识蒸馏、在线硬提示采样、量化等系列优化策略，构建轻量级 “万物分割” 模型，解决了原始 SAM 模型计算量大、部署困难的痛点，助力高效分割任务在资源受限场景下的应用。

- 参考实现：

  ```
  url= https://github.com/xinghaochen/TinySAM
  commit_id= 1492efb28b6cd009f0940fb7b8167e48ac8a1459
  ```


## 输入输出数据<a name="section4622531142820"></a>

### image_encoder:
  | 输入数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | image    | FLOAT32 | 1 x 3 x 448 x 448    | NCHW        |

  | 输出数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | image_embeddings | FLOAT32 | 1 x 256 x 28 x 28 | NCHW   |

### prompt_encoder:
  | 输入数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | box_lt    | FLOAT32 | 1 x 2                | NCHW        |
  | box_rb    | FLOAT32 | 1 x 2                | NCHW        |

  | 输出数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | sparse_embedding | FLOAT32 | 1 x 2 x 256  | NCHW        |

### mask_decoder:
  | 输入数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | image_embeddings | FLOAT32 | 1 x 256 x 28 x 28 | NCHW   |
  | sparse_embedding | FLOAT32 | 1 x 2 x 256  | NCHW        |

  | 输出数据  | 数据类型   | 大小               | 数据排布格式 |
  | -------- |-----------| -------------------| ----------- |
  | iou_predictions | FLOAT32 | 1 x 1 x 1 x 1 | NCHW        |
  | low_res_masks | FLOAT32 | 1 x 1 x 448 x 448 | NCHW      |


## 目录结构<a name="section540883920407"></a>
```
.
├── data
│   ├── ...                  // 测试数据

├── model
│   ├── ...                  // 模型文件

├── script                   
│   ├── accuracy.py //  后处理脚本，得出精度结果

├── src
│   ├── CMakeLists.txt    // 编译脚本
│   ├── main.cpp          // 资源初始化/销毁相关函数的实现文件
│   ├── sam_predictor.h   // 模型推理类实现文件
│   ├── sam_predictor.cpp   // 模型推理类头文件

├── README.md             // readme说明 
├── *.json			      // 模型信息
├── LICENSE			      // 模型LICENSE
├── requirements.txt  
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

1. 执行命令查看芯片名称。

    ```
    cat /proc/umap/sys
    #该设备芯片名为SS928V100 （自行替换）
    回显如下：
    [SYS] Version: [SS928V100XXXXXXXXX
    ```

2. 该模型需要以下环境

    **表 1** 版本配套表

    | 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
    | --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/OpenHarmony%20Small%E7%89%88%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/OpenHarmony%20Small%E7%89%88%E6%9C%AC%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md#%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9t43v4pec00)上进行下载。

创建`model`文件夹，并将om模型文件移动到`./model`目录下。
```
mkdir -p model
```
备注：若需要体验om模型转化过程，请参考[安装依赖](#section183221994410)和[模型转化](#section741711594517)章节。

### 编译代码和运行应用

#### 编译代码

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为`build`。
    ```
    mkdir -p build
    ```

2. 切换到`build`目录，执行**cmake**生成编译文件。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    例如，开发环境为X86架构、运行环境为ARM架构时，执行以下命令进行交叉编译。交叉编译工具链按运行环境操作系统，可选toolchain_aarch64_linux.cmake或toolchain_aarch64_ohos.cmake；SOC_VERSION按算力引擎可选SS928V100或OPTG，请根据运行环境和算力引擎平台选择。
    ```
    cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${toolchain.cmake} -DSOC_VERSION=${soc_version}
    ```
    比如
    ```
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake -DSOC_VERSION=SS928V100
    ```

3. 执行**make**命令，生成的可执行文件main在“./out“目录下。


#### 运行应用

1. 将modelzoo代码上传到板端运行环境。
2. 以运行用户登录板端运行环境。
3. 切换到可执行文件main所在的目录，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4. 需要参考[准备数据集](section183221994411)下载测试图片。切换到可执行文件main所在的目录，运行可执行文件。
    
    ```
    ./main ../data/file_list_1.json ../data/coco/val2017 ../model/image_encoder_deploy_model_release.om:../model/prompt_encoder_deploy_model_release.om:../model/mask_decoder_deploy_model_release.om
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.7.5
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集

   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`TinySam/data`模型目录下创建`coco/val2017`文件夹，数据集图片存放在`val2017`文件夹下。创建eval文件夹，存放[ViTDet模型生成的检测框文件](https://github.com/xinghaochen/TinySAM/releases/download/2.0/coco_instances_results_vitdet.json)，结构如下：
   ```
    ├── data
        ├── coco
            ├── val2017
            ├── instances_val2017.json
        ├── eval
            ├── coco_instances_results_vitdet.json
   ```  


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取源码。
   ```
   cd ./script
   git clone https://github.com/xinghaochen/TinySAM
   cd ./TinySAM
   git reset --hard 1492efb28b6cd009f0940fb7b8167e48ac8a1459
   cd ../../
   ```

2. 获取权重文件。

    [下载链接](https://github.com/xinghaochen/TinySAM/releases/download/1.0/tinysam.pth) 下载tinysam.pth文件到model目录下
    ```bash
    # 创建model目录，并将下载的tinysam.pth文件放到model目录下
    mkdir -p model
    ```

3. 使用amct工具进行训练后量化生成pth和onnx文件。

    ```
    python script/calibration_box.py
    ```
    (注:需要安装amct工具)
   
4. 生成om模型文件。

    ```
    ./script/generate_om.sh
    ```
    (注:需要安装mindcmd工具)


## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。
    ```
    ./main ../data/eval/coco_instances_results_vitdet.json ../data/coco/val2017 ../model/image_encoder_deploy_model_release.om:../model/prompt_encoder_deploy_model_release.om:../model/mask_decoder_deploy_model_release.om
    ```

2. 精度验证。
    ```
    python ./script/accuracy.py --image_dir=./data/coco/val2017 --anno_path=./data/coco/instances_val2017.json --vit_det_file_path=./data/eval/coco_instances_results_vitdet.json --res_dir=./out/result
    ```
    
    SVP_NNN平台上精度结果如下：
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.313
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.570
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.302
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.352
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.539
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.269
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.399
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.478
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
    ```

3. 性能验证。

   验证om模型的性能，在板端执行参考命令如下：
   ```
   ./perf_test --model ${model_path} --loop ${loop}
   ```
   - --model：om模型文件的路径
   - --loop： 循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值
  
  该模型由三个om文件组成，通过将三个om模型的耗时求和来计算帧率。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，TinySam模型的性能和精度参考下列数据。

| 芯片型号     | Batch Size | 数据集    | mAP（IoU=0.50:0.95）| mAP（IoU=0.50）|  性能(FPS)  |
| ----------- | ---------  | -------- |  ------------ | ------------| ------------|
| Hi3403V100 SVP_NNN | 1   | coco 2017 |  31.3%       | 57.0%      |  3.99  |
