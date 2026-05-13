



# HRNet模型-推理指导

- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)
    - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型推理](#ZH-CN_TOPIC_0000001126281700)

  - [快速开始(推荐)](#section4622531142816)
  - [安装依赖](#section183221994410)
  - [准备数据集](#section183221994411)
  - [模型转化](#section741711594517)
  - [精度&性能评估](#section741711594518)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

HigherHRNet 是一种新型的自下而上人体姿态估计算法，它在训练阶段引入多分辨率监督机制，在推理阶段采用多分辨率聚合策略，不仅能有效应对自下而上多人姿态估计任务中的尺度变化难题，还可实现关键点的高精度定位，尤其在小尺寸人体目标的处理上表现突出。

- 参考实现：

  ```
  url=https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
  branch=master
  commit_id=aa23881492ff511185acf756a2e14725cc4ab4d7
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 512 x 768 | NCHW         |

- 输出数据

  | 输出数据 |  数据类型   |       大小         |
  | -------- | --------  | ------------------ |
  | class    | FP32      | 1 x 34 x 128 x 192 |
  | 2990    | FP32       | 1 x 17 x 256 x 384 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── pth2onnx.py     //pth转onnx脚本
│   ├── accurary.py     //精度验证脚本
│   ├── gen_filelist_and_quant_file.py  //生成输入文件列表和模型量化校准数据

├── src
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── *.json			//模型信息
├── LICENSE			//模型LICENSE
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
    | Hi3403V100 | SVP_NNN | SS928V100  | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |
    | Hi3403V100 | NNN     | OPTG      | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9qdl2hh6k00)上进行下载。

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

4. 切换到可执行文件main所在的目录，运行可执行文件。测试图片上模型推理命令参考：
    
    ```
    ./main --model ../model/hrnet_512_768.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 python 3.7.5
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

    该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`modelzoo/datasets`源码目录下创建`coco2017`文件夹，文件结构如下：

    ```
    coco2017
        ├── val2017
          ├── 00000000139.jpg
          ├── 00000000285.jpg
          ……
          └── 00000581781.jpg
        ├── annotations
          ├── person_keypoints_val2017.json
    ...
    ```

2. 生成输入文件列表和模型量化校准数据

    ```
    python3 ./script/gen_filelist_and_quant_file.py --input_path ${input_path} --output_path ${output_path}
    ```

    参数说明:

    - --input_path：数据集路径，需要使用相对路径
    - --output_path：生成的file_list.json文件和quant.bin文件路径

    例如：
    ```
    python3 ./script/gen_filelist_and_quant_file.py --input_path ../../../../datasets/coco2017/val2017 --output_path ./data
    ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。


1. 获取权重文件。

    [pose_higher_hrnet_w32_512.pth](https://drive.google.com/drive/folders/1zJbBbIHVQmHJp89t5CD1VF5TIzldpHXn)
      ```bash
      # 创建model目录，并将下载的pose_higher_hrnet_w32_512.pth文件放到model目录下
      mkdir -p model
      ```

2. 导出onnx文件。

    ```
    git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git
    cd HigherHRNet-Human-Pose-Estimation
    git reset --hard aa23881492ff511185acf756a2e14725cc4ab4d7
    patch -p1 < ../HigherHRNet.patch
    cd ../
    python3 script/pth2onnx.py --cfg ./HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml --input model/pose_higher_hrnet_w32_512.pth --hw 512 768 --output ./model/hrnet_512_768.onnx
    ```

3. 使用ATC工具将ONNX模型转OM模型。
    
    Hi3403V100 SVP_NNN上的om模型转换命令:
    ```
    atc --framework=5 --model="./model/hrnet_512_768.onnx" --input_shape="image:1,3,512,768" --output="model/hrnet_512_768" --image_list="./data/quant.bin" --compile_mode=1 --soc_version=SS928V100
    ```
    Hi3403V100 NNN上的om模型转换命令:
    ```
    atc --framework=5 --model="./model/hrnet_512_768.onnx" --input_shape="image:1,3,512,768" --output="model/hrnet_512_768" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG 
    ```
    运行成功后生成hrnet_512_768.om模型文件。

    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_shape：输入数据的shape。
    - --output：输出的OM模型。
    - --image_list: 量化校准数据。数据获取参考[准备数据集](#section183221994411)章节。
    - --compile_mode:编译模式，参数值1代表使用16bit量化数据，使用8bit量化权重。
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。
    - --soc_version：处理器型号。


## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。
    ```
    ./main --model ../model/hrnet_512_768.om --input ../data/file_list.json
    ```
  结果会保存在main程序所在目录下的result目录下。


2. 精度验证。

    执行脚本将结果文件与数据集标签比对，可以获得精度数据，结果保存在accuracy.txt中。

    ```
    python3 script/accuracy.py --cfg ./HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml --input data/file_list.json --dir ./out/result/bin/
    ```

    SVP_NNN平台上精度结果：
      ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.646
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.869
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.708
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.600
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.736
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.697
        Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.886
        Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.742
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.637
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.802
      ```
    NNN平台上精度结果：

     ```
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.658
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.871
      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.724
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.609
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.753
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.705
      Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.889
      Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.751
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.647
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.807
     ```

3. 验证om模型的性能，参考命令如下：

    ```
    ./main --model ../model/hrnet_512_768.om --input ../data/file_list_1.json
    ```

    参数说明：
    - --model：om模型文件路径。
    - --input: 输入的图像列表文件

    file_list_1.json中的配置代表对一张输入图片重复推理100次，程序执行时会在板端会输出打印推理的平均时间和帧率。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，HRNet模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   |  mAP（IoU=0.50:0.95） | mAP（IoU=0.50） |      性能(FPS)|
| ----------- | ---------- | -------- |  ------------------ | ------------------ |-----------------|
| Hi3403V100 SVP_NNN | 1          | coco2017 |  64.6%            | 86.9%            | 13.84   |
| Hi3403V100 NNN     | 1          | coco2017 |  65.8%            | 87.1%            | 5.52    |