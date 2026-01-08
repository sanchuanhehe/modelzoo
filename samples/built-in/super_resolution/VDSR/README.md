# 基于VDSR实现图片超分
- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)
    - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [快速上手](#ZH-CN_TOPIC_0000001126281700)

  - [获取源码](#section4622531142816)
  - [准备数据集](#section183221994411)
  - [模型转化](#section741711594517)
  - [模型推理](#section741711594518)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

VDSR（Very Deep Super-Resolution Network）是一种20层深度卷积神经网络，通过残差学习实现图像超分辨率重建。

- 参考实现：

  ```
  https://github.com/Lornatang/VDSR-PyTorch
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | input  | FP32 | 1 x 1 x 516 x 516 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | output  | FP32     | 1 x 1 x 516 x 516 | NCHW |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess        //量化文件
│   ├── ├──bin            //量化文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── acc.py           //精度评估脚本
│   ├── generate_quant_bin.py     //atc量化bin文件生成脚本
│   ├── pth2onnx.py     //pt转onnx脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── ...	      //模型文件

├── CMakeLists.txt    //编译脚本，调用src目录下CMakeLists文件
├── *.json			//模型信息
├── LICENSE			//模型LICENSE
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

1. 执行命令查看芯片名称。

   ```
   cat /proc/umap/sys
   #该设备芯片名为SS928V100 （自行替换）
   回显如下：
   [SYS] Version: [SS928V100XXXXXXXXX]
   ```

2. 该模型需要以下环境

**表 1** 版本配套表

| 芯片型号  | npu     | soc_version | 环境准备指导  | cann包版本 | 编译工具链 | os  | sdk  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  linux  | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  linux  | SS928 V100R001C02SPC022                                                       |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取参考代码仓源码
 
   ```bash
   git clone https://github.com/Lornatang/VDSR-PyTorch.git
   cd VDSR-PyTorch
   git checkout a573e5c1e3d6d0a072cf3602022ab4954743a273
   cd ../
   mkdir model
   mkdir datasets
   ```


2. 安装依赖。

   ```bash
   # 建议使用Python 3.7.5
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   点击 [链接](https://github.com/Lornatang/VDSR-PyTorch#download-datasets) 下载数据集Set5进行精度评估，放到`VDSR/datasets`目录下，文件结构如下：
   
   ```
   Set5
      ├── original（5张图片）
      ├── X2（10张图片）
         ├── GT（高质量人脸，作为真值数据）
            ├── baby.png
            ├── bird.png
             ……
         ├── LR（低质量人脸，作为模型输入数据）
            ├── baby.png
            ├── bird.png
             ……
   ...
   ```

2. 准备量化bin文件
   ```
   cp VDSR-PyTorch/imgproc.py script/imgproc.py
   cp VDSR-PyTorch/model.py script/model.py
   cd data
   mkdir preprocess
   cd preprocess
   mkdir bin
   cd ../../
   cd script
   python3 generate_quant_bin.py
   cd ../
   ```



## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   点击 [链接](https://github.com/Lornatang/VDSR-PyTorch?tab=readme-ov-file#download-weights) 下载 SuperResolution/VDSR下的vdsr-TB291-fef487db.pth.tar 。
   
   下载成功后，将vdsr-TB291-fef487db.pth.tar文件放到./model路径下

2. pth转onnx。

      先使用script/pth2onnx.py精简不需要的输入和输出参数；

      然后使用onnxsim命令去除冗余节点和计算，提升模型推理效率。

      ```bash
      cd script
      python3 pth2onnx.py
      cd ../
      ```

3. 使用ATC工具将ONNX模型转OM模型。

      在当前模型的代码根目录下，执行ATC命令。
      1. SS928V100 SVP_NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/vdsr.onnx" --input_shape="input:1,1,516,516" --output="model/vdsr" --image_list="data/preprocess/bin/butterfly_2.bin" --soc_version=SS928V100
         ```
      2. SS928V100 NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/vdsr.onnx" --input_shape="input:1,1,516,516" --output="model/vdsr" --enable_single_stream=true --soc_version=OPTG  
         ```
         
         运行成功后生成vdsr.om模型文件。
    
         参数说明：
    
         - --framework：原始框架类型，5代表ONNX模型。
         - --model：ONNX模型文件路径。
         - --input_shape：输入数据的shape。
         - --output：输出的OM模型路径。
         - --image_list：转换模型生成量化参数时用的校准数据。(如果您没有现成的bin文件，您需要参考后面的python脚本使用说明，使用preprocess.py脚本先生成bin文件。如果您想使用多个bin文件进行数据校准，多个文件间请使用;分割，例如a.bin;b.bin。)
         - --soc_version：处理器型号。
         - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：SVP_NNN上选取其他编译模式可能导致精度下降
         - --enable_single_stream: 推理时使用一条stream。
      
         注意：如果出现atc命令找不到，参考推理环境准备。


## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```bash
    mkdir -p build
    ```

2.  切换到“build“目录，执行**cmake**生成编译文件。
    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    例如，当开发环境为X86架构，运行环境为ARM架构时，执行以下命令进行交叉编译。其中交叉编译工具链有toolchain_aarch64_linux.cmake和toolchain_aarch64_ohos.cmake两个选项，SOC_VERSION根据使用npu的不同有SS928V100和OPTG两个选项，请根据开发和运行环境选择使用。
	  
	  ```
    cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${toolchain.cmake} -DSOC_VERSION=${soc_version}
	  ```
    比如
    ```
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake -DSOC_VERSION=SS928V100
    ```


3.  执行**make**命令，生成的可执行文件main在“./out“目录下。

	```bash
	make
	```

**步骤2：运行应用。**

1. 以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2. 以运行用户登录运行环境（Host）。
3. 切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，给该目录下的main文件加执行权限。

    ```bash
    chmod +x main

4. 切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/vdsr.om  --input ../data/file_list.json
    ```
    修复后的图像默认会保存在data/result/img目录下，模型推理的原始bin文件默认会保存在data/result/bin目录下, 精度结果放到data/result/txt下。

**步骤3：验证精度和性能**

1. 精度验证。

   使用acc.py将模型推理的结果与数据集中原始的高清人脸图像进行对比，获取评估结果。

   ```bash
   python3 script/acc.py
   ```

   SVP_NNN平台上精度结果：
   ```bash
   X2 的PSNR平均值为：37.1062 dB
   X3 的PSNR平均值为：33.2009 dB
   X4 的PSNR平均值为：30.8195 dB
   ```

   NNN平台上精度结果：
   ```bash
   X2 的PSNR平均值为：37.1911 dB
   X3 的PSNR平均值为：33.2236 dB
   X4 的PSNR平均值为：30.8863 dB
   ```


2. 验证batch_size的om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/vdsr.om  --input ../data/file_list_1.json
   ```
  

   参数说明：(此模式下，输入路径为一张图片)
   - --acl:  ACL 配置文件路径
   - --model: 指定待验证性能的 OM 模型文件路径。
   - --input： 指定输入数据的列表文件路径（此场景下为单图路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可, loop设为1000）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 66.28ms, frame rate: 15.09fps
   ```

   NNN平台上性能结果如下：
   ```bash
   execution time: 228.43ms, frame rate: 4.38fps
   ```
   

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 平均PSNR| 性能(fps) |
| ----------- | ---------- | -------- | ------------------------------------- |---|
| SS928V100 SVP_NNN | 1          | Set5  |X2：37.1062 dB，X3：33.2009 dB，X4：30.8195 dB | 15.09 |
| SS928V100 NNN | 1          | Set5  |X2：37.1911 dB，X3：33.2236 dB，X4：30.8863 dB | 4.38 |

