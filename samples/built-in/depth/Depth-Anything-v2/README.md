# 基于Depth-Anything-v2网络实现图片分类
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

Depth Anything V2在细节和鲁棒性方面显著优于 [V1](https://github.com/LiheYoung/Depth-Anything)。与基于 SD 的模型相比，它具有更快的推理速度、更少的参数和更高的深度精度。本示例使用的是Depth-Anything-V2-Small
- 参考实现：

  ```
  https://github.com/DepthAnything/Depth-Anything-V2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 518 x 518 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | class    | FP32     | 1x518x518 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── inc
│   ├── ...            //声明头文件

├── script
│   ├── pth2onnx.py     //python执行脚本

├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

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

2、该模型需要以下环境

  **表 1** 版本配套表

     | 芯片型号   | npu     | soc_version | 环境准备指导                                                 | cann包版本                                                   | 编译工具链                                                   | os                                                           | sdk                                                          |
    | ---------- | ------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz) | [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100环境搭建指南/Hi3403V100环境搭建指南.md#241安装clang交叉编译器) | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100环境搭建指南/Hi3403V100环境搭建指南.md) | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)                                                      | aarch64-mix210-linux-gcc                                     | linux                                                        | SS928 V100R001C02SPC022                                                       |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 安装依赖(注意：python使用3.9.11版本)。

   ```
   pip install -r requirements.txt
   ```
3. 源代码下载。
   ```
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git
   git reset --hard e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1
   cd Depth-Anything-V2
   cp ../depth.patch ./
   git apply depth.patch
   cd ..
   ```


## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[ImageNet](https://huggingface.co/datasets/depth-anything/DA-2K/tree/main)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集data路径下。数据集目录结构如下所示：

   ```
   DA-2K/
   |-- DA-2K
   |   |-- assets
   |   |-- images
   |   |   |-- adverse_style
   |   |   |   |-- 196078476_9609c2c45d_k.jpg
   |   |   |   |-- ...
   |   |   |-- ...
   |   |-- annotations.json
   |   ...
   ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
  
    将preprocess.py复制到下载好的源码Depth-Anything-V2中，执行下面命令完成完成数据预处理。
    
    ```
    cd Depth-Anything-V2
    python ./preprocess.py
    ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   前往[Pytorch官方文档](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true)下载对应权重。

   将下载好的模型文件放在Depth-Anything-V2/checkpoints目录中

2. 导出onnx文件。

    将pth2onnx.py复制到下载好的源码Depth-Anything-V2中，执行下面命令完成完成模型转换到onnx。

    ```
    cd Depth-Anything-V2
    python ./pth2onnx.py
    cd ..
    ```
    生成模型depth_anything_v2_vits.onnx。在model目录下

3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. Hi3403V100 SVP_NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/depth_anything_v2_vits.onnx" --input_shape="input:1,3,518,518" --output="./model/depthanything" --image_list="./data/img/123949917_fd08c80d60_b.bin" --compile_mode=1 --softmax_optimize_enable=1 --fusion_switch_file=TransformerFusion:on --soc_version=SS928V100 
        ```
   
        运行成功后生成depthanything.om模型文件。

        参数说明：
      
        - --framework：5代表ONNX模型。
        - --model：为ONNX模型文件。
        - --input_shape：输入数据的shape。
        - --insert_op_conf：aipp算子配置，用于输入数据处理。
        - --output：输出的OM模型。
        - --image_list: 量化校准数据。
        - --enable_small_channel:使能small channel优化。
        - --enable_single_stream:推理时使用一条stream。
        - --soc_version：处理器型号。

## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1.  切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```
    mkdir -p build
    ```

2.  切换到“build“目录，执行**cmake**生成编译文件。

    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    例如，开发环境为X86架构、运行环境为ARM架构时，执行以下命令进行交叉编译。交叉编译工具链按运行环境操作系统，可选toolchain_aarch64_linux.cmake或toolchain_aarch64_ohos.cmake；SOC_VERSION按算力引擎可选SS928V100或OPTG，请根据运行环境和算力引擎平台选择。
	  
    ```bash
    cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${toolchain.cmake} -DSOC_VERSION=${soc_version}
    ```
    比如
    ```bash
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake -DSOC_VERSION=SS928V100
    ```
    
3.  执行**make**命令，生成的可执行文件main在“./out“目录下。

    ```
    make
    ```

**步骤2：运行应用。**

1.  以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2.  以运行用户登录运行环境（Host）。
3.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。
    需要先将opensource/opencv/lib/中的so加入到LD_LIBRARY_PATH。比如：
    LD_LIBRARY_PATH=XXX/samples/opensource/opencv/lib/:$LD_LIBRARY_PATH

    ```
    ./main --acl ../src/acl.json --model ../model/depthanything.om --input ../data/file_list.txt
    ```

**步骤3：输出后处理**

1. 精度验证。

    调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据。

    ```
    python ./script/postprocess.py
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/bin/

    例如：  `python ./script/postprocess.py --output ./out/result/bin/`
      
    SVP_NNN平台上精度结果：
    ```
    accuary:  0.933752417794971
    ```

2. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/depthanything.om --input ../data/file_list_1.txt --loop 100
    ```

    参数说明：(此模式下，file_list_1.txt只放一张图片)

    - --model：om模型路径。
    - --output:  后处理后结果所在位置
    - --model: 模型所在位置
    - --loop：循环执行多少次取结果， loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

    在板端会输出显示，SVP_NNN平台上性能结果如下：
    ```
    [INFO]  time: 26623028, fps: 3.756147
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Depth-Anything-v2模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标 |性能(fps) |
| ----------- | ---------- | -------- | ------------------ |----------- |
| Hi3403V100 SVP_NNN | 1          | DA-2K  |  0.93           |3.756    |