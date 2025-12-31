# 基于**Siamese Network**网络实现人脸识别
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

Siamese Network（孪生神经网络）是一种通过共享权重的两个相同子网络来度量两个输入样本相似性的深度学习框架，广泛应用于人脸识别、签名验证等任务。
- 参考实现：

  ```
  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image_1    | BGR_FP32 | 1 x 3 x 100 x 100 | NCHW         |
  | image_2    | BGR_FP32 | 1 x 3 x 100 x 100 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | output_1 | FP32     | 1x5 |
  | output_2 | FP32     | 1x5 |




## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── accuary.py     //精度评估脚本
│   ├── generate_file_list.py     //数据集目录文件生成脚本
│   ├── preprocess.py     //预处理脚本
│   ├── pth2onnx.py     //pth转onnx脚本

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

| 芯片型号  | npu     | soc_version | 环境准备指导  | cann包版本 | 编译工具链 | os  | sdk  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | ss928v100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN | ss928v100   | [推理环境准备](https://gitee.com/Hispark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | SPC 022  |  aarch64-mix210-linux-gcc |  linux  |  SPC 022  |
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  linux  |  SPC 022 |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```
3. 获取开源源码
   ```
   pip install nbconvert
   git clone https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch.git
   cd Facial-Similarity-with-Siamese-Networks-in-Pytorch
   sudo apt install jupyter-core
   pip3 install nbconvert
   jupyter nbconvert --to Siamese-networks-medium.ipynb
   cd ../
   cp Facial-Similarity-with-Siamese-Networks-in-Pytorch/Siamese-networks-medium.py script
   cd script/
   git apply ../siamese.patch
   mv Siamese-networks-medium.py model.py
   cd ../
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   ```
   mkdir datasets
   cp Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces datasets -R
   ```

2. 处理图片和生成量化校准数据
  
    ```
    cd data
    mkdir preprocess
    cd preprocess
    mkdir bin
    cd ../../script
    python3 preprocess.py 
    python3 script/generate_file_list.py
    cd ../
    ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   ```
   mkdir model
   cd script
   python3 Siamese-networks-medium.py
   cd ../

2. 导出onnx文件。
    ```
    cd script
    python3 ./pth2onnx.py
    cd ..
    ```
         
3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. SS928V100 SVP_NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/siamese_network.onnx" --input_shape="img1:1,1,100,100;img2:1,1,100,100" --image_list="./data/preprocess/bin/0_0_s6_3.bin;./data/preprocess/bin/0_0_s6_4.bin" --output="./model/siamese_network" --compile_mode=5 --soc_version=SS928V100
        ```
    2. SS928V100 NNN上的om模型转换命令
      ```
      atc --framework=5 --model="./model/siamese_network.onnx" --input_shape="img1:1,1,100,100;img2:1,1,100,100" --output="./model/siamese_network" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG
      ```
        运行成功后生成siamese_network.om模型文件。

        参数说明：      
        - --framework：5代表ONNX模型。
        - --model：为ONNX模型文件。
        - --input_shape：输入数据的shape。
        - --insert_op_conf：aipp算子配置，用于输入数据处理。
        - --output：输出的OM模型。
        - --image_list: 量化校准数据。
        - --enable_small_channel:使能small channel优化。
        - --enable_single_stream:推理时使用一条stream。
        - --compile_mode：量化时使用a16w8。
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
    ./main --acl ../src/acl.json --model ../model/siamese_network.om --input ../data/file_list.json
    ```
    

**步骤3：输出后处理**

1. 精度验证。
    ```
    python3 script/accuary.py
    ```

    SVP_NNN平台上精度结果：
    
    ```
    acc: 94.483%
    ```

    NNN平台上精度结果：
    
    ```
    acc: 95.632%
    ```
2. 验证batch_size的om模型的性能，参考命令如下：
    ```
    执行./main --acl ../src/acl.json --model ../model/siamese_network.om --input ../data/file_list_1.json
    ```
    
    参数说明：(此模式下，file_list_1.txt只放一张图片)

    - --model：om模型路径。
    - --output:  后处理后结果所在位置
    - --model: 模型所在位置

    file_list_1.json中loop设为1000

    在板端会输出显示，SVP_NNN平台上性能结果如下：
    ```
    execution time: 4.60ms, frame rate: 217.44fps
    ```
    NNN平台上性能结果如下：
    ```
    execution time: 28.81ms, frame rate: 34.71fps
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，SiameseNetwork模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标 | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ |  ------------------ |
| SS928V100 SVP_NNN | 1          | Faces | 94.483%       | 217.44  |
| SS928V100 NNN | 1          | Faces | 95.632%       | 34.71   |
