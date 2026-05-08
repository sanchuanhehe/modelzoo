# SuperPoint模型-推理指导
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

我们设计了一种称为SuperPoint的全卷积神经网络架构，该架构对全尺寸图像进行操作，并在单次前向传递中产生伴随固定长度描述符的兴趣点检测。该模型有一个单一的共享编码器来处理和减少输入图像的维数。在编码器之后，该架构分成两个解码器“头”，它们学习任务特定权重——一个用于兴趣点检测，另一个用于感兴趣点描述。大多数网络参数在两个任务之间共享，这与传统系统不同，传统系统首先检测兴趣点，然后计算描述符，并且缺乏跨两个任务共享计算和表示的能力。



- 参考实现：

  ```
  url= https://github.com/eric-yyjau/pytorch-superpoint
  commit_id= 5eb75d74df27c07f6e7311df8f167e2a9c01a798
  model_name= SuperPointNet_gauss2
  ```


## 输入输出数据<a name="section4622531142820"></a>


- 输入数据

  | 输入数据 | 数据类型 | 大小                        | 数据排布格式 |
  | -------- |---------------------------| ------------------------- | ------------ |
  | input    | RGB_FP32 | batchsize x 1 x 240 x 320 | NCHW         |


- 输出数据

  | 输出数据    | 数据类型                | 大小 | 数据排布格式 |
  | -------- |---------------------------|--------| ------------ |
  | output1 | FLOAT32 | 1 x 256 x 30 x 40  | NHCW   |
  | output2 | FLOAT32  | 1 x 65 x 30 x 40  | NHCW   |


## 目录结构<a name="section540883920407"></a>
```
.
├── data
│   ├── ...                  // 测试数据

├── model
│   ├── ...                  // 模型文件

├── model_cfg                // 模型配置文件
│   ├── SS928V100_NNN        //     dlite配置
│   └── SS928V100_SVP_NNN    //     dpico配置

├── script                   
│   ├── superpoint_postprocess.py //  后处理脚本，得出精度结果
│   ├── superpoint_preprocess.py  // 预处理脚本，得到模型的输入文件
│   └── superpoint_pth2onnx.py    // pth 转onnx 脚本

├── src
│   ├── acl.json          // 系统初始化的配置文件
│   ├── CMakeLists.txt    // 编译脚本
│   ├── main.cpp          // 资源初始化/销毁相关函数的实现文件

├── README.md             // readme说明 
├── CMakeLists.txt        // 编译脚本，调用src目录下CMakeLists文件
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
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j3n1o7csso00)上进行下载。

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

4. 切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出输入图片的top5置信度的类别标识。测试图片上模型推理命令参考：
    
    ```
    ./main --acl ../src/acl.json --model ../model/superpoint_bs1.om --input ../data/file_list_1.json 
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>
1. 获取源码。
   ```
   git clone https://github.com/eric-yyjau/pytorch-superpoint.git
   cd pytorch-superpoint
   git reset --hard 5eb75d74df27c07f6e7311df8f167e2a9c01a798
   patch -p3 < ../sp.patch
   ```

2. 安装依赖  
   ```
   #建议python版本3.7.5
   conda create -n myenvpoint python=3.7.5 -y
   pip install -r requirements.txt 
   ```

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集

   数据集名称: [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)，解压命令参考:
   ```
   tar –xvf  *.tar
   tar –zxvf  *.tar.gz
   unzip *.zip
   ```
   数据集存放在当前路径下的datasets文件夹(如果没有，请新建)中

   ```
   datasets/
   |-- hpatches
   |   |-- i_ajuntament
   |   |-- ...
   |-- hpatches_preprocessed
   |   |-- i_ajuntament_1.bin
   |   |-- ...
   ```


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

    [下载链接](https://github.com/eric-yyjau/pytorch-superpoint/tree/master/logs/superpoint_coco_heat2_0/checkpoints) 采用名称为superPointNet_170000_checkpoint.pth.tar的权重文件

2. 导出onnx文件。

   在代码主目录使用superpoint_pth2onnx.py导出onnx文件。
   
   ```
   python ./script/superpoint_pth2onnx.py --model_path=$model_path --batch_size=$batch_size
   ```
   
   | 参数           | 说明                      |                                
   |---------------|--------------------------------------|                               
   | model_path    | pth模型路径(superPointNet_170000_checkpoint.pth.tar) |
   | batch_size    | batch大小(1、4、8、16、32、64)                       |

    例如执行如下命令后，获得superpoint_bs1.onnx文件
    ```
    python ./script/superpoint_pth2onnx.py --model_path=./model/superPointNet_170000_checkpoint.pth.tar --batch_size=1
    ```
   
3. 使用ATC工具将ONNX模型转OM模型。

    1). Hi3403V100 SVP_NNN上的om模型转换命令:
    ```
    atc --framework=5 \
    --model=./model/superpoint_bs1.onnx  \
    --input_shape="image:1,1,240,320"  \
    --output=./model/superpoint_bs1  \
    --compile_mode=0   \
    --input_type=image:FP32  \
    --image_list="./data/hpatches_preprocessed/i_ajuntament_1.bin"    \
    --soc_version=SS928V100
    ```
    2). Hi3403V100 NNN上的om模型转换命令

        atc --framework=5 --model="./model/superpoint_bs1.onnx" --input_format="NCHW" --input_shape="image:1,1,240,320" --output="./model/superpoint_bs1_dlite" --enable_single_stream=true --soc_version=OPTG
    运行成功后生成superpoint_bs1.om模型文件。
    
    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_shape：输入数据的shape。
    - --insert_op_conf：aipp算子配置，用于输入数据处理。
    - --output：输出的OM模型。
    - --input_type: 输入的数据的类型
    - --image_list: 量化校准数据。
    - --compile_mode:编译模式，参数值1代表使用16bit量化数据，使用8bit量化权重。
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。
    - --soc_version：处理器型号。

## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。

    ```
    ./main --acl ../src/acl.json --model ../model/superpoint_bs1.om --input ../data/file_list.json 
    ```

2. 精度验证。

    在代码主目录进行精度计算

    ```
    python ./script/superpoint_postprocess.py --img_path=$config_path --result_path=$result_path
    ```
    | 参数        | 说明                                         |
    | -           | -                                            |
    | img_path    | config路径(./configs/magicpoint_repeatability_heatmap.yaml) |
    | result_path | 推理文件保存的位置                            |

    例如执行: `python ./script/superpoint_postprocess.py --img_path=./pytorch-superpoint/configs/magicpoint_repeatability_heatmap.yaml --result_path=./out/result/bin`
    
    备注：./pytorch-superpoint/configs/magicpoint_repeatability_heatmap.yaml 中的pretrained: 'logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar'地址需要修改为：./pytorch-superpoint/logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar

    Hi3403V100 SVP_NNN平台上精度结果如下：
    ```
    mean AP 0.8017776076290524
    end
    ```
    Hi3403V100 NNN平台上精度结果如下：
    ```
    mean AP 0.8009278098804761
    ```

2. 性能验证。

   验证om模型的性能，在板端执行参考命令如下：
   ```
   ./main --acl ../src/acl.json --model ../model/superpoint_bs1.om --input ../data/file_list_1.json 
   ```

   参数说明：(此模式下，输入路径文本文件file_list_1.json内容为一张图片路径,请将该文件中的loop改为100)

   - --model：om模型路径。
   - --acl: acl.json文件的路径，默认放在src目录下。
   - --input: 输入的图像数据列表路径
   - --loop： 循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值。NNN平台修改json里面的loop值
   
   在板端会输出显示，Hi3403V100 SVP_NNN平台上性能结果如下：
   ```
   [INFO] time: 311917, fps: 320.598108
   ```
   Hi3403V100 NNN平台上性能结果如下：
   ```
   [INFO] time: 15.21ms, frame rate: 65.76fps
   ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，SuperPoint模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   |  精度          |    性能(FPS)|
| ----------- | --------- | -------- |  ------------ | ------------|
| Hi3403V100 SVP_NNN | 1   | hpatches |  80.17%       | 320.60      |
| Hi3403V100 NNN | 1   | hpatches |  80.09%       | 65.76      |
