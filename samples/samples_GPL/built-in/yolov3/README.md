
# 基于**YOLOv3**网络实现目标检测
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

YOLOv3是一种端到端的one-stage目标检测模型。相比YOLOv2，YOLOv3采用了一个新的backbone-Darknet-53来进行特征提取工作，
这个新网络比Darknet-19更加强大，也比ResNet-101或者ResNet-152更加高效。
同时，对于一张输入图片，YOLOv3可以在3个不同尺度预测物体框，每个尺度预测三种大小的边界框，通过多尺度联合预测的方式有效提升了小目标的检测精度。
- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov3/tree/v9.6.0
  tag=v9.6.0
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | 0      | FP32     | 1x255x20x20 |
  | 1      | FP32     | 1x255x40x40 |
  | 2  | FP32     | 1x255x80x80 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── accuracy.py     //模型精度测试脚本
│   ├── yolov3_postprocess.py     //python后处理脚本
│   ├── yolov3_preprocess.py     //python前处理脚本

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

| 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j8pkk8lctk00)上进行下载。

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
    ./main --acl ../src/acl.json --model ../model/yolov3.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.7.5
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在项目根目录下新建`datasets`文件夹，数据集放到`datasets`里，文件结构如下：
   
   ```
    datasets
        coco
            ├── images
                ├── val2017
                    ├── 00000000139.jpg
                    ├── 00000000285.jpg
                    ……
                    └── 00000581781.jpg
            ├── instances_val2017.json
            └── val2017.txt
        ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
    执行 ../../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    
    ```
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```
    python3 ../../../../utils/generate_file_list.py ../../../../datasets/coco/images/val2017
    ```

3. 生成量化数据
    ```
    python3 ./yolov3_preprocess.py
    ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取开源源码
   ```
   git clone https://github.com/ultralytics/yolov3.git
   cd yolov3
   git checkout v9.6.0  # 切换到所用版本
   git apply ../v9.6.0.patch
   cd ../
   ```

2. 获取权重文件。

   在[链接](https://github.com/ultralytics/yolov3/tags)中找到所需版本下载并放置到model文件夹下（如果不存在，请在当前模型路径下创建），也可以使用下述命令下载。
      ```
     mkdir model
     cd model
     wget https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt
     cd ../
      ```

3. 导出onnx文件。

    1. 使用开源源码中的导出方法

         ``
         cd yolov3
         python export.py --weights=../model/yolov3.pt --opset=11
         ```
         
         获得yolov3.onnx文件。

4. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. Hi3403V100 SVP_NNN上的om模型转换命令
        ```
        python ./script/yolov3_preprocess.py
        atc --framework=5 --model="./model/yolov3.onnx" --input_shape="images:1,3,640,640" --output="./model/yolov3" --image_list="./data/prep_data_aipp/000000000139.bin" --soc_version=SS928V100
        ```
    2. Hi3403V100 NNN上的om模型转换命令

        ```
        atc --framework=5 --model="./model/yolov3.onnx" --input_format="NCHW" --input_shape="images:1,3,640,640" -input_fp16_nodes="images" -output="./model/yolov3" --enable_single_stream=true --soc_version=OPTG
        ```
        运行成功后生成yolov3.om模型文件。

        参数说明：
      
        - --framework：5代表ONNX模型。
        - --model：为ONNX模型文件。
        - --input_shape：输入数据的shape。
        - --insert_op_conf：aipp算子配置，用于输入数据处理。
        - --output：输出的OM模型。
        - --image_list: 量化校准数据。数据获取参考[准备数据集](#section183221994411)章节。
        - --enable_single_stream:推理时使用一条stream。
        - --soc_version：处理器型号。

## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出的选框会保存再./out/result/txt/目录下

    ```
    ./main --model ../model/yolov3.om --input ../data/file_list.json
    ```

2. 精度验证。

    调用脚本可以获得精度数据。

    ```
    python ./script/accuracy.py --ground_truth_json "./datasets/coco/instances_val2017.json" --output "./out/result/txt"
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/bin/

    - --ground_truth_json：真值标签文件所在路径。
      
    运行accuracy.py脚本会输出推理的精度信息：

    Hi3403V100 SVP_NNN平台上精度结果：
    ```
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.658
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.499
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.297
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.509
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.592
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.592
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.647
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.503
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.693
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
    ```
    Hi3403V100 NNN平台上精度结果：
    ```
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.660
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.503
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.301
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.362
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.596
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.652
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.509
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.790
    ```
3. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/yolov3.om --input ../data/file_list_1.json
    ```

    参数说明：(此模式下，file_list_1.json只放一张图片)

    - --acl:  ACL 配置文件路径
    - --model：om模型路径。
    - --input：指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可,循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值）

    在板端会输出显示，Hi3403V100 SVP_NNN平台上性能结果如下：
    ```
     time: 5771126, fps: 17.3276
    ```
   Hi3403V100  NNN平台上性能结果如下：
    ```
     execution time: 130.05ms, frame rate: 7.69fps
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，yolov3模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | AP（IoU=0.50） | AP（IoU=0.50:0.95） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | coco2017  | 65.8%       | 46%         | 17.3276 |
| Hi3403V100 NNN | 1          | coco2017  | 66.0%       | 46.5%         | 7.69 |
