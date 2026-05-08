# 基于**yolov5s**网络实现图片分类
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

YOLO系列网络模型是最为经典的one-stage算法，也是目前工业领域使用最多的目标检测网络，YOLOv5网络模型在继承了原有YOLO网络模型优点的基础上，具有更优的检测精度和更快的推理速度。  

YOLOv5每个版本主要有4个开源模型，分别为YOLOv5s、YOLOv5m、YOLOv5l 和 YOLOv5x，四个模型的网络结构基本一致，只是其中的模块数量与卷积核个数不一致。YOLOv5s模型最小，其它的模型都在此基础上对网络进行加深与加宽。

本示例使用YOLOv5s
- 参考实现：

  ```
  https://github.com/ultralytics/yolov5
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | 328      | FP32     | 1x255x20x20 |
  | 327      | FP32     | 1x255x40x40 |
  | output0  | FP32     | 1x255x80x80 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── inc
│   ├── ...            //声明头文件

├── script
│   ├── accuracy.py     //精度计算脚本
│   ├── generate_quant_bin.py     //量化数据生成脚本

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

2. 该模型需要以下环境

    **表 1** 版本配套表

    | 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
    | --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang) |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |
    | Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022 |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j8pr351otk00)上进行下载。

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
    ./main --acl ../src/acl.json --model ../model/yolov5s.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.8
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`yolov5`源码根目录下新建`datasets`文件夹，数据集放到`datasets`里，文件结构如下：
   
   ```
    datasets
        coco
            ├── val2017
                ├── 00000000139.jpg
                ├── 00000000285.jpg
                ……
                └── 00000581781.jpg
            ├── instances_val2017.json
        ...
   ```
2. 生成列表文件file_list.json，将原始数据集图片地址转换为模型的输入数据。
  
    执行 ../../../../utils/generate_file_list.py 脚本，在data目录下生成file_list.json文件。
    
    ```
    python3 ../../../../utils/generate_file_list.py datasets/coco/val2017
    ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取开源源码
   ```
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   git checkout v7.0  # 切换到所用版本
   cd ..
   ```

2. 获取权重文件。

   在[链接](https://github.com/ultralytics/yolov5/tags)中找到所需版本下载，也可以使用下述命令下载。
      ```
      wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
      ```

3. 导出onnx文件。
    ```
    cd yolov5
    git apply ../yolov5_7.patch
    python export.py --weights=../model/yolov5s.pt --opset=11
    ```
    
    获得yolov5s.onnx文件。

4. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. Hi3403V100 SVP_NNN上的om模型转换命令
        ```bash
        # 需要完成数据集下载
        python3 ./script/generate_quant_bin.py --data_path "./datasets/coco/val2017/000000000139.jpg" --img_format NCHW
        atc --framework=5 --model="./model/yolov5s.onnx" --input_shape="images:1,3,640,640" --output="./model/yolov5s" --image_list="./data/000000000139.bin" --soc_version=SS928V100
        ```
    2. Hi3403V100 NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/yolov5s.onnx"  --input_shape="images:1,3,640,640" --insert_op_conf="./model_cfg/SS928V100_NNN/insert_op.cfg" --output="./model/yolov5s" --enable_single_stream=true --soc_version=OPTG
        ```
   
        运行成功后生成yolov5s.om模型文件。

        参数说明：
      
        - --framework：5代表ONNX模型。
        - --model：为ONNX模型文件。
        - --input_shape：输入数据的shape。
        - --insert_op_conf：aipp算子配置，用于输入数据处理。
        - --output：输出的OM模型。
        - --image_list: 量化校准数据。
        - --enable_single_stream:推理时使用一条stream。
        - --soc_version：处理器型号。


## 精度&性能评估<a name="section741711594518"></a>


**步骤2：运行应用。**

1.  登录到板端运行环境，切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出的选框会保存在./out/result/txt/目录下

    ```
    ./main --acl ../src/acl.json --model ../model/yolov5s.om --input ../data/file_list.json
    ```

2. 精度验证。

    调用脚本可以获得精度数据。

    ```
    python3 ./script/accuracy.py --ground_truth_json "./datasets/coco/instances_val2017.json" --output "./out/result/txt"
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/txt/

    - --ground_truth_json：真值标签文件所在路径。

    SVP_NNN平台上精度结果：
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.563
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.293
    ```
    NNN平台上精度结果：
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.395
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.208
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.662
    ```
3. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行./main --model ../model/yolov5s.om --input ../data/file_list_1.txt
    ```
    - --model：om模型路径。
    - --input: 指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可,循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值）


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，yolov5s模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | AP（IoU=0.50） | AP（IoU=0.50:0.95） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | coco2017  | 56.3%       | 36.5%         | 85.412 |
| Hi3403V100 NNN | 1          | coco2017  | 56.5%       | 36.9%         | 35.541 |
