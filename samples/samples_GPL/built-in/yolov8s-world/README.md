# 基于Yolov8s-World网络实现目标检测
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

Yolov8s-World 是 Ultralytics 推出的一种基于 YOLO-World 架构的轻量级目标检测模型，它通过视觉-语言预训练实现了无需针对特定类别进行训练即可识别任意物体的 “开放词汇” (Open-Vocabulary) 实时检测功能。

- 参考实现：

  ```
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-worldv2.yaml
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |
  | text_feature(仅SVP_NNN算力引擎下存在)  | RGB_FP32 | 1 x 80 x 512 | NCHW         |


- 输出数据

  | 输出数据 | 数据类型 | 大小        |
  | -------- | -------- | ----------- |
  | output  | FP32     | 1x84x8400 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── accuracy.py         //推理精度脚本
│   ├── drawRectangle.py        //画框脚本
│   ├── pth2onnx
│   ├── ├──pth2onnx.py     //pt转onnx文件脚本
│   ├── deploy             //模型量化脚本依赖

├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── model_cfg
│   ├── SS928V100_NNN	//模型配置文件
|   |	├── insert_op.cfg		//aipp配置文件
│   ├── SS928V100_SVP_NNN	//模型配置文件
|   |	├── insert_op.cfg		//aipp配置文件

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
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | 6.10.t01spc030b700(请联系FAE获取)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  6.10.t01spc030b700(请联系FAE获取)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t14.7.b140  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |



# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j93dspootk00)上进行下载。
如果在 SVP_NNN 算力引擎上推理还要在[网站](https://modelzoo.hispark.hisilicon.com/#/ModelZoo)的Yolov8s-World主页`源模型下载`处下载`text_feature.zip`文件，并将解压得到的`text_feature.bin`拷贝到`data`目录下。

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
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_linux.cmake -DSOC_VERSION=SS928V100
    ```

3. 执行**make**命令，生成的可执行文件main在“./out“目录下。


#### 运行应用

1. 将modelzoo代码上传到板端运行环境。
2. 以运行用户登录板端运行环境。
3. 切换到可执行文件main所在的目录，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4. 切换到可执行文件main所在的目录，运行可执行文件。
    
    ```
    ./main --model ../model/yolov8s-worldv2.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议安装两套环境 
# Python 3.8.20(Hi3403V100 NNN芯片时转ONNX以及两种芯片的精度评估用) 
pip install -r requirements38.txt
# Python 3.7.5(Hi3403V100 SVP_NNN芯片转ONNX时用)
pip install -r requirements375.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`modelzoo/datasets`目录下新建`coco`文件夹，数据集放到`coco`里，文件结构如下：

   ```
   datasets
      ├──coco
         ├── val2017
            ├── 00000000139.jpg
            ├── 00000000285.jpg
            ……
            └── 00000581781.jpg
         ├── instances_val2017.json
   ...
   ```
2. 生成数据集目录文件
   ```
   python3 ../../../../utils/generate_file_list.py ../../../../datasets/coco/val2017
   ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取模型源码
 
   ```
   git clone https://github.com/ultralytics/ultralytics.git ultralytics-master
   cd ./ultralytics-master
   git reset --hard cf081c4db7972a4fe42794a234d867ce0930b4b2
   cd ../
   ```

2. 获取权重文件。

   通过[链接](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt)将yolov8s-worldv2.pt下载，存储至 model 文件夹。
   ```
   mkdir -p model
   ```

3. 导出onnx文件。
      Hi3403V100 SVP_NNN上的onnx模型转换命令,建议将python 版本切换为 3.7.5
      ```
      # 安装CANN包版本对应的AMCT, MindCmd(联系FAE获取)

      # 对 ultralytics 项目打补丁,代码如下
      cp ./data/amct_yolov8s_world.patch ultralytics-master
      cd ./ultralytics-master
      git apply --reject amct_yolov8s_world.patch
      cd ../
      cp ultralytics-master/ultralytics script -R

      # 修正 script/ultralytics/cfg/datasets/coco.yaml 文件path和val路径(建议使用绝对路径)

      # 如报错OSError: libcudart.so.11.0: cannot open shared object file: No such file or directory, 
      # 将libcudart.so.11.0目录添加到环境变量中, 
      # 例如 export LD_LIBRARY_PATH=/home/hispark/.local/lib/python3.7/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH;
      python ./script/yolov8_world_a8w8_ptq.py --model ./model/yolov8s-worldv2.pt

      # 将量化文件移植对应目录
      cp script/work_dir/yolov8s-worldv2/*.bin script/work_dir/yolov8s-worldv2/*.txt data
      cp script/work_dir/yolov8s-worldv2/*.onnx model
      ```

      Hi3403V100 NNN上的onnx模型转换命令,建议将python 版本切换为 3.8.20
      ```
      python3 script/pt2onnx/pth2onnx.py
      ```

4. 使用ATC工具将ONNX模型转OM模型。

      执行ATC命令。

      Hi3403V100 SVP_NNN上的om模型转换命令
      ```
      atc --input_type="images:UINT8;text_feature:FP32" --input_format=NCHW --output="./model/yolov8s-worldv2" --soc_version=SS928V100 --insert_op_conf=./model_cfg/SS928V100_SVP_NNN/insert_op.cfg --framework=5 --compile_mode=0 --model="./model/yolov8s-worldv2_deploy_model.onnx" --image_list="images:./data/image_ref_list.txt;text_feature:./data/text_feature.bin" --amplify_conv_alpha_enable=1 --gfpq_param_file="./data/yolov8s-worldv2_quant_param_record.txt"
      ```

      Hi3403V100 NNN上的om模型转换命令
      ```
      atc --framework=5 --model="./model/yolov8s-worldv2.onnx"  --input_shape="images:1,3,640,640" --insert_op_conf="./model_cfg/SS928V100_NNN/insert_op.cfg" --output="./model/yolov8s-worldv2" --enable_single_stream=true --soc_version=OPTG
      ```
       
      运行成功后生成yolov8s-worldv2.om模型文件。
    
      参数说明：
    
      - --framework：原始框架类型，5代表ONNX模型。
      - --model：ONNX模型文件路径。
      - --input_shape：输入数据的shape。
      - --insert_op_conf：插入图像预处理的配置
      - --output：输出的OM模型路径。
      - --image_list：转换模型生成量化参数时用的校准数据。数据获取参考[准备数据集](#section183221994411)章节。
      - --enable_single_stream：推理时使用一条stream。
      - --soc_version：处理器型号。
      - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：选取其他编译模式可能导致精度下降
      


## 精度&性能评估<a name="section741711594518"></a>

1. 切换到可执行文件main所在的目录，数据集上运行命令参考：

   ```
   ./main --model ../model/yolov8s-worldv2.om --input ../data/file_list.json
   ```
    结果会保存在数据集所在目录下的result目录下，推理结果会保存在result目录下的bin目录下，后处理后的box结果会保存在result目录下的txt目录下

2. 精度验证。

   ```
   python script/accuracy.py --ground_truth_json ${ground_truth_json}
   # 例如
   python3 script/accuracy.py --ground_truth_json ../../../../datasets/coco/instances_val2017.json
   ```

   参数说明：
   - --ground_truth_json：数据集标注文件路径, 比如../../../../datasets/coco/instances_val2017.json

   SVP_NNN平台上精度结果：
   ```
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.503
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.386
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.390
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.478
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.506
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.555
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.351
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
   ```

   NNN平台上精度结果：
   ```
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.400
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.628
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.712
   ```

3. 验证om模型的性能，参考命令如下：
   ```
   ./main --model ../model/yolov8s-worldv2.om --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，输入路径为一张图片,file_list_1.json 中 loop 参数建议设定1000)

   - --model：om模型路径。
   - --output:  后处理后结果所在位置
   - --model: 模型所在位置

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```
   execution time: 23.98ms, frame rate: 41.71fps
   ```
   NNN平台上性能结果如下：
   ```
   execution time: 48.45ms, frame rate: 20.64fps
   ```

4. 可视化推理结果(可选)
   调用脚本，可以对推理结果进行可视化画框操作。
   ```
   cd script
   python3 drawRectangle.py -i ../../../../../../../modelzoo-dev/datasets/coco/val2017/000000000139.jpg -r ../out/result/txt/000000000139_result.txt -o ../out/000000000139_show.jpg --iou 0.45
   cd ../
   ```

   参数说明：
      -i, --image：输入图片路径（必需）
      -r, --result：检测结果文件路径（必需）
      -t, --iou：IOU阈值（可选，默认0.45）
      -o, --output：输出文件名（可选，默认demo_show.jpg）

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Yolov8s-World模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | AP（IoU=0.50） | AP（IoU=0.50:0.95） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | coco2017  | 50.3%       | 35.6%         | 41.71 |
| Hi3403V100 NNN | 1          | coco2017  | 51.5%       | 36.9%         | 20.64 |
