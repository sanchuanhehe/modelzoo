# 基于yolov8l网络实现目标检测
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

YOLO系列网络模型是最为经典的one-stage算法，也是目前工业领域使用最多的目标检测网络，YOLOv8在之前的YOLO版本的基础上进行了改进，在继承了原有YOLO网络模型优点的基础上，引入了新的特效和优化，具有更高的检测精度。

本示例使用YOLOv8s

- 参考实现：

  ```
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        |
  | -------- | -------- | ----------- |
  | output0  | FP32     | 84x8400 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── accuracy.py         //推理精度脚本
│   ├── drawRectangle.py        //画框脚本
│   ├── preprocess.py     //数据预处理脚本
│   ├── pth2onnx.py     //pt转onnx文件脚本

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


| 芯片型号  | npu     | soc_version | 环境准备指导  | cann包版本 | 编译工具链 | os  | sdk  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/Hispark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | SPC 022  |  aarch64-mix210-linux-gcc |  linux  |  SPC 022  |



# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码
 
   ```
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   git reset --hard 7a7c8dc7b70cf4bc0be18763a6b66805974ecbe6
   pip3 install -e .
   cd  ..
   ```


2. 安装依赖。

   ```
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型使用 [coco2017 val数据集](https://cocodataset.org/#download) 进行精度评估，在`yolov8`源码根目录下新建`coco`文件夹，数据集放到`coco`里，文件结构如下：

   ```
   coco
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

   执行preprocess.py 脚本，完成数据预处理。

   ```
   python preprocess.py --input_dir ../coco/val2017/ --output_dir ../coco
   ```

   参数说明：

   - --input_dir：原数据集所在路径。
   - --output_dir：预处理后的图片保存路径，建议放置在coco目录下，与val2017同级；同时会在该目录下生成文件列表文件file_list.txt
   
   

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   在[链接](https://github.com/ultralytics/assets/releases/tag/v0.0.0)中找到所需版本下载，也可以使用下述命令下载。
      ```
      wget https://github.com/ultralytics/assets/releases/download/v0.0.0/${model}.pt
      ```
   参数说明：

   - {model}:可选yolov8[n/s/m/l/x]

2. 导出onnx文件。

      使用./script/pth2onnx.py导出onnx模型

      ```
      python pth2onnx.py
      ```
      在安装ultralytics后，也可以使用下述命令直接导出onnx
      ```
      yolo export model="yolov8l.pt" format=onnx opset=13 project=export name="yolov8l.onnx"
      ```

3. 使用ATC工具将ONNX模型转OM模型。

      执行ATC命令。
      ```
      atc --framework=5 --model="./model/yolov8l.onnx" --input_shape="images:1,3,640,640" --insert_op_conf="./model_cfg/SS928V100_SVP_NNN/insert_op.cfg" --output="model/yolov8l" --image_list="./data/image_ref_list.txt" --soc_version=SS928V100 --compile_mode=6
      ```
      运行成功后生成yolov8s.om模型文件。
    
      参数说明：
    
      - --framework：原始框架类型，5代表ONNX模型。
      - --model：ONNX模型文件路径。
      - --input_shape：输入数据的shape。
      - --insert_op_conf：插入图像预处理的配置
      - --output：输出的OM模型路径。
      - --image_list：转换模型生成量化参数时用的校准数据
      - --soc_version：处理器型号。
      - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：选取其他编译模式可能导致精度下降
      


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

    ```
    ./main ../../src/acl.json ../../model/yolov8l.om ../../coco/file_list.txt
    ```
    结果会保存在数据集所在目录下的result目录下，推理结果会保存在result目录下的bin目录下，后处理后的box结果会保存在result目录下的txt目录下

**步骤3：输出后处理**

1. 精度验证。

   调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result.json中。

   ```
   python accuracy.py --bin_dir ../coco/result/bin --img_dir ../coco/val2017 --output_json ../coco/result.json --gt_annotations ../coco/annotations/instances_val2017.json
   ```

   参数说明：

   - --bin_dir：推理结果文件路径

   - --img_dir：原数据集所在路径

   - --output_json：脚本推理结果的json文件路径

   - --gt_annotations：数据集标注文件路径

   SVP_NNN平台上精度结果：
   ```
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.690
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.557
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.337
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.381
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.617
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.658
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.487
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.715
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.804

   ```

2. 验证batch_size的om模型的性能，参考命令如下：

   ```
   执行./main ../../src/acl.json ../../model/yolov8l.om ..data/file_list.txt 100
   ```

   参数说明：(此模式下，输入路径为一张图片)

   - --model：om模型路径。
   
   - --output:  后处理后结果所在位置
   
   - --model: 模型所在位置
   
   - --loop： 循环执行多少次取结果

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```
   [INFO]  time: 9384209, fps: 10.656199
   ```

3. 可视化推理结果
   调用脚本，可以对推理结果进行可视化画框操作。
   ```
   python drawRectangle.py --image ../data/coco/val2017/000000006818.jpg --annotation ../data/coco/result/txt/00000006818_result.txt
   ```

   参数说明：

   - --image:原始图片路径

   - --annotation：图片对应的推理结果文件路径


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Yolov8s模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | AP（IoU=0.50） | AP（IoU=0.50:0.95） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | coco2017  | 51.5%       | 69.0%         | 10.656 |
