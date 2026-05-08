# 基于yolo6s网络实现目标检测
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

YOLOv6s 是一种轻量高效的 one-stage 目标检测模型。相比前代 YOLO 模型，YOLOv6s 采用了 EfficientRep 作为 backbone 和 Rep-PAN 作为颈部网络，兼顾了检测精度与推理速度，更适用于边缘计算场景。

- 参考实现：

  ```
  https://github.com/meituan/YOLOv6
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | output0  | FP32     | 8400x85 | NCHW |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess        //数据预处理结果
│   ├── result            //模型推理结果
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── yolo6s_preprocess.py     //数据预处理脚本
│   ├── yolo6s_infer.py          //模型推理脚本
│   ├── yolo6s_postprocess.py    //数据后处理脚本
│   ├── yolo6s_evaluate.py       //精度评估脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── ...	      //模型文件

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
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9j2lpl16k00)上进行下载。

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

4. 切换到可执行文件main所在的目录，运行可执行文件。
    
    ```
    ./main --acl ../src/acl.json --model ../model/yolov6s.om  --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用 Python 3.7.5
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
         ├── instances_val2017.json
   ...
   ```

2. 生成file_list.json

   main函数从file_list.json文件读取输入文件列表进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:

   ```
   data
      ├── file_list.json
      ├── file_list_1.json（单张图片输入，但是多了loop参数，适合main函数循环推理测试性能）
      ├── cfg.txt（配置文件）
   ```
   
   执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    ```bash
    # 在当前项目根目录下执行，生成的文件在data/file_list.json
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```bash
    python3 ../../../../utils/generate_file_list.py ../../../../datasets/coco2017/val2017
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取参考代码仓源码
 
   ```bash
   git clone https://github.com/meituan/YOLOv6.git
   cd YOLOv6
   git checkout e86a483f3f6bded25d45970b56831345a99744a4
   cd ..
   ```

2. 获取权重文件。

   在 https://github.com/meituan/YOLOv6 中找到所需版本下载，也可以使用下述命令下载。
      ```
      wget https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
      ```

3. 导出onnx文件。

      使用YOLOv6/deploy/ONNX/export_onnx.py导出onnx模型

      ```bash
      mkdir -p model
      cd YOLOv6
   
      python3 ./deploy/ONNX/export_onnx.py \
         --weights yolov6s.pt \
         --img 640 \
         --batch 1 \
         --simplify
   
      cd ..
      mv YOLOv6/yolov6s.onnx ./model
      ```

4. 使用ATC工具将ONNX模型转OM模型。

      执行ATC命令。
      1. Hi3403V100 SVP_NNN上的om模型转换命令
         ```bash
         # 如果您没有现成的bin文件，您需要参考后面的python脚本使用说明，使用preprocess.py脚本先生成bin文件。如果您想使用多个bin文件进行数据校准，多个文件间请使用;分割，例如a.bin;b.bin。
         atc --framework=5 --model="model/yolov6s.onnx" --input_shape="images:1,3,640,640" --output="model/yolov6s" --soc_version=SS928V100 --image_list="data/preprocess/bin/000000006818.bin" --compile_mode=6
         ```
      2. Hi3403V100 NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/yolov6s.onnx" --input_shape="images:1,3,640,640" --output="model/yolov6s" --enable_single_stream=true --input_fp16_nodes="images" --soc_version=OPTG
         ```
         运行成功后生成yolov6s.om模型文件。
    
         参数说明：
    
         - --framework：原始框架类型，5代表ONNX模型。
         - --model：ONNX模型文件路径。
         - --input_shape：输入数据的shape。
         - --output：输出的OM模型路径。
         - --image_list：转换模型生成量化参数时用的校准数据。
         - --soc_version：处理器型号。
         - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：SVP_NNN上选取其他编译模式可能导致精度下降
      
         注意：如果出现atc命令找不到，参考推理环境准备。


## 精度&性能评估<a name="section741711594518"></a>

1. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数。
    - 注意点1：cfg.txt中默认配置了推理时保存前处理和模型推理生成的原始bin文件，会占据较大的磁盘空间，您可以将该配置改为空字符串不保存节约磁盘空间。
    - 注意点2：conf_threshold默认设置为0.03，是为了在评估精度时提供尽可能全的检测框计算ROC曲线，实际推理时conf_threshold一般设置为0.25。
    ```bash
    ######################### 建议按照如下配置修改配置文件 #########################
    # 预处理后的二进制文件，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_preprocess_bin=""
    
    # 模型推理原始二进制结果，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_result_bin=""
    
    # 推理结果json文件保存路径
    save_result_txt="../out/result/txt"
    
    # 置信度阈值（过滤低置信度候选框，默认设置为0.001，是为了为精度评估脚本提供尽可能全的检测框计算ROC曲线，实际推理时conf_threshold一般设置为0.25）
    conf_threshold=0.03
    
    # NMS交并比阈值（过滤重复检测框）
    nms_threshold=0.65
    ```

2.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/yolov6s.om  --input ../data/file_list.json
    ```
    结果会保存在数据集所在目录下的result目录下，推理结果会保存在result目录下的bin目录下

3. 精度验证。

   调用脚本与数据集标签person_keypoints_val2017.json比对，可以获得模型在coco2017验证集上的评估结果。

   ```bash
   cd script # 切换到script目录
   # 更改下开发板输出文件的权限，确保脚本能访问
   sudo chmod -R 777 ../out
   python3 yolo6s_evaluate.py \
      --result_dir "../out/result/txt" \
      --gt_annotations "../../../../../datasets/coco2017/annotations/instances_val2017.json"
   ```

   参数说明：

   - --result_dir：存放检测结果.txt文件的目录。脚本会读取该目录下所有.txt文件，解析其中的边界框、置信度、类别ID等信息，作为待评估的“预测结果”。

   - --gt_annotations：COCO数据集的ground truth（真值）标注文件路径。脚本通过pycocotools加载该文件，获取图片的真实边界框、类别等信息，作为评估的“基准真值”，用于计算mAP、AR等指标。

   SVP_NNN平台上精度结果：
   ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.578
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.468
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.583
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.536
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.354
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.656
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.746
   ```
   NNN平台上精度结果：
   ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.610
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.478
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.494
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.556
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.680
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
   ```
   

4. 验证om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/yolov6s.om  --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，输入路径为一张图片)
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 23.89ms, frame rate: 41.86fps
   ```
   在板端会输出显示，NNN平台上性能结果如下：
      ```bash
   execution time: 33.07ms, frame rate: 30.24fps
   ```

**（可选步骤）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 yolo6s_preprocess.py \
      --file_list "../data/file_list.json" \
      --output_dir "../data/preprocess/bin" \
      --target_size 640 640
   ```

   参数说明：

   - --file_list：输入的 JSON 文件路径，内含待处理图片的路径列表（格式需符合脚本预期：包含 fileList 字段，每个元素为图片路径）。脚本会从该文件读取所有待转换的图片路径。
   - --output_dir：预处理后生成的 .bin 文件保存目录。若目录不存在，脚本会自动创建。
   - --target_size：图片缩放填充后的目标尺寸，格式为 宽 高（与脚本中 letterbox 函数的 target_size 参数对应）。

2. 使用python脚本进行模型推理（可以跟CPP版本的推理结果进行对比）
   ```bash
   cd script # 切换到script目录
   python3 yolo6s_infer.py \
      --preprocess_bin_dir "../data/preprocess/bin" \
      --infer_bin_dir "../out/result_pc/bin" \
      --file_list_path "../data/preprocess/file_list.txt" \
      --onnx_model_path "../model/yolov6s.onnx" \
      --input_size 640 640
   ```

   参数说明：

   - --preprocess_bin_dir: 预处理后生成的二进制（.bin）文件存放目录。

   - --infer_bin_dir：推理结果（.bin 格式）的输出目录。若目录不存在，脚本会自动创建。

   - --file_list_path：python预处理阶段生成的文件列表（记录所有待推理的 .bin 文件路径）。脚本会从该文件读取待处理文件路径。

   - --onnx_model_path：待加载的 ONNX 模型文件路径。脚本会验证该文件是否存在，不存在则报错。
   
   - --input_size：模型输入张量的尺寸，格式为 高 宽（与脚本中 input_h, input_w 对应）。需与预处理阶段的目标尺寸一致，否则会导致张量形状不匹配。

4. 使用python脚本进行模型推理结果的后处理（解码bin文件、置信度过滤、分类NMS等操作）
   ```bash
   cd script # 切换到script目录
   python3 yolo6s_postprocess.py \
      --bin_dir "../out/result_pc/bin" \
      --img_dir "../../../../../datasets/coco2017/val2017" \
      --output_dir "../out/result_pc/txt" \
      --nms_threshold 0.65 \
      --conf_threshold 0.03 \
      --target_size 640 640
   ```

   参数说明：

   - --bin_dir: 存放 YOLO 模型推理输出的 .bin 文件目录。脚本会先验证该目录是否存在，不存在则报错。

   - --img_dir：原始图片（如 JPG）的存放目录。脚本需读取图片获取原始尺寸，用于还原边界框到真实大小。

   - --output_dir：后处理结果（.txt 文件）的输出目录。若目录不存在，脚本会自动创建。

   - --nms_threshold：NMS（非极大值抑制）的 IOU 阈值。用于过滤重叠度过高的检测框，值越大保留的重叠框越多（默认 0.65）。
   
   - --conf_threshold：置信度过滤阈值。仅保留检测置信度 ≥ 该值的结果，值越大过滤越严格（默认 0.03）。
   
   - --target_size：模型输入尺寸，格式为 宽 高（需与预处理、推理阶段的输入尺寸完全一致）。脚本会根据该尺寸计算图片缩放比例和填充量，用于还原边界框。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | mAP（IoU=0.50:0.95） | mAP（IoU=0.50） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | coco2017  | 41.9%       | 57.8%         | 41.86 |
| Hi3403V100 NNN | 1          | coco2017  | 44.3%       | 61.0%         | 30.24 |
