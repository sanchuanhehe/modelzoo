# 基于YOLOv9s网络实现目标检测

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

YOLOv9s 是 Ultralytics 推出的目标检测模型，采用了 GELAN 架构和 PGI 训练策略。相比于之前的版本，YOLOv9 在保持高效推理的同时，通过可编程梯度信息（PGI）进一步提升了检测精度。该模型在 COCO 数据集上进行了训练和验证。

- 参考实现：

  ```
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v9/yolov9s.yaml
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | output0  | FP32     | 1 x 84 x 8400 | NCHW |

## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── out
│   ├── preprocess        //C++预处理结果
│   ├── result            //C++模型推理结果和后处理结果
├── data
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS
├── script
│   ├── yolov9s_preprocess.py     //数据预处理脚本
│   ├── yolov9s_evaluate.py       //精度评估脚本
│   ├── export_onnx.py            //ONNX 导出脚本
├── src
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口
├── model
│   ├── ...           //模型文件
├── LICENSE           //模型LICENSE
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

| 芯片型号  | 算力引擎 | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux  | SS928 V100R001C02SPC022 |
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t14.7.b140  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j9brj6ooi000)上进行下载。

创建`model`文件夹，并将om模型文件移动到`./model`目录下。
```
mkdir -p model
```
备注：若需要体验om模型转化过程，请参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[模型转化](#section741711594517)章节。

### 编译代码和运行应用

#### 编译代码

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为`build`。
    ```
    mkdir -p build
    ```

2. 切换到`build`目录，执行**cmake**生成编译文件。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    "../src"表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    例如，开发环境为X86架构、运行环境为ARM架构时，执行以下命令进行交叉编译。交叉编译工具链按运行环境操作系统，可选toolchain_aarch64_linux.cmake或toolchain_aarch64_ohos.cmake；SOC_VERSION按算力引擎可选SS928V100或OPTG，请根据运行环境和算力引擎平台选择。
    ```
    cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${toolchain.cmake} -DSOC_VERSION=${soc_version}
    ```
    比如
    ```
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake -DSOC_VERSION=SS928V100
    ```

3. 执行**make**命令，生成的可执行文件main在"./out"目录下。


#### 运行应用

1. 将modelzoo代码上传到板端运行环境。
2. 以运行用户登录板端运行环境。
3. 切换到可执行文件main所在的目录，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4. 切换到可执行文件main所在的目录，运行可执行文件。测试图片上模型推理命令参考：
    
    ```bash
    cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov9s/out
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"
    ./main --model ../model/yolov9s.om  --input ../data/file_list.json
    ```
    图片推理结果会保存在 `out/result/bin` 目录下。

    备注：若需要在COCO全量验证集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

1. 获取 YOLOv9 (Ultralytics) 源码

   ```bash
   cd samples/samples_GPL/built-in/yolov9s/
   git clone https://github.com/ultralytics/ultralytics.git -b v8.4.10 --depth=1
   cd ultralytics
   pip3 install -e .
   cd  ..
   ```

2. 安装依赖。

   ```bash
   # 建议使用Python 3.9.23
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   该模型使用 [COCO2017数据集](https://cocodataset.org/#download) 进行训练和验证。

   在 `samples/samples_GPL/built-in/yolov9s/` 目录下创建 `datasets` 文件夹，
   将下载的 COCO 数据集解压到该目录。
   
   数据集结构如下：

   ```
   datasets/coco/
   |-- images/
   |   |-- val2017/         # 验证集图片
   |-- labels/
   |   |-- val2017/         # 验证集标签
   |-- annotations/         # 标注文件
   |   |-- instances_val2017.json
   ...
   ```

2. 生成文件集file_list.json，将原始数据集图片地址转换为模型的输入数据。

    执行 ../../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。

    ```
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```
    python3 ../../../../utils/generate_file_list.py datasets/coco/images/val2017/
    ```

    备注：data/file_list.json中默认只有3张示例图片。执行上述命令后，会生成包含全量数据集的file_list.json。

## 模型转化<a name="section741711594517"></a>

使用 Ultralytics 导出 ONNX，再使用 ATC 工具转为 OM 模型。

1. 导出onnx。

   在 `samples/samples_GPL/built-in/yolov9s` 目录下执行，生成 `model/yolov9s.onnx` 文件。

   ```sh
   mkdir model && cd model
   python ../script/export_onnx.py
   cd ..
   ```

2. 生成模型校准数据。
   
      选取几张图片生成模型校准数据，引用的图片数据默认在samples/samples_GPL/built-in/yolov9s/data/file_list.json中描述（当前包含示例图片，用于量化校准已足够）。校准数据文件默认保存在out/preprocess/bin目录下。

      ```sh
      cd script/
      python yolov9s_preprocess.py
      cd ..
      ```

3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      Hi3403V100 SVP_NNN 上的 om 模型转换命令

      ```bash
      # 需确保 out/preprocess/bin 下有用于量化的校准数据，或去掉 --image_list 参数
      # 如果包含--image_list参数，需指定一个真实存在的 bin 文件路径（例如 out/preprocess/bin/000000000139.bin），请检查实际生成的文件名
      atc --framework=5 --model="model/yolov9s.onnx" --input_shape="images:1,3,640,640" --output="model/yolov9s" --soc_version=SS928V100 --image_list="out/preprocess/bin/000000000139.bin;out/preprocess/bin/000000000285.bin;out/preprocess/bin/000000000632.bin" --compile_mode=6
      ```
      
      Hi3403V100 NNN上的 om 模型转换命令
      ```
      atc --framework=5 --model="model/yolov9s.onnx" --input_format="NCHW" --input_shape="images:1,3,640,640" --input_fp16_nodes="images" --output="./model/yolov9s" --enable_single_stream=true --soc_version=OPTG
      ```
      运行成功后生成 `model/yolov9s.om` 模型文件。

      参数说明：
      
      - --framework：5代表ONNX模型。
      - --model：为ONNX模型文件。
      - --input_shape：输入数据的shape。
      - --output：输出的OM模型。
      - --image_list: 量化校准数据。
      - --enable_single_stream:推理时使用一条stream。
      - --soc_version：处理器型号。

## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。

    ```bash
    cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov9s/out
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"
    ./main --model ../model/yolov9s.om  --input ../data/file_list.json
    ```
    图片推理结果会保存在 `out/result/bin` 目录下。

2. 精度验证。

   在服务器使用 `script/yolov9s_evaluate.py` 进行精度评估。

   ```bash
   python script/yolov9s_evaluate.py --input out/result/txt/
   ```
   
   SVP_NNN平台上精度结果：
   
   ```
   ============================================================
   Class                | mAP@0.5    | mAP@0.5:0.95
   ------------------------------------------------------------
   person               | 0.804      | 0.585     
   bicycle              | 0.553      | 0.339     
   car                  | 0.654      | 0.439     
   motorcycle           | 0.723      | 0.494     
   airplane             | 0.922      | 0.758     
   bus                  | 0.798      | 0.69      
   train                | 0.896      | 0.725     
   truck                | 0.502      | 0.354     
   boat                 | 0.501      | 0.279     
   traffic light        | 0.504      | 0.273     
   fire hydrant         | 0.874      | 0.715     
   stop sign            | 0.764      | 0.693     
   parking meter        | 0.625      | 0.501     
   bench                | 0.39       | 0.277     
   bird                 | 0.518      | 0.357     
   cat                  | 0.9        | 0.766     
   dog                  | 0.778      | 0.667     
   horse                | 0.807      | 0.637     
   sheep                | 0.762      | 0.566     
   cow                  | 0.735      | 0.558     
   elephant             | 0.877      | 0.708     
   bear                 | 0.874      | 0.757     
   zebra                | 0.917      | 0.728     
   giraffe              | 0.914      | 0.748     
   backpack             | 0.263      | 0.149     
   umbrella             | 0.649      | 0.466     
   handbag              | 0.287      | 0.165     
   tie                  | 0.547      | 0.363     
   suitcase             | 0.623      | 0.439     
   frisbee              | 0.846      | 0.666     
   skis                 | 0.507      | 0.281     
   snowboard            | 0.517      | 0.398     
   sports ball          | 0.57       | 0.41      
   kite                 | 0.634      | 0.449     
   baseball bat         | 0.569      | 0.362     
   baseball glove       | 0.62       | 0.398     
   skateboard           | 0.782      | 0.584     
   surfboard            | 0.629      | 0.425     
   tennis racket        | 0.801      | 0.56      
   bottle               | 0.556      | 0.389     
   wine glass           | 0.532      | 0.364     
   cup                  | 0.577      | 0.432     
   fork                 | 0.57       | 0.425     
   knife                | 0.321      | 0.198     
   spoon                | 0.303      | 0.206     
   bowl                 | 0.595      | 0.458     
   banana               | 0.413      | 0.278     
   apple                | 0.298      | 0.217     
   sandwich             | 0.518      | 0.387     
   orange               | 0.385      | 0.305     
   broccoli             | 0.404      | 0.246     
   carrot               | 0.35       | 0.222     
   hot dog              | 0.492      | 0.379     
   pizza                | 0.741      | 0.579     
   donut                | 0.603      | 0.486     
   cake                 | 0.583      | 0.4       
   chair                | 0.515      | 0.341     
   couch                | 0.639      | 0.495     
   potted plant         | 0.477      | 0.297     
   bed                  | 0.672      | 0.517     
   dining table         | 0.492      | 0.347     
   toilet               | 0.831      | 0.676     
   tv                   | 0.8        | 0.626     
   laptop               | 0.774      | 0.67      
   mouse                | 0.786      | 0.605     
   remote               | 0.53       | 0.337     
   keyboard             | 0.744      | 0.571     
   cell phone           | 0.537      | 0.379     
   microwave            | 0.768      | 0.601     
   oven                 | 0.571      | 0.413     
   toaster              | 0.69       | 0.508     
   sink                 | 0.612      | 0.422     
   refrigerator         | 0.798      | 0.678     
   book                 | 0.256      | 0.142     
   clock                | 0.725      | 0.518     
   vase                 | 0.536      | 0.387     
   scissors             | 0.43       | 0.335     
   teddy bear           | 0.688      | 0.552     
   hair drier           | 0.0691     | 0.0425    
   toothbrush           | 0.452      | 0.298     
   ------------------------------------------------------------
   ALL                  | 0.613      | 0.456     
   ============================================================
   ```
   
   NNN平台上精度结果：
   
   ```
   ============================================================
   Class                | mAP@0.5    | mAP@0.5:0.95
   ------------------------------------------------------------
   person               | 0.804      | 0.588     
   bicycle              | 0.557      | 0.344     
   car                  | 0.656      | 0.444     
   motorcycle           | 0.72       | 0.498     
   airplane             | 0.924      | 0.759     
   bus                  | 0.798      | 0.692     
   train                | 0.901      | 0.726     
   truck                | 0.5        | 0.359     
   boat                 | 0.5        | 0.28      
   traffic light        | 0.515      | 0.281     
   fire hydrant         | 0.871      | 0.714     
   stop sign            | 0.767      | 0.694     
   parking meter        | 0.622      | 0.489     
   bench                | 0.39       | 0.277     
   bird                 | 0.516      | 0.363     
   cat                  | 0.897      | 0.762     
   dog                  | 0.786      | 0.678     
   horse                | 0.799      | 0.631     
   sheep                | 0.762      | 0.567     
   cow                  | 0.74       | 0.567     
   elephant             | 0.876      | 0.71      
   bear                 | 0.863      | 0.751     
   zebra                | 0.917      | 0.73      
   giraffe              | 0.916      | 0.742     
   backpack             | 0.266      | 0.152     
   umbrella             | 0.648      | 0.471     
   handbag              | 0.289      | 0.167     
   tie                  | 0.55       | 0.365     
   suitcase             | 0.63       | 0.442     
   frisbee              | 0.846      | 0.671     
   skis                 | 0.506      | 0.286     
   snowboard            | 0.517      | 0.398     
   sports ball          | 0.572      | 0.424     
   kite                 | 0.638      | 0.45      
   baseball bat         | 0.568      | 0.365     
   baseball glove       | 0.621      | 0.398     
   skateboard           | 0.783      | 0.584     
   surfboard            | 0.623      | 0.43      
   tennis racket        | 0.812      | 0.564     
   bottle               | 0.557      | 0.393     
   wine glass           | 0.534      | 0.367     
   cup                  | 0.579      | 0.435     
   fork                 | 0.565      | 0.429     
   knife                | 0.316      | 0.197     
   spoon                | 0.312      | 0.21      
   bowl                 | 0.598      | 0.463     
   banana               | 0.418      | 0.283     
   apple                | 0.305      | 0.221     
   sandwich             | 0.517      | 0.393     
   orange               | 0.388      | 0.308     
   broccoli             | 0.404      | 0.245     
   carrot               | 0.351      | 0.225     
   hot dog              | 0.497      | 0.379     
   pizza                | 0.736      | 0.577     
   donut                | 0.606      | 0.491     
   cake                 | 0.585      | 0.405     
   chair                | 0.516      | 0.344     
   couch                | 0.647      | 0.507     
   potted plant         | 0.481      | 0.299     
   bed                  | 0.68       | 0.522     
   dining table         | 0.495      | 0.35      
   toilet               | 0.829      | 0.683     
   tv                   | 0.802      | 0.63      
   laptop               | 0.772      | 0.675     
   mouse                | 0.788      | 0.617     
   remote               | 0.528      | 0.338     
   keyboard             | 0.746      | 0.578     
   cell phone           | 0.539      | 0.382     
   microwave            | 0.762      | 0.6       
   oven                 | 0.572      | 0.415     
   toaster              | 0.705      | 0.528     
   sink                 | 0.614      | 0.423     
   refrigerator         | 0.801      | 0.684     
   book                 | 0.258      | 0.142     
   clock                | 0.726      | 0.518     
   vase                 | 0.533      | 0.387     
   scissors             | 0.432      | 0.342     
   teddy bear           | 0.688      | 0.555     
   hair drier           | 0.0686     | 0.0412    
   toothbrush           | 0.455      | 0.297     
   ------------------------------------------------------------
   ALL                  | 0.615      | 0.459     
   ============================================================
   ```

3. 验证om模型的性能，参考命令如下：

    ```sh
    cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov9s/out
    ./main --model ../model/yolov9s.om  --input ../data/file_list_1.json
    ```

    参数说明：
    - --model：om模型文件路径。
    - --input: 输入图片路径文件，将file_list_1.json的loop参数设置为300

    file_list_1.json中的配置代表对一张输入图片重复推理300次，程序执行时会在板端会输出打印推理的平均时间和帧率。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | mAP@0.5 | mAP@0.5:0.95 | 性能(fps) |
| ----------- | ---------- | -------- | ------- | ------------- | --------- |
| Hi3403V100 SVP_NNN | 1          | COCO2017 | 61.3%   | 45.6%         | 35.48     |
| Hi3403V100 NNN     | 1          | COCO2017 | 61.5%   | 45.9%         | 18.07     |
