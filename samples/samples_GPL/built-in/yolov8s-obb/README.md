# 基于YOLOv8s-OBB网络实现旋转目标检测

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

YOLOv8s-OBB 是 Ultralytics 推出的基于 YOLOv8 的旋转目标检测（Oriented Bounding Box, OBB）模型。相比于水平框检测，OBB 能够更准确地检测倾斜或不规则排列的目标（如航拍图像中的车辆、船只等）。该模型在 DOTA 数据集上进行了训练和验证。

- 参考实现：

  ```
  https://github.com/ultralytics/ultralytics
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 1024 x 1024 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | output0  | FP32     | 1 x 20 x 21504 | NCHW |
  
  > **注意**：输出大小取决于模型导出时的配置。上述大小基于 `imgsz=1024`，且 DOTA 数据集（15类）导出时的典型输出 (4 box + 1 angle + 15 classes = 20 channels)。

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
│   ├── yolov8s_obb_preprocess.py     //数据预处理脚本
│   ├── yolov8s_obb_evaluate.py       //精度评估脚本
│   ├── export_onnx.py                //ONNX 导出脚本
│   ├── split_dota.py                 //DOTA 数据集切割脚本

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

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9t0hkf56k00)上进行下载。

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
    ./main --model ../model/yolov8s-obb.om  --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用Python 3.9.23
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   该模型通常使用 [DOTAv1数据集](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip) 进行训练和验证。

   在 `samples/samples_GPL/built-in/yolov8s-obb/` 目录下创建 `datasets` 文件夹（或建立软链接），
   将下载的 DOTAv1.zip 拷贝到该目录并执行 `unzip DOTAv1.zip` 进行解压。

2. 数据切割

   原始的DOTAv1数据都是航拍的大图，分辨率很高，被缩放至1024*1024后会丢失大量的图片细节，所以要先将图片进行切割。执行下面的指令：

   ```sh
   cd samples/samples_GPL/built-in/yolov8s-obb
   python script/split_dota.py --origin_path datasets/DOTAv1 --saved_path datasets/DOTAv1-split
   ```
   
   修改ultralytics/cfg/datasets/DOTAv1.yaml中的path字段为DOTAv1-split。


## 模型转化<a name="section741711594517"></a>

使用 Ultralytics 导出 ONNX，再使用 ATC 工具转为 OM 模型。

1. 获取 YOLOv8 (Ultralytics) 源码

   ```bash
   cd samples/samples_GPL/built-in/yolov8s-obb/
   git clone -b v8.3.230 https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip3 install -e .
   cd  ..
   ```

2. 导出onnx。

   在 `samples/samples_GPL/built-in/yolov8s-obb` 目录下执行，生成 `model/yolov8s-obb.onnx` 文件。

   ```sh
   mkdir model && cd model
   python ../script/export_onnx.py
   cd ..
   ```

2. 生成模型校准数据。
   
      选取几张图片生成模型校准数据，引用的图片数据默认在samples/samples_GPL/built-in/yolov8s-obb/data/file_list.json中描述（当前仅包含3张示例图片，用于量化校准已足够）。校准数据文件默认保存在out/preprocess/bin目录下。

      ```sh
      cd samples/samples_GPL/built-in/yolov8s-obb/script/
      python yolov8s_obb_preprocess.py
      ```

3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      Hi3403V100 SVP_NNN 上的 om 模型转换命令

      ```bash
      cd samples/samples_GPL/built-in/yolov8s-obb
      # 需确保 out/preprocess/bin 下有用于量化的校准数据，或去掉 --image_list 参数
      # 如果包含--image_list参数，需指定一个真实存在的 bin 文件路径（例如 out/preprocess/bin/P0146__1024__0___0.bin），请检查实际生成的文件名
      atc --framework=5 --model="model/yolov8s-obb.onnx" --input_shape="images:1,3,1024,1024" --output="model/yolov8s-obb" --soc_version=SS928V100 --image_list="out/preprocess/bin/P0146__1024__0___0.bin" --compile_mode=6
      ```
      
      Hi3403V100 NNN上的 om 模型转换命令
      ```
      atc --framework=5 --model="model/yolov8s-obb.onnx" --input_format="NCHW" --input_shape="images:1,3,1024,1024" --output="./model/yolov8s_obb" --enable_single_stream=true --soc_version=OPTG
      ```
      运行成功后生成 `model/yolov8s-obb.om` 模型文件。

## 精度&性能评估<a name="section741711594518"></a>

1.  修改配置文件 `cfg.txt`（可选，用于控制是否保存 bin 文件）。
2.  运行推理。（**your_path**是板端文件系统具体的路径前缀）

    ```bash
    cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov8s-obb/out
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"
    ./main --model ../model/yolov8s-obb.om  --input ../data/file_list.json
    ```
    图片推理结果会保存在 `out/result/bin` 目录下。
    
    > **注意**：file_list.json中当前只包含3张示例图片，需要按你的数据集路径进行修改。如果需要更多图片，可以在服务器的代码仓的modelzoo/samples/samples_GPL/built-in/yolov8s-obb目录下用`python ../../../../utils/generate_file_list.py datasets/DOTAv1-split/images/val` 生成完整的文件列表，注意核对生成的文件信息在板端文件系统的具体路径。

3. 精度验证。

   在服务器使用 `script/yolov8s_obb_evaluate.py` 进行精度评估。

   ```bash
   python3 script/yolov8s_obb_evaluate.py \
     --result_dir out/result/txt \
     --gt_annotations datasets/DOTAv1-split/labels/val \
     --img_dir datasets/DOTAv1-split/images/val
   ```
   
   **关键说明**：
   
   - 脚本会自动根据 `result_dir` 中存在的 txt 文件过滤待评估图片
   - 多进程数量可在脚本中修改 `NUM_WORKERS` 变量（默认 6）

   SVP_NNN平台上精度结果：
   
   ```
   ============================================================
   Class                | mAP@0.5    | mAP@0.5:0.95
   ------------------------------------------------------------
   plane                | 0.9368     | 0.8344
   ship                 | 0.9368     | 0.7759
   storage-tank         | 0.8280     | 0.6893
   baseball-diamond     | 0.8026     | 0.6124
   tennis-court         | 0.9166     | 0.8741
   basketball-court     | 0.6367     | 0.5656
   ground-track-field   | 0.6638     | 0.5565
   harbor               | 0.8004     | 0.5599
   bridge               | 0.5591     | 0.3412
   large-vehicle        | 0.8348     | 0.6746
   small-vehicle        | 0.7565     | 0.5839
   helicopter           | 0.7632     | 0.5568
   roundabout           | 0.7133     | 0.5431
   soccer-ball-field    | 0.5416     | 0.4529
   swimming-pool        | 0.7628     | 0.4913
   ------------------------------------------------------------
   ALL                  | 0.7635     | 0.6075
   ============================================================
   ```
   
   NNN平台上精度结果：
   
   ```
   ============================================================
   Class                | mAP@0.5    | mAP@0.5:0.95   
   ------------------------------------------------------------
   plane                | 0.9391     | 0.8393
   ship                 | 0.9388     | 0.7857
   storage-tank         | 0.8321     | 0.6996
   baseball-diamond     | 0.8029     | 0.6185
   tennis-court         | 0.9160     | 0.8748
   basketball-court     | 0.6364     | 0.5691
   ground-track-field   | 0.6646     | 0.5613
   harbor               | 0.8059     | 0.5688
   bridge               | 0.5614     | 0.3437
   large-vehicle        | 0.8375     | 0.6875
   small-vehicle        | 0.7521     | 0.5847
   helicopter           | 0.7648     | 0.5669
   roundabout           | 0.7125     | 0.5391
   soccer-ball-field    | 0.5400     | 0.4526
   swimming-pool        | 0.7664     | 0.4942
   ------------------------------------------------------------
   ALL                  | 0.7647     | 0.6124
   ============================================================
   
4. 推理耗时和 FPS。

   在板端执行下面指令，运行结束后，性能参数会输出在终端。（注意，**your_path**是板端文件系统具体的路径前缀，如果报了`libopencv_world.so.412: cannot open shared object file`的问题，执行`export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"`）

   ```sh
   cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov8s-obb/out
   ./main --model ../model/yolov8s-obb.om  --input ../data/file_list_1.json
   ```

   - --model：om模型路径。
   - --input：指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可,循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值）
   
   SVP_NNN平台上性能结果：
   
   ```
   execution time: 55.28ms, frame rate: 18.09fps
   ```
   
   NNN平台上性能结果：
   
   ```
   execution time: 274.40ms, frame rate: 3.64fps
   ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用 ACL 接口推理计算，模型的性能和精度参考下列数据（以实际运行为准）。

| 芯片型号    | Batch Size | 数据集   | mAP（IoU=0.50） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | DOTAv1   | 76.4    | 18.09 |
| Hi3403V100 NNN | 1          | DOTAv1   | 76.47    | 3.64 |