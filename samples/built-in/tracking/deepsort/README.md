# 基于DeepSORT实现多目标跟踪

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

DeepSORT（Deep Simple Online and Realtime Tracking）是一种经典的多目标追踪算法，在原始 SORT 算法基础上引入了深度外观特征（ReID），有效降低了 ID Switch 率。该模型在 MOT16 数据集上进行了验证。

- 参考实现：

  ```
  https://github.com/ZQPei/deep_sort_pytorch
  ```

## 输入输出数据<a name="section540883920406"></a>
### 检测网络

- 输入数据


  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 640 x 640 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | feature_map_1  | FP32     |  1 x 3 x 80 x 80 x 85 | NCHW         |
  | feature_map_2  | FP32     |  1 x 3 x 40 x 40 x 85 | NCHW         |
  | feature_map_3  | FP32     |  1 x 3 x 20 x 20 x 85 | NCHW         |
### ReID
- 输入数据
  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images  | RGB_FP32 | 1 x 3 x 128 x 64 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | output0  | FP32     | 1 x 512 | NCHW |

  
  > **注意**：输出大小取决于模型导出时的配置。

## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data                       // 运行时模型参数配置文件
│   └── deepsort.json
├── deepsort_py.patch          // pyhton版本补丁
├── LICENSE                    //模型LICENSE   
├── out                        // 编译输出结果  
├── README.md                   
├── model                      // 模型文件
├── requirements.txt            // python依赖
├── script
│   ├── evaluate_mot.py       // 精度评估脚本
│   ├── export_onnx.py        // ONNX导出脚本 
│   └── generate_real_bin.py    // 生成模型校准数据脚本
└── src
    ├── CMakeLists.txt      // 编译脚本
    ├── deepsort.cpp        // deepsort帧处理主流程
    ├── deepsort.h
    ├── hungarian.cpp       // 匈牙利算法
    ├── hungarian.h
    ├── kalmanfilter.cpp     // 卡尔曼滤波
    ├── kalmanfilter.h
    ├── main.cpp            // main函数入口
    ├── nn_matching.cpp     // 特征匹配和度量学习的核心类
    ├── nn_matching.h
    ├── track.cpp           // 单个轨迹
    ├── tracker.cpp         // 轨迹管理
    ├── tracker.h
    └── track.h
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
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b140  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j6v87p1oi000)上进行下载。

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

4. 切换到可执行文件main所在的目录，运行可执行文件。测试图片上模型推理命令参考：
    
    ```
    ./main ../model/yolov5s.om ../model/reid_net.om ../datasets/MOT16/train/MOT16-09/img1
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用Python 3.8.20
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

   该模型通常使用 [MOT16数据集](https://motchallenge.net/data/MOT16/) 进行训练和验证。

   在 `samples/built-in/tracking/deepsort/` 目录下创建 `datasets` 文件夹（或建立软链接），将下载的数据集拷贝到该目录并进行解压，解压结果如下：

   ```
   datasets
   └── MOT16
    ├── test
    │   ├── MOT16-01
    │   ├── MOT16-03
    │   ├── MOT16-06
    │   ├── MOT16-07
    │   ├── MOT16-08
    │   ├── MOT16-12
    │   └── MOT16-14
    └── train
        ├── MOT16-02
        ├── MOT16-04
        ├── MOT16-05
        ├── MOT16-09
        ├── MOT16-10
        ├── MOT16-11
        └── MOT16-13
   ```



## 模型转化<a name="section741711594517"></a>

使用 `export_onnx.py` 导出 ONNX，再使用 ATC 工具转为 OM 模型。

1. 获取 DeepSort 源码

   ```bash
    git clone https://github.com/ZQPei/deep_sort_pytorch.git
    git reset 4f910afc16860ff05c6be408b120a49524bd4f68
    cd deep_sort_pytorch
    patch -p1 < ../deepsort_py.patch
    cd ..
   ```

2. 导出onnx。
   下载权重文件：
   YOLO: https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
   下载后归档在`deep_sort_pytorch/detector/YOLOv5`
   ReID: https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6
   下载后归档在`deep_sort_pytorch/deep_sort/deep/checkpoint`

   在 `samples/built-in/tracking/deepsort/deep_sort_pytorch` 目录下执行，生成 `model/*.onnx` 文件。

   ```sh
   cd samples/built-in/tracking/deepsort/deep_sort_pytorch
   cp ../script/export_onnx.py .
   python export_onnx.py --output_dir ../model
   ```

3. 生成模型校准数据。
   
      选取几张图片生成模型校准数据，引用的图片数据默认在 datasets/MOT16/train。校准数据文件默认保存在datasets/prep_data_aipp目录下。

      ```sh
      cd samples/built-in/tracking/deepsort/script
      python generate_real_bin.py
      ```

4. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      Hi3403V100 SVP_NNN 上的 om 模型转换命令

      ```bash
      cd samples/built-in/tracking/deepsort
      atc --framework=5 --model=model/yolov5s.onnx --input_shape=images:1,3,640,640 --output=model/yolov5s --soc_version=SS928V100 --image_list=./datasets/prep_data_aipp/yolo_real.bin

      atc --framework=5 --model=model/reid_net.onnx --input_shape=input:1,3,128,64 --output=model/reid_net --soc_version=SS928V100 --image_list=./datasets/prep_data_aipp/reid_real.bin
      ```
      
      Hi3403V100 NNN上的 om 模型转换命令
      ```bash
      cd samples/built-in/tracking/deepsort
      atc --framework=5 --model=model/yolov5s.onnx --input_shape=images:1,3,640,640 --output=model/yolov5s --enable_single_stream=true --soc_version=OPTG 

      atc --framework=5 --model=model/reid_net.onnx --input_shape=input:1,3,128,64 --output=model/reid_net --enable_single_stream=true --soc_version=OPTG 
      ```
      运行成功后生成 `model/yolov5s.om` 和 `model/reid_net.om` 模型文件。

## 精度&性能评估<a name="section741711594518"></a>

1.  修改配置文件 `data/deepsort.json`（可选用于调整置信度等参数）。
2.  运行推理。（**your_path**是板端文件系统具体的路径前缀）

    ```bash
    cd ${your_path}/modelzoo/samples/built-in/tracking/deepsort/out
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"
    for folder in MOT16-02 MOT16-04 MOT16-05 MOT16-09 MOT16-10 MOT16-11 MOT16-13; do
        ./main ../model/yolov5s.om ../model/reid_net.om ../datasets/MOT16/train/$folder/img1
    done
    ```
    图片推理结果会保存在 `out` 目录下。
    
    > **注意**：图片路径下图片文件名需要按照时间升序排列。


3. 精度验证。

   在服务器使用 `script/evaluate_mot.py` 进行精度评估。

   ```bash
    # 指定结果目录和 GT 目录
    python evaluate_mot.py --ts_dir ${your_path}/modelzoo/samples/built-in/tracking/deepsort/out --gt_dir ${your_path}/modelzoo/samples/built-in/tracking/deepsort/datasets/MOT16/train
    # 只评估单个序列
    python evaluate_mot.py --gt /path/to/gt.txt --ts /path/to/result.txt
   ```
  

   Hi3403V100 SVP_NNN平台上精度结果：
   
   ```
              IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN  IDs    FM  MOTA  MOTP  IDt  IDa IDm
    MOT16-02 12.0% 24.3%  8.0% 25.4% 77.2%  54  5  18  31 1341 13295  496   499 15.1% 0.238  204  289  20
    MOT16-04 22.2% 35.8% 16.1% 40.9% 90.8%  83  7  39  37 1965 28111 1167  1229 34.3% 0.208  417  711  25
    MOT16-05 18.1% 21.5% 15.7% 57.8% 79.1% 125 22  81  22 1043  2879  470   390 35.6% 0.252  332  188  75
    MOT16-09 29.8% 32.3% 27.6% 64.2% 75.0%  25  5  18   2 1125  1882  305   284 37.0% 0.209   75  211   4
    MOT16-10 23.1% 32.6% 17.9% 44.3% 80.9%  54  7  24  23 1285  6860  509   598 29.7% 0.247  121  363  15
    MOT16-11 30.9% 36.8% 26.6% 59.0% 81.5%  69 11  32  26 1232  3761  436   399 40.8% 0.171   69  333  10
    MOT16-13 12.2% 23.1%  8.3% 30.4% 84.7% 107  6  48  53  629  7966  521   552 20.4% 0.260  195  340  36
    OVERALL  20.9% 31.7% 15.6% 41.3% 84.1% 517 63 260 194 8620 64754 3904  3951 30.0% 0.219 1413 2435 185
   ```
   
   Hi3403V100 NNN平台上精度结果：
   
   ```
              IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN  IDs    FM  MOTA  MOTP  IDt  IDa IDm
    MOT16-02 12.7% 25.6%  8.4% 25.3% 76.9%  54  5  19  30 1357 13323  489   512 14.9% 0.236  174  308  18
    MOT16-04 25.5% 41.1% 18.5% 40.8% 90.8%  83  6  39  38 1975 28163 1188  1210 34.1% 0.206  352  783  21
    MOT16-05 21.1% 25.0% 18.3% 57.9% 79.1% 125 20  84  21 1046  2870  455   402 35.9% 0.249  287  223  73
    MOT16-09 29.2% 31.6% 27.1% 64.6% 75.4%  25  6  17   2 1109  1862  301   290 37.8% 0.214   76  204   6
    MOT16-10 30.9% 44.6% 23.7% 43.5% 82.0%  54  7  22  25 1177  6956  499   604 29.9% 0.247   93  385  15
    MOT16-11 33.5% 40.0% 28.7% 59.1% 82.3%  69  9  34  26 1165  3753  457   431 41.4% 0.171   62  357   8
    MOT16-13 13.0% 25.2%  8.7% 29.5% 85.3% 107  4  49  54  581  8069  563   594 19.5% 0.258  163  395  31
    OVERALL  23.8% 36.3% 17.7% 41.1% 84.4% 517 57 264 196 8410 64996 3952  4043 29.9% 0.218 1207 2655 172
    ```
   
4. 推理耗时和 FPS(ReID)。
  将data/deepsort.json的TEST_REID_FPS字段改成true，开启Reid fps测试功能

   在板端执行下面指令，运行结束后，性能参数会输出在终端。（注意，**your_path**是板端文件系统具体的路径前缀，如果报了`libopencv_world.so.412: cannot open shared object file`的问题，执行`export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"`）

   ```sh
    cd ${your_path}/modelzoo/samples/built-in/tracking/deepsort/out
    ./main ../model/yolov5s.om ../model/reid_net.om ../datasets/MOT16/train/MOT16-09/img1 ../data/deepsort.json
   ```
   
   Hi3403V100 SVP_NNN平台上性能结果：
   
   ```
   execution time: 1.44ms, frame rate: 693.28fps
   ```
   
   Hi3403V100 NNN平台上性能结果：
   
   ```
   execution time: 14.19ms, frame rate: 70.49fps
   ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用 ACL 接口推理计算，模型的性能和精度参考下列数据（以实际运行为准）。

| 芯片型号    | Batch Size | 数据集   | MOTA | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | MOT16   | 30.0%    | 693.28 |
| Hi3403V100 NNN | 1          | MOT16   | 29.9%    | 70.49 |