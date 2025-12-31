# 基于YOLOv8s-OBB网络实现旋转目标检测

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

| 芯片型号  | npu     | soc_version | 环境准备指导  | cann包版本 | 编译工具链 | os  | sdk  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  SPC022  |  aarch64-mix210-linux-gcc |  linux  | SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  linux  | SPC022                                                       |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取 YOLOv8 (Ultralytics) 源码

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip3 install -e .
   cd  ..
   ```

2. 安装依赖。

   ```bash
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。
   该模型通常使用 [DOTAv1数据集](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip) 进行训练和验证。

   在 `samples/samples_GPL/built-in/yolov8s-obb/` 目录下创建 `datasets` 文件夹（或建立软链接），
   将下载的 DOTAv1.zip 拷贝到该目录并执行 `unzip DOTAv1.zip` 进行解压，确保 `data/file_list.json` 中的相对路径可用。

2. 数据切割

   原始的DOTAv1数据都是航拍的大图，分辨率很高，被缩放至1024*1024后会丢失大量的图片细节，所以要先将图片进行切割。执行下面的指令：

   ```sh
   cd samples/samples_GPL/built-in/yolov8s-obb
   python script/split_dota.py --origin_path datasets/DOTAv1 --saved_path datasets/DOTAv1-split
   ```
   
   修改ultralytics/cfg/datasets/DOTAv1.yaml中的path字段为DOTAv1-split。

3. 生成file_list.json

   main函数从file_list.json文件读取输入文件列表进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:
   
   执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    ```bash
    # 在当前项目根目录下执行，生成的文件在data/file_list.json
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```bash
    python3 ../../../../utils/generate_file_list.py datasets/DOTAv1-split/images/val
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。


## 模型转化<a name="section741711594517"></a>

使用 Ultralytics 导出 ONNX，再使用 ATC 工具转为 OM 模型。

1. 导出onnx。

   在 `samples/samples_GPL/built-in/yolov8s-obb` 目录下执行，生成 `model/yolov8s-obb.onnx` 文件。

   ```sh
   mkdir model && cd model
   python ../script/export_onnx.py
   cd ..
   ```

2. 生成模型校准数据。

      选取几张图片生成模型校准数据，引用的图片数据默认在samples/samples_GPL/built-in/yolov8s-obb/data/file_list.json中描述，也可以自定义数据。校准数据文件默认保存在out/preprocess/bin目录下。

      ```sh
      cd samples/samples_GPL/built-in/yolov8s-obb/script/
      python yolov8s_obb_preprocess.py
      ```

      > 注意，file_list.json默认使用 `../datasets` 的相对路径，需确保该路径指向你的DOTAv1切割验证集图片目录。

3. 使用 ATC 工具将 ONNX 模型转 OM 模型。

      Hi3403V100 SVP_NNN 上的 om 模型转换命令

      ```bash
      cd samples/samples_GPL/built-in/yolov8s-obb
      # 需确保 out/preprocess/bin 下有用于量化的校准数据，或去掉 --image_list 参数
      atc --framework=5 --model="model/yolov8s-obb.onnx" --input_shape="images:1,3,1024,1024" --output="model/yolov8s-obb" --soc_version=SS928V100 --image_list="out/preprocess/bin/P0146__1024__0___0.bin" --compile_mode=6
      ```
      
      Hi3403V100 NNN上的 om 模型转换命令
      ```
      atc --framework=5 --model="model/yolov8s-obb.onnx" --input_format="NCHW" --input_shape="images:1,3,1024,1024" --output="./model/yolov8s_obb" --enable_single_stream=true --soc_version=OPTG
      ```
      运行成功后生成 `model/yolov8s-obb.om` 模型文件。

## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```bash
    mkdir -p build
    ```

2. 切换到 build 目录，执行 **cmake** 生成编译文件，soc_version按平台区分，在第一章的推理环境准备中有具体介绍。

      ```bash
      cd build
      cmake ../src -DSOC_VERSION=${soc_version} -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_linux.cmake
      ```
    
3.  执行 **make** 命令，生成的可执行文件 main 在 “./out“ 目录下。

    ```bash
    make
    ```

**步骤2：运行应用。**

1.  将编译好的 `main` 和相关目录（data, model）上传到开发板。
2.  赋予执行权限：`chmod +x main`
3.  修改配置文件 `cfg.txt`（可选，用于控制是否保存 bin 文件）。
4.  运行推理。（your_path是你板端具体的路径前缀）

    ```bash
    cd ${your_path}/modelzoo/samples/samples_GPL/built-in/yolov8s-obb/out
    
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${your_path}/modelzoo/samples/samples_GPL/opensource/opencv/lib"
    
    ./main --model ../model/yolov8s-obb.om  --input ../data/file_list.json
    ```
    图片推理结果会保存在 `out/result/bin` 目录下。
    
    > **注意**：file_list.json中当前只包含三张示例图片，需要按你的数据集路径进行修改。如果需要更多图片，可以在服务器的代码仓的modelzoo/samples/samples_GPL/built-in/yolov8s-obb目录下用`python ../../../../utils/generate_file_list.py datasets/DOTAv1-split/images/val` 生成完整的文件列表，注意核对生成的文件信息在板端文件系统的具体路径。

**步骤3：验证精度和性能**

1. 精度验证。

   在服务器使用 `script/yolov8s_obb_evaluate.py` 进行精度评估。

   ```bash
   python3 script/yolov8s_obb_evaluate.py \
     --result_dir ../out/result/txt \
     --gt_annotations ../datasets/DOTAv1-split/labels/val \
     --img_dir ../datasets/DOTAv1-split/images/val
   ```
   
   **关键说明**：
   
   - 脚本会自动根据 `result_dir` 中存在的 txt 文件过滤待评估图片
   - 多进程数量可在脚本中修改 `NUM_WORKERS` 变量（默认 6）

   精度结果示例如下：
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
   
2. 推理耗时和 FPS。

   板端运行完demo后，会在串口或者ssh最后一行打印推理耗时和 FPS。

   ```bash
   execution time: 55.28ms, frame rate: 18.09fps
   ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用 ACL 接口推理计算，模型的性能和精度参考下列数据（以实际运行为准）。

| 芯片型号    | Batch Size | 数据集   | mAP（IoU=0.50） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | DOTAv1   | 76.4    | 18.09 |
| Hi3403V100 NNN | 1          | DOTAv1   | 76.47    | 3.64 |