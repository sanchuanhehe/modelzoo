# 基于FaceNet实现人脸识别
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

FaceNet 是一种基于深度卷积神经网络的端到端人脸识别与特征嵌入模型。相比传统基于手工特征或分阶段匹配的方法，它通过将人脸图像直接映射为固定维度的紧凑特征向量（Embedding），并采用三元组损失（Triplet Loss）优化特征相似度度量，能有效缩小类内差异、扩大类间距离，兼顾识别精度与推理效率，适用于身份验证、人脸检索、监控安防等大规模人脸识别场景。

- 参考实现：

  ```bash
  https://github.com/timesler/facenet-pytorch
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | NPU | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN    | input    | BGR_FP32 | 1 x 3 x 160 x 160 | NCHW         |
  | NNN    | input    | BGR_FP16 | 1 x 3 x 160 x 160 | NCHW         |

- 输出数据

  | NPU | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | -------- | ------ |
  | SVP_NNN | output | FP32     | 1x512 |
  | NNN | output | FP16     | 1x512 |


## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess/       //数据预处理结果
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── facenet_prepare_dataset.py    //将原始的LFW数据集进行人脸检测和裁剪的脚本
│   ├── facenet_pth2onnx.py           //pytorch模型转onnx模型脚本
│   ├── facenet_preprocess.py         //数据预处理脚本
│   ├── facenet_infer.py              //模型推理脚本
│   ├── facenet_postprocess.py        //数据后处理脚本
│   ├── facenet_evaluate.py           //精度评估脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── 20180402-114759-vggface2.pt	      //facenet原始pytorch模型文件
│   ├── facenet_vggface2_static.onnx	  //转换出来的onnx模型文件
│   ├── facenet_vggface2.om	              //转换出来的om模型文件

├── out
│   ├── result            //模型推理结果
│   ├── main              //main.cpp编译出来的可执行文件

├── CMakeLists.txt      //编译脚本，调用src目录下CMakeLists文件
├── requirements.txt    //python依赖包
├── *.json			    //模型信息
├── LICENSE			    //模型LICENSE
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

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=iapedo790s00)上进行下载。

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
    ./main --acl ../src/acl.json --model ../model/facenet_vggface2.om  --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用 python3.8
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   LFW（Labeled Faces in the Wild）是一个经典的无约束人脸识别基准数据集，包含 5749 个不同身份的 13233 张真实场景人脸图像，涵盖姿态、光照、表情等自然变异，适用于评估人脸识别算法在非受控环境下的泛化性能与匹配精度。
   
   点击 [LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 下载数据集，点击 [pairs.txt](http://vis-www.cs.umass.edu/lfw/pairs.txt) 下载pairs.txt进行精度评估。
   在`modelzoo/datasets`源码目录下创建`LFW`文件夹，文件结构如下：
   
   ```
   LFW/                                        # 数据集根目录
   ├── lfw/                                    # 核心图像存储目录（与根目录同名，解压后直接可见）
   │   ├── Abdul_Qadeer_Khan/                  # 单个身份文件夹（以人物姓名命名，空格替换为下划线）
   │   │   ├── Abdul_Qadeer_Khan_0001.jpg      # 该身份的人脸图像（命名规则：姓名_四位序号.jpg）
   │   │   ├── Abdul_Qadeer_Khan_0002.jpg
   │   │   └── ...（同身份多张图像，序号递增）
   │   ├── Aaron_Eckhart/
   │   ├── Aaron_Guiel/
   │   └── ...（共5749个身份文件夹，对应所有数据集人物）
   ├── pairs.txt                               # 标准测试对列表（用于人脸验证任务，含"3000相同身份对"和"3000不同身份对"）
   ├── lfw.tgz                                 # 原始压缩包（解压后可删除）
   ```

2. 将原始的LFW数据集进行人脸检测和裁剪

   ```bash
   cd script # 切换到script目录
   # 裁剪后的人脸保存在FaceNet/data/preprocess/lfw_cropped目录下
   python3 facenet_prepare_dataset.py \
      --raw_dir "../../../../../datasets/LFW/lfw" \
      --cropped_dir "../data/preprocess/lfw_cropped"
   cd ..
   ```

   参数说明：
   - --raw_dir: 原始LFW数据集根目录（包含子文件夹）。
   - --cropped_dir: 裁剪后人脸保存目录。


3. 生成file_list.json

   main函数从file_list.json文件读取输入文件列表进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:

   ```
   data
      ├── file_list.json
      ├── file_list_1.json（单张图片输入，但是多了loop参数，适合main函数循环推理测试性能）
      ├── cfg.txt（配置文件）
   ```
   
   执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理。
    ```bash
    # 在当前项目根目录下执行，生成的文件在data/file_list.json
    python3 ../../../../utils/generate_file_list.py -r ${dataset_path}
    ```
    例如:
    ```bash
    python3 ../../../../utils/generate_file_list.py -r data/preprocess/lfw_cropped
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。
   - --r：开启递归查找。

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取参考代码仓源码
 
   ```bash
   # 此处clone主要是用来参考学习，后续的python脚本使用的是直接pip安装的facenet-pytorch包
   git clone https://github.com/timesler/facenet-pytorch.git
   cd facenet-pytorch
   git checkout 555aa4bec20ca3e7c2ead14e7e39d5bbce203e4b
   cd ..
   ```

2. 获取权重文件。

   点击[facenet](https://github.com/timesler/facenet-pytorch) 进入facenet开源首页，下载模型权重文件20180402-114759-vggface2.pt，或者点击[链接](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt) 直接下载。

   下载模型文件20180402-114759-vggface2.pt，将文件放到~/.cache/torch/checkpoints/路径下

   ```bash
    # 记录当前所在目录
    original_path=$(pwd)
    mkdir -p ~/.cache/torch/checkpoints/
    cd ~/.cache/torch/checkpoints/
    wget https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
    # 返回之前的目录
    cd "$original_path"
   ```

2. 生成onnx文件。

      使用script/facenet_pth2onnx.py将pytorch模型转换成ONNX模型；

      ```bash
      cd script
      python3 facenet_pth2onnx.py --output_onnx_path ../model/facenet_vggface2_static.onnx
      cd ..
      ```

3. 使用ATC工具将ONNX模型转OM模型。

      在当前模型的代码根目录下，执行ATC命令。
      1. Hi3403V100 SVP_NNN上的om模型转换命令
         ```bash
         # 如果您没有现成的bin文件，您需要参考后面的python脚本使用说明，使用preprocess.py脚本先生成bin文件。如果您想使用多个bin文件进行数据校准，多个文件间请使用;分割，例如a.bin;b.bin。
         atc --framework=5 --model="model/facenet_vggface2_static.onnx" --input_shape="input:1,3,160,160" --input_type="input:FP32" --output="model/facenet_vggface2" --image_list="data/preprocess/bin/Derek_Lowe_0001.bin" --soc_version=SS928V100 --compile_mode=6
         ```
      2. Hi3403V100 NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/facenet_vggface2_static.onnx" --input_shape="input:1,3,160,160" --output="model/facenet_vggface2" --enable_single_stream=true --input_fp16_nodes="input" --output_type=FP16 --soc_version=OPTG
         ```
         运行成功后生成facenet_vggface2.om模型文件。
    
         参数说明：
   
         - --framework：原始框架类型，5代表ONNX模型。
         - --model：ONNX模型文件路径。
         - --input_shape：输入数据的shape。
         - --output：输出的OM模型路径。
         - --image_list：转换模型生成量化参数时用的校准数据。
         - --enable_single_stream：推理时使用一条stream。
         - --input_fp16_nodes：输入数据使用FP16格式。
         - --soc_version：处理器型号。
         - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：SVP_NNN上选取其他编译模式可能导致精度下降
      
         注意：如果出现atc命令找不到，参考推理环境准备。


## 精度&性能评估<a name="section741711594518"></a>

1. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数。
    ```bash
    # 预处理后的二进制文件，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_preprocess_bin="../data/preprocess/bin"
    
    # 模型推理结果保存目录
    save_result_txt="../out/result/txt"
    
    # 模型推理原始二进制结果，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_result_bin="../out/result/bin"
    ```

2. 登录到板端运行环境，切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/facenet_vggface2.om  --input ../data/file_list.json
    ```
    每张图片的预测结果会保存在out/result/txt目录下，模型推理的原始bin文件默认会保存在out/result/bin目录下。

3. 精度验证。

   使用facenet_evaluate.py将模型推理的结果与数据集中的标签文件进行对比，获取评估结果。

   ```bash
   cd script # 切换到script目录
   # 更改下开发板输出文件的权限，确保脚本能访问
   sudo chmod -R 777 ../out
   python3 facenet_evaluate.py \
      --pairs_path "../../../../../datasets/LFW/pairs.txt" \
      --txt_dir "../out/result/txt"
   cd ..
   ```

   参数说明：

   - --pairs_path：LFW配对文件路径。
   - --txt_dir：特征向量txt文件目录。

   SVP_NNN平台上精度结果：
   ```bash
    各折准确率: [0.98833333 0.99333333 0.97833333 0.98       0.97333333 0.99333333
               0.97666667 0.99166667 0.98666667 0.99333333]
    平均准确率: 0.9855 ± 0.0073
   ```
   NNN平台上精度结果：
   ```bash
    各折准确率: [0.995      0.99166667 0.99       0.99       0.98833333 0.99666667
               0.98666667 0.995      0.99666667 0.99666667]
    平均准确率: 0.9927 ± 0.0036
   ```

4. 验证om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/facenet_vggface2.om  --input ../data/file_list_1.json
   cd ..
   ```

   参数说明：(此模式下，输入路径为一张图片)
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 4.07ms, frame rate: 245.73fps
   ```
   在板端会输出显示，NNN平台上性能结果如下：
   ```bash
   execution time: 6.27ms, frame rate: 159.41fps
   ```

**（可选步骤）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 facenet_preprocess.py \
      --file_list "../data/file_list.json" \
      --output_dir "../data/preprocess/bin"
   cd ..
   ```

   参数说明：
   - --file_list：输入的 JSON 文件路径，内含待处理图片的路径列表（格式需符合脚本预期：包含 fileList 字段，每个元素为图片路径）。脚本会从该文件读取所有待转换的图片路径。
   - --output_dir：预处理后生成的 .bin 文件保存目录。若目录不存在，脚本会自动创建。

2. 使用python脚本进行模型推理（可以跟CPP版本的推理结果进行对比）
   ```bash
   cd script # 切换到script目录
   python3 facenet_infer.py \
      --file_list_path "../data/preprocess/bin_file_list.txt" \
      --output_bin_dir "../out/result/bin"
   cd ..
   ```

   参数说明：
   - --file_list_path: 预处理生成的bin文件列表路径。
   - --output_bin_dir: 推理结果bin文件输出目录。

3. 使用python脚本进行模型推理结果的后处理
   ```bash
   cd script # 切换到script目录
   python3 facenet_postprocess.py \
      --bin_dir "../out/result/bin" \
      --output_txt_dir "../out/result/txt"
   cd ..
   ```

   参数说明：

   - --bin_dir: 存放推理输出bin文件的目录。
   - --output_txt_dir：特征向量txt文件输出目录。
   

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 平均准确率（10折交叉验证） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | LFW  | 0.9855      | 245.73      |
| Hi3403V100 NNN | 1          | LFW  | 0.9927      | 159.41      |

