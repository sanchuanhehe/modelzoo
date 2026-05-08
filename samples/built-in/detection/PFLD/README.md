# 基于**PFLD**网络实现人脸关键点检测
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

PFLD 是一种基于轻量级卷积神经网络的实时人脸关键点检测模型。相比传统复杂模型或分阶段检测方法，它通过单阶段端到端架构直接输出人脸98个（或更多）关键点坐标，采用多尺度特征融合与自适应损失函数优化定位精度，在保证毫米级检测误差的同时，兼顾极快的推理速度（毫秒级），适用于人脸对齐、表情识别、姿态估计、美颜美妆等实时交互场景。
- 参考实现：

  ```
  https://github.com/polarisZhao/PFLD-pytorch.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | NPU | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN    | input_1    | BGR_FP32 | 1 x 3 x 112 x 112 | NCHW         |
  | NNN    | input_1    | BGR_FP16 | 1 x 3 x 112 x 112 | NCHW         |

- 输出数据

  | NPU | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | -------- | ------ |
  | SVP_NNN | output_1 | FP32     | 1x64x28x28 |
  | SVP_NNN      | 405      | FP32     | 1x196      |
  | NNN | output_1 | FP16     | 1x64x28x28 |
  | NNN      | 405      | FP16     | 1x196      |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess        //数据预处理结果
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── pfld_gen_quant_bin.py      //为SVP_NNN核转OM模型生成所需的量化数据
│   ├── pfld_preprocess.py         //数据预处理脚本
│   ├── pfld_infer.py              //模型推理脚本
│   ├── pfld_postprocess.py        //数据后处理脚本
│   ├── pfld_evaluate.py           //精度评估脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── pfld-sim.onnx     //转换出来的onnx模型文件
│   ├── pfld.om	          //转换出来的om模型文件

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

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9j1qkj1ec00)上进行下载。

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
    ./main --acl ../src/acl.json --model ../model/pfld.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用python3.7 or 3.8
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[WFLW](https://wywu.github.io/projects/LAB/WFLW.html)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集modelzoo/datesets/WFLW路径下。数据集目录结构如下所示：
   
   ```
    WFLW/  # 数据集根目录
    ├─ WFLW_images/                          # 原始图像核心存储文件夹，包含所有10000张人脸图像（7500训练+2500测试）
    │  ├─ 0--Parade/                         # 子场景文件夹（官方细分60个真实场景之一，如“游行”场景）
    │  │  ├─ 0_Parade_0_100.jpg              # 图像文件示例：JPG格式，无语义化命名（仅作唯一标识）
    │  │  └─ ...（该场景下所有相关人脸图像）
    │  ├─ 1--Handshaking/                    # 子场景文件夹（如“握手”场景），所有子场景覆盖真实生活场景（会议、演讲、群体等）
    │  │  └─ ...（对应场景图像）
    │  └─ ...                                # 共60个子场景文件夹，每个文件夹下图像尺寸不固定，多为400×600左右中等分辨率，单张图含1个人脸，覆盖大姿态、遮挡、模糊等挑战场景
    └─ WFLW_annotations/                     # 标注总文件夹，整合关键点、边界框、属性、场景划分等所有标注信息
       ├─ list_98pt_rect_attr_train_test/    # 训练集+完整测试集的“一站式整合标注”（模型训练/整体测试核心依赖）
       │  ├─ list_98pt_rect_attr_train.txt   # 训练集标注文件（7500个样本，每行对应1张训练图，含完整标注）
       │  │                                  # 数据格式（每行共211个字段，空格分隔）：
       │  │                                  # [图像相对路径] [x1 y1 x2 y2 ... x98 y98] [rect_x1 rect_y1 rect_x2 rect_y2] [attr1 attr2 ... attr10]
       │  │                                  # 字段拆解：
       │  │                                  # 1. 第1字段：图像相对路径（如“0--Parade/0_Parade_0_100.jpg”），拼接“WFLW/WFLW_images/”得到完整路径
       │  │                                  # 2. 第2-197字段：98个人脸关键点坐标（x1 y1为第1个关键点，x2 y2为第2个...x98 y98为第98个），像素级绝对坐标
       │  │                                  # 3. 第198-201字段：人脸边界框坐标（rect_x1/rect_y1=左上角，rect_x2/rect_y2=右下角），用于快速裁剪人脸
       │  │                                  # 4. 第202-211字段：10个人脸属性标签（0=无，1=有），依次为：戴眼镜、戴口罩、戴帽子、有妆容、有胡须、有遮挡、模糊、大姿态、表情夸张、光照异常
       │  └─ list_98pt_rect_attr_test.txt    # 完整测试集标注文件（2500个样本，格式与训练集完全一致，无场景细分，用于模型整体精度评估）
       ├─ list_98pt_test/                    # 细分场景测试集标注（用于评估模型在特定挑战场景下的鲁棒性）
       │  ├─ list_98pt_test.txt              # 完整测试集简化标注（仅含“图像相对路径+98个关键点坐标”，无边界框和属性，备用标注文件），数据格式（每行共197个字段）：[图像相对路径] [x1 y1 x2 y2 ... x98 y98]
       │  ├─ list_98pt_blur_test.txt         # 细分场景：模糊测试集（从2500测试集中筛选的图像模糊样本）
       │  ├─ list_98pt_occlusion_test.txt    # 细分场景：遮挡测试集（人脸被口罩、头发、手部等遮挡的样本）
       │  ├─ list_98pt_largepose_test.txt    # 细分场景：大姿态测试集（俯仰角/偏航角/滚转角较大的样本，如侧脸、抬头、低头）
       │  ├─ list_98pt_expression_test.txt   # 细分场景：表情变化测试集（夸张表情样本，如大笑、皱眉、惊讶）
       │  ├─ list_98pt_illumination_test.txt # 细分场景：光照变化测试集（极端光照样本，如过亮、过暗、逆光、侧光）
       │  └─ list_98pt_makeup_test.txt       # 细分场景：带妆容测试集（人脸含浓妆的样本）
       └─ Mirror98.txt                       # 关键点镜像映射表（用于数据增强，水平翻转图像时同步调整关键点坐标）
                                             # 数据格式：共98行，每行1个整数（范围1-98），第i行的整数表示“第i个关键点水平翻转后对应的原始关键点索引”
                                             # 示例：第1行是“98”→ 第1个关键点翻转后对应第98个关键点，第2行是“97”→ 第2个关键点翻转后对应第97个（对称关键点互换）
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
  
    执行源码中SetPreparation.py脚本，完成数据预处理。（原始数据集中的图片有多张人脸，只有1个是有标注的待检测人脸，需要将其裁剪出来，裁剪出来的单个人脸的图片会保存到PFLD/data/wflw_test_cropped目录下。）
    
    ```bash
    cd PFLD-pytorch/data/
    python3 SetPreparation.py
    cd ../../
    ```
   
3. 生成量化校准数据

    ```bash
    # 生成量化校准数据（SVP_NNN核转换OM模型需要提供量化校准数据）
    cd script
    python3 pfld_gen_quant_bin.py
    cd ..
    ```

4. 生成file_list.json

   main函数从file_list.json文件读取输入文件列表进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:

   ```
   data
      ├── file_list.json
      ├── file_list_1.json（单张图片输入，但是多了loop参数，适合main函数循环推理测试性能）
   ```
   
   执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    ```bash
    # 在当前项目根目录下执行，生成的文件在data/file_list.json
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```bash
    python3 ../../../../utils/generate_file_list.py ../../../../samples/built-in/detection/PFLD/data/wflw_test_cropped/imgs/
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。
   

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取开源源码和权重文件。
   ```bash
   git clone https://github.com/polarisZhao/PFLD-pytorch.git
   cd PFLD-pytorch
   git checkout b00750c37deaf5d133933d080d4b7a6421826812
   # 更改数据集处理的输入输出路径
   git apply ../pfld.patch
   cd ..
   ```

   权重默认在checkpoint目录下，已经跟随源码下载

2. 导出onnx文件。

    1. 使用PFLD-pytorch源码中pytorch2onnx.py导出onnx文件。

         ```bash
         cd PFLD-pytorch
         python ./pytorch2onnx.py --onnx_model_sim="../model/pfld-sim.onnx"
         cd ..
         ```
         
3. 使用ATC工具将ONNX模型转OM模型。
      在当前模型的代码根目录下，执行ATC命令。
      1. Hi3403V100 SVP_NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/pfld-sim.onnx" --input_shape="input_1:1,3,112,112" --output="model/pfld" --image_list="data/quant/quant.txt" --compile_mode=5 --soc_version=SS928V100
         ```
      2. Hi3403V100 NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/pfld-sim.onnx" --input_shape="input_1:1,3,112,112" --output="model/pfld" --enable_single_stream=true --input_fp16_nodes="input_1" --output_type=FP16 --soc_version=OPTG
         ```
         运行成功后生成pfld.om模型文件。
      
      参数说明：
      
        - --framework：5代表ONNX模型。
        - --model：为ONNX模型文件。
        - --input_shape：输入数据的shape。
        - --output：输出的OM模型。
        - --image_list: 量化校准数据。数据获取参考[准备数据集](#section183221994411)章节。
        - --enable_small_channel:使能small channel优化。
        - --enable_single_stream:推理时使用一条stream。
        - --compile_mode：量化方式。5：数据和权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算子使用fp16格式。6：数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算子使用fp16格式。
        - --soc_version：处理器型号。
        
        注意：请参考环境准备章节配置ATC环境变量。

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

2. 切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/pfld.om  --input ../data/file_list.json
    ```
    每张图片的预测结果会保存在out/result/txt目录下，模型推理的原始bin文件默认会保存在out/result/bin目录下。

3. 精度验证。

   使用pfld_evaluate.py将模型推理的结果与数据集中的crow.mat标签文件进行对比，获取评估结果。

   ```bash
   cd script # 切换到script目录
   # 更改下开发板输出文件的权限，确保脚本能访问
   sudo chmod -R 777 ../out
   python3 pfld_evaluate.py \
      --pred_txt_dir "../out/result/txt" \
      --gt_list_file "../data/wflw_test_cropped/list.txt" \
      --output_file "../out/result/metrics_results.txt" 
   cd ..
   ```

   参数说明：

   - --pred_txt_dir：预测关键点txt文件目录。
   - --gt_list_file：真实标注文件路径。（SetPreparation.py脚本进行数据集预处理的时候会生成此标签文件）
   - --output_file：评估结果保存文件。

   SVP_NNN平台上精度结果：
   ```bash
    平均NME: 0.0720
    AUC@0.1: 0.3823
    失败率: 0.1704
   ```
   NNN平台上精度结果：
   ```bash
    平均NME: 0.0696
    AUC@0.1: 0.3973
    失败率: 0.1580
   ```
   
4. 验证om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/pfld.om --input ../data/file_list_1.json
    ```

    参数说明：(此模式下，file_list_1.txt只放一张图片)

    - --model：om模型路径。
    - --output:  后处理后结果所在位置
    - --model: 模型所在位置
    - --loop：循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

    在板端会输出显示，SVP_NNN平台上性能结果如下：
    ```
    execution time: 0.62ms, frame rate: 1624.65fps
    ```
    在板端会输出显示，NNN平台上性能结果如下：
    ```
    execution time: 1.63ms, frame rate: 611.95fps
    ```

**（可选步骤）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 pfld_preprocess.py \
      --file_list "../data/file_list.json" \
      --output_dir "../data/preprocess/bin"
   ```

   参数说明：

   - --file_list：输入的 JSON 文件路径，内含待处理图片的路径列表（格式需符合脚本预期：包含 fileList 字段，每个元素为图片路径）。
   - --output_dir：预处理后生成的 .bin 文件保存目录。若目录不存在，脚本会自动创建。


2. 使用python脚本进行模型推理（可以跟CPP版本的推理结果进行对比）
   ```bash
   cd script # 切换到script目录
   python3 pfld_infer.py \
      --infer_engine "pytorch" \
      --preprocess_bin_dir "../data/preprocess/bin" \
      --file_list_path "../data/preprocess/file_list.txt" \
      --infer_bin_dir "../out/result_pc/bin" \
      --pytorch_model_path "../PFLD-pytorch/checkpoint/snapshot/checkpoint.pth.tar"
   ```

   参数说明：

   - --infer_engine: 推理引擎选择：onnx 或 pytorch。
   
   - --preprocess_bin_dir: 预处理后生成的二进制（.bin）文件存放目录。

   - --infer_bin_dir：推理结果（.bin 格式）的输出目录。若目录不存在，脚本会自动创建。

   - --file_list_path：python预处理阶段生成的文件列表（记录所有待推理的 .bin 文件路径）。脚本会从该文件读取待处理文件路径。

   - --pytorch_model_path：pytorch模型权重文件路径（仅infer_engine=pytorch时有效）。
   
   - --onnx_model_path：ONNX模型文件路径（仅infer_engine=onnx时有效）。

3. 使用python脚本进行模型推理结果的后处理
   ```bash
   cd script # 切换到script目录
   python3 pfld_postprocess.py \
      --bin_dir "../out/result_pc/bin" \
      --output_dir "../out/result_pc/txt"
   ```

   参数说明：

   - --bin_dir: 存放模型推理得到的bin文件的目录。

   - --output_txt_dir：存放后处理结果txt文件的目录。



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，PFLD模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标1（NME） | 精度指标2（AUC @ 0.1 failureThreshold） | 精度指标3（失败率） | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | WFLW | 0.0720       | 0.3823       | 0.1704       | 1624.65  |
| Hi3403V100 NNN | 1          | WFLW | 0.0696       | 0.3973       | 0.1580       | 611.95  |