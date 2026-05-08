# 基于CrowdCount实现人群计数
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

CrowdCount是一种基于多尺度卷积神经网络（MSCNN）的高精度人群计数模型。相比传统多列 / 多网络方法，它通过单列网络中的多尺度特征块（MSB）与尺度自适应密度图回归技术，能有效应对透视畸变导致的人物尺度差异问题，兼顾计数精度与模型轻量化，适用于监控图像、公共场所等密集人群计数场景。

- 参考实现：

  ```bash
  https://github.com/zzubqh/CrowdCount
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | input  | RGB_FP32 | 1 x 3 x 224 x 224 | NCHW        |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | model_1  | FP32     | 1 x 1 x 56 x 56 | NCHW |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess        //数据预处理结果
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── crowdcount_tf2onnx.py            //TensorFlow模型转onnx模型脚本
│   ├── crowdcount_preprocess.py         //数据预处理脚本
│   ├── crowdcount_infer.py              //模型推理脚本
│   ├── crowdcount_postprocess.py        //数据后处理脚本
│   ├── crowdcount_evaluate.py           //精度评估脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── mscnn_model_weights.h5	      //原始TensorFlow模型文件
│   ├── mscnn_model.onnx	          //转换出来的onnx模型文件
│   ├── mscnn_model.om	              //转换出来的om模型文件

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

  **表 1** 版本配套表                                                   |  -                                                            |

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

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9i4tr9hec00)上进行下载。

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
    ./main --acl ../src/acl.json --model ../model/mscnn_model.om  --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 源作者使用的是python3.6，建议使用相同版本创建虚拟环境
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   CrowDensity为代码开源作者提供的标注好的人头位置信息的数据集，数据集可以通过作者提供的百度网盘地址下载。
   
   点击 [CrowDensity](https://pan.baidu.com/s/1T5EfBovMnpe4meIYcXSa8w) 下载数据集进行精度评估，在`modelzoo/datasets`源码目录下创建`CrowDensity`文件夹，文件结构如下：
   
   ```
   CrowDensity
      ├── Crowdata（标注好的人头位置信息的数据集）
         ├── img（图片）
            ├── people1.jpg
            ├── people6.jpg
             ……
         ├── crow.mat（标签文件）
   ```
   

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取参考代码仓源码
 
   ```bash
   git clone https://github.com/zzubqh/CrowdCount.git
   cd CrowdCount
   git checkout 0afa770a04c2e965a44ba1934ac646418f787384
   cd ..
   ```

2. 获取权重文件。

   点击 [链接](https://github.com/zzubqh/CrowdCount) 进入CrowdCount开源首页，下载模型权重文件mscnn_model_weights.h5。或者直接访问作者提供的 [百度网盘](https://pan.baidu.com/s/105FM8Di3MqsWsN6-l-S2vQ) ，提取码为 i2dn 。
   
   下载成功后，将mscnn_model_weights.h5文件放到./model路径下

3. 生成onnx文件。

      使用script/crowdcount_tf2onnx.py将TensorFlow模型转换成ONNX模型；

      ```bash
      cd script
      python3 crowdcount_tf2onnx.py --input_h5 ../model/mscnn_model_weights.h5 --output_onnx ../model/mscnn_model.onnx
      cd ..
      ```

4. 使用ATC工具将ONNX模型转OM模型。

      在当前模型的代码根目录下，执行ATC命令。
      1. Hi3403V100 SVP_NNN上的om模型转换命令
         ```bash
         # 如果您没有现成的bin文件，您需要参考后面的python脚本使用说明，使用preprocess.py脚本先生成bin文件。如果您想使用多个bin文件进行数据校准，多个文件间请使用;分割，例如a.bin;b.bin。
         atc --framework=5 --model="model/mscnn_model.onnx" --input_shape="input:1,3,224,224" --input_type="input:FP32" --output="model/mscnn_model" --image_list="data/preprocess/bin/people1015.bin" --soc_version=SS928V100 --compile_mode=6
         ```
      2. Hi3403V100 NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/mscnn_model.onnx" --input_shape="input:1,3,224,224" --output="model/mscnn_model" --enable_single_stream=true --soc_version=OPTG
         ```
         运行成功后生成crowdcount.om模型文件。
    
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

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数，cfg.txt中默认配置了推理时保存前处理和模型推理生成的原始bin文件，会占据较大的磁盘空间，您可以将该配置改为空字符串不保存节约磁盘空间。
    ```bash
    #### 3个主要消耗磁盘空间的参数如下 ####
    
    # 预处理后的二进制文件，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_preprocess_bin="../data/preprocess/bin"

    # 模型推理原始二进制结果，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_result_bin="../out/result/bin
   
    # 后处理得到的密度图，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_density_maps_dir="../out/result/density_maps"
    ```

2. 切换到可执行文件main所在的目录，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/mscnn_model.om  --input ../data/file_list.json
    ```
    每张图片的预测结果会保存在out/result/txt目录下，模型推理的原始bin文件默认会保存在out/result/bin目录下。

3. 精度验证。

   使用crowdcount_evaluate.py将模型推理的结果与数据集中的crow.mat标签文件进行对比，获取评估结果。

   ```bash
   cd script # 切换到script目录
   # 更改下开发板输出文件的权限，确保脚本能访问
   sudo chmod -R 777 ../out
   python3 crowdcount_evaluate.py \
      --pred_txt_dir "../out/result/txt" \
      --gt_mat "../../../../../datasets/CrowDensity/Crowdata/crow.mat" \
      --output_file "../out/result/metrics_results.txt" 
   cd ..
   ```

   参数说明：

   - --pred_txt_dir：预测人数结果txt文件目录。
   - --gt_mat：真实人数标注mat文件路径（如crow.mat）。
   - --output_file：评估结果保存文件。

   SVP_NNN平台上精度结果：
   ```bash
   评估结果 (预测样本数: 359)
   MAE (平均绝对误差): 3.9666
   RMSE (均方根误差): 9.7852
   ```
   NNN平台上精度结果：
   ```bash
   评估结果 (预测样本数: 359)
   MAE (平均绝对误差): 3.9443
   RMSE (均方根误差): 9.7378
   ```

4. 验证om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/mscnn_model.om  --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，输入路径为一张图片)
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可，例如设置为30）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 217.01ms, frame rate: 4.61fps
   ```
   在板端会输出显示，NNN平台上性能结果如下：
   ```bash
   execution time: 349.99ms, frame rate: 2.86fps
   ```

**（可选步骤）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 crowdcount_preprocess.py \
      --file_list "../data/file_list.json" \
      --output_dir "../data/preprocess/bin" \
      --target_size 224 224
   cd ..
   ```

   参数说明：

   - --file_list：输入的 JSON 文件路径，内含待处理图片的路径列表（格式需符合脚本预期：包含 fileList 字段，每个元素为图片路径）。脚本会从该文件读取所有待转换的图片路径。
   - --output_dir：预处理后生成的 .bin 文件保存目录。若目录不存在，脚本会自动创建。
   - --target_size：图片缩放填充后的目标尺寸，格式为 宽 高。

2. 使用python脚本进行模型推理（可以跟CPP版本的推理结果进行对比）
   ```bash
   cd script # 切换到script目录
   python3 crowdcount_infer.py \
      --infer_engine "tf" \
      --preprocess_bin_dir "../data/preprocess/bin" \
      --file_list_path "../data/preprocess/file_list.txt" \
      --infer_bin_dir "../out/result_pc/bin" \
      --tf_model_path "../model/mscnn_model_weights.h5" \
      --input_size 224 224
   cd ..
   ```

   参数说明：

   - --infer_engine: 推理引擎选择：onnx 或 tf。
   
   - --preprocess_bin_dir: 预处理后生成的二进制（.bin）文件存放目录。

   - --infer_bin_dir：推理结果（.bin 格式）的输出目录。若目录不存在，脚本会自动创建。

   - --file_list_path：python预处理阶段生成的文件列表（记录所有待推理的 .bin 文件路径）。脚本会从该文件读取待处理文件路径。

   - --tf_model_path：TF模型权重文件路径（.h5格式，仅infer_engine=tf时有效）。
   
   - --onnx_model_path：ONNX模型文件路径（仅infer_engine=onnx时有效）。
   
   - --input_size：模型输入张量的尺寸，格式为 高 宽（与脚本中 input_h, input_w 对应）。需与预处理阶段的目标尺寸一致，否则会导致张量形状不匹配。

3. 使用python脚本进行模型推理结果的后处理
   ```bash
   cd script # 切换到script目录
   python3 crowdcount_postprocess.py \
      --bin_dir "../out/result_pc/bin" \
      --output_dir "../out/result_pc" \
      --threshold 0.06 \
      --target_size 56 56
   cd ..
   ```

   参数说明：

   - --bin_dir: 存放推理输出bin文件的目录。

   - --output_dir：输出结果目录（包含密度图和txt计数）。
   
   - --target_size：模型输出尺寸，格式为 高 宽，224x224输入对应56x56输出
   
   - --threshold：密度图阈值过滤，默认0.06



# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | MAE (平均绝对误差) | RMSE (均方根误差) | 性能 (FPS) |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | CrowDensity  | 3.9666      | 9.7852         | 4.61         |
| Hi3403V100 NNN | 1          | CrowDensity  | 3.9443      | 9.7378         | 2.86         |

