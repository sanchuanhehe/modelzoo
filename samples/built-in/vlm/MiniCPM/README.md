# 基于MiniCPM-4v-0.5B的视觉语言模型(Vision Language Model, 简称VLM)
- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)
    - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型推理](#ZH-CN_TOPIC_0000001126281700)

  - [快速开始(推荐)](#section4622531142816)
  - [安装依赖](#section183221994410)
  - [准备数据集](#section183221994411)
  - [模型转化](#section741711594517)
  - [性能评估](#section741711594518)

- [模型推理性能](#ZH-CN_TOPIC_0000001172201573)

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>
MiniCPM-4v-0.5B: 小参数, 大智慧——端侧多模态模型

由面壁智能(OpenBMB)打造的MiniCPM-4v-0.5B, 以0.53B的精简参数量，在端侧设备上实现了卓越的图文理解与交互能力。专为边缘计算场景设计，让每一分算力都充分发挥价值。

三大核心优势：
    1. 创新架构，轻装上阵
通过创新的稀疏感知训练和视觉压缩技术，实现 16:1 视觉特征压缩比，在保证高精度的同时，极大降低了推理算力消耗与内存占用。
    2. 高能数据，以小博大
依托高密度数据体系，数据准备成本下降90%。通过汇聚全球高质量语料进行精细化对齐，用优质的数据训练模型，实现越级性能表现。
    3. 高效训练，成本锐减
采用原创 WSD 调度策略与“模型风洞”技术，相比传统方案，搜索算力节省超99%，整体训练成本节省约60%，为端侧模型的持续迭代提供高效路径。

海思平台技术支撑：
现已适配 Hi3403V100 平台，持续生成速度达 21 tokens/s，为边缘侧设备提供流程、敏捷的智能视觉体验。

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | NPU | 输入数据(vision.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | image  | FP32 | 1 x 3 x 16 x 16384 | NCT         |

  | NPU | 输入数据(resample.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | vision_hidden_states  | FP32 | 1 x 1 x 73448 | ND         |

  | NPU | 输入数据(prefill_decode.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | inputs_embeds  | FP32 | 1 x 200 x 1024 | ND         |
  | SVP_NNN  | attention_mask  | FP32 | 1 x 1 x 200 x 200 | NCT         |

  | NPU | 输入数据(decode.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | input_ids  | FP32 | 1 x 1 x 1024 | ND         |
  | SVP_NNN  | attention_mask  | FP32 | 1 x 1 x 1 x 1024 | NCT         |
  | SVP_NNN  | indices  | FP32 | 1 x 1 | ND         |
  | SVP_NNN  | unsqueeze_6  | FP32 | 1 x 1 x 1 x 64 | NCT         |
  | SVP_NNN  | unsqueeze_7  | FP32 | 1 x 1 x 1 x 64 | NCT         |
  | SVP_NNN  | past_keys_0...23/present_values_0...23  | FP32 | 1024 x 1 x 2 x 64 | NCT         |

- 输出数据

  | NPU | 输出数据(vision.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | vision_features_report_0_0  | FP32 | 1 x 1024 x 768 | ND         |

  | NPU | 输出数据(resample.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | MatMul_report_0_0  | FP32 | 1 x 64 x 1024 | ND         |

  | NPU | 输出数据(prefill_decode.om) | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | keys  | FP32 | 24 x 200 x 1 x 2 x 64 | ND         |
  | SVP_NNN  | values  | FP32 | 24 x 200 x 1 x 2 x 64 | ND         |
  | SVP_NNN  | logits_report_0_2  | FP32 | 1 x 200 x 73448 | ND         |

  | NPU | 输出数据（decode.om） | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | -------- | ----------- | ----------- |
  | SVP_NNN  | past_keys_0...23/present_values_0...23  | FP32 | 1024 x 1 x 2 x 64 | NCT         |
  | SVP_NNN  | logits  | FP32     | 1 x 1 x 73448 | ND         |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── cfg.txt                  // 参数配置文件
│   ├── file_list.json           // 输入问题
│   ├── file_list_1.json         // 输入问题, 单个问题循环测试TPS
│   ├── rotary_position_emb0.bin // RoPE旋转位置编码权重文件（位置编码 embedding part0）
│   ├── rotary_position_emb1.bin // RoPE旋转位置编码权重文件（位置编码 embedding part1）
│   ├── token_emb.bin            // Token Embedding 权重文件（二进制词向量数据）
│   ├── tokenizer.json           // 分词器配置文件（词表）

├── doc
│   ├── 快速开始.md              // 快速开始

├── src
│   ├── acl.json                 // 系统初始化的配置文件
│   ├── CMakeLists.txt           // 编译脚本
│   ├── main.cpp                 // 模型运行CPP函数入口（模型加载、推理流程调度、输入输出控制）

├── model
│   ├── decode.om                // 自回归解码模型（Decoder阶段推理）
│   ├── prefill_decode.om        // Prefill 模型（首token生成）
│   ├── resample.om              // 多模态重采样模型（视觉特征与文本token对齐）
│   ├── vision.om                // 视觉编码模型（Vision Encoder，图像特征提取）

├── CMakeLists.txt               // 编译脚本，调用src目录下CMakeLists文件
├── *.json                       // 模型信息（模型结构描述、参数映射、运行模式配置等）
├── LICENSE                      // 模型LICENSE
├── README.md                    // README

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
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.5](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.5.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.3/docs/zh-CN/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.3/docs/zh-CN/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.3/) |
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.5](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.5.tgz)  |  aarch64-mix210-linux-gcc |  Linux |  SS928V100V1.0.2.5 B060  |

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=indvkfmcp400)上提供转化成功的om模型文件，可以从 Hispark ModelZoo 主页进行下载。

创建`model`文件夹，并将om模型文件移动到`./model`目录下。
```
mkdir -p model
```

### 编译代码和运行应用

#### 编译代码

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为`build`。
    ```
    mkdir -p build
    ```

2. 切换到`build`目录，执行**cmake**生成编译文件。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    例如，开发环境为X86架构、运行环境为ARM架构时，执行以下命令进行交叉编译。交叉编译工具链按运行环境操作系统，可选toolchain_aarch64_linux.cmake；SOC_VERSION按算力引擎可选SS928V100，请根据运行环境和算力引擎平台选择。
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

4. 切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，输出推理结果。测试图片上模型推理命令参考：
    
    ```
    ./main --input ../data/file_list_1.json
    ```


## 准备数据集<a name="section183221994411"></a>

1. 构建数据集。

   在模型主目录下创建 `datasets` 文件夹。
   ```bash
   mkdir datasets
   ```

   下载图片放置到datasets目录下，建议下载512x512大小左右的jpg或png格式图片。
   
   在`data/file_list.json`文件内fileList item下添加图片地址和问题描述，格式如下：
   ```json
    {
        "fileList": [
            [
                "../../../../../datasets/testdata/16.png",
                "描述图片内容"
            ],
            [
                "../../../../../datasets/testdata/16.png",
                "描述图片内容"
            ]
        ]
    }
   ```


## 模型获取<a name="section741711594517"></a>

   请下载如下om/bin/json文件：
   rotary_position_emb0.bin
   rotary_position_emb1.bin
   token_emb.bin
   tokenizer.json
   decode.om
   prefill_decode.om
   resample.om
   vision.om

## 性能评估<a name="section741711594518"></a>

1. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数，请确认如下参数路径正确。
    ```bash
      # vision.om 文件路径, 不可为空
      vision_om_path="../model/vision.om"

      # resample.om 文件路径, 不可为空
      resample_om_path="../model/resample.om"

      # prefill_decode.om 文件路径, 不可为空
      prefill_decode_om_path="../model/prefill_decode.om"

      # decode.om 文件路径, 不可为空
      decode_om_path="../model/decode.om"

      # rotary_position_emb0.bin 文件路径, 不可为空
      rotary_position_emb0_bin_path="../data/rotary_position_emb0.bin"

      # rotary_position_emb1.bin 文件路径, 不可为空
      rotary_position_emb1_bin_path="../data/rotary_position_emb1.bin"

      # token_emb.bin 文件路径, 不可为空
      token_emb_bin_path="../data/token_emb.bin"

      # tokenizer.json 文件路径, 不可为空
      tokenizer_json_path="../data/tokenizer.json"

      # 推理结果保存路径, 不可为空
      result_dir="../out/result"
    ```

2. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。

    ```
    ./main --input ../data/file_list.json
    ```
   参数说明：
   - --input:  问题描述和图片地址文件。
   推理结果会保存在../out/result目录下

3. 验证om模型的性能，参考命令如下：

   执行
   ```bash
   ./main --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，file_list_1.json只放一组图片-文字)
   - --input:  问题描述和图片地址文件，将file_list_1.json的loop参数设置为100

    file_list_1.json中的配置代表对一个问题重复推理100次，程序执行时会在板端会输出打印推理的平均时间和帧率。


# 模型推理性能<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，MiniCPM-4v-0.5B 模型的性能参考下列数据。

| 芯片型号    | Batch Size | TTFT | TPS |
| ----------- | ---------- | ------------- | ------------- |
| Hi3403V100 SVP_NNN | 1          | 519.74 ms | 21.30 token/s |