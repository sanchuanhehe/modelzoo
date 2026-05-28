# 基于CharCTC网络实现小云唤醒词检测
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

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

小云唤醒词检测模型基于CharCTC（Character-level Connectionist Temporal Classification）架构，用于检测语音中的特定唤醒词"小云小云"。模型提取音频的Fbank特征，通过LFR（Low Frame Rate）降采样和CMVN归一化后进行CTC解码，输出关键词检测结果。

- 参考实现：

  ```
  https://www.modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 说明 |
  | -------- | -------- | ---------------- | ---- |
  | audio    | WAV      | 16kHz, 16bit PCM | 音频波形 |

- 预处理后输入

  | 特征数据 | 数据类型 | 大小        | 说明 |
  | -------- | -------- | ----------- | ---- |
  | fbank    | FP32     | 151 x 400     | LFR后的Fbank特征 |

- 输出数据

  | 输出数据 | 数据类型 | 大小     | 说明 |
  | -------- | -------- | -------- | ---- |
  | logits   | FP16     | 151 x 2599 | CTC预测的token概率分布 |

- 后处理输出

  | 输出数据 | 数据类型 | 说明 |
  | -------- | -------- | ---- |
  | keyword  | string   | 检测到的关键词 "小云小云" |
  | score    | float    | 检测置信度分数 |

## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── file_list_1.json     //测试数据列表
│   ├── test_audio.wav       //测试音频

├── script
│   ├── export_charctc_onnx.py       //模型导出脚本
│   ├── gen_calibration_batch.py     //生成量化校准数据
│   ├── gen_groundtruth.py           //生成真值数据
│   ├── accuracy.py         //精度评估

├── src
│   ├── acl.json             //系统初始化的配置文件
│   ├── CMakeLists.txt       //编译脚本
│   ├── main.cpp             //主程序

├── model
│   ├── kws-610.om           //模型文件，支持4s左右音频
│   ├── kws-610-a.om           //精度验证模型文件，支持10s左右音频

├── process
│   ├── xiaoyun_postprocess.cpp           //模型后处理
│   ├── xiaoyun_preprocess.cpp           //模型前处理

├── doc
│   ├── 快速开始.md          //快速开始文档

├── CMakeLists.txt           //编译脚本
├── LICENSE                  //许可证
├── requirements.txt         //Python依赖
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

1. 执行命令查看芯片名称。
    ```
    cat /proc/umap/sys
    #该设备芯片名为HI3516CV610 （自行替换）
    回显如下：
    [SYS] Version: [HI3516CV610xxxx],
    ```

2. 该模型需要以下环境

    **表 1** 版本配套表

    | 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
    | --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
    | Hi3516CV610 | SVP_NNN | Hi3516CV610   |  驱动和开发环境安装指南 | Ascend-cann-toolkit_6.10.t06spc020b023_linux.x86_64.run |  gcc-20250305-arm-v01c02-linux-musleabi.tgz |  Linux |  Hi3516CV610R001C01SPC020  |
   

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelZoo)上进行下载。

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

    "../src"表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    例如，开发环境为 X86、运行环境为 ARM 时交叉编译。本样例面向 `Hi3516CV610`（板端 32 位 ARM），使用 `toolchain_arm_v01c02_linux.cmake`。
    ```
    cd build
    cmake ../src -DSOC_VERSION=Hi3516CV610 -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_arm_v01c02_linux.cmake
    ```

3. 执行**make**命令，生成的可执行文件main在"./out"目录下。

#### 运行应用

1. 将modelzoo代码上传到板端运行环境。
2. 以运行用户登录板端运行环境。
3. 切换到可执行文件main所在的目录，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4. 切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出检测到的关键词和置信度分数。测试音频上模型推理命令参考：
    
    ```
    ./main --model ../model/kws-610.om --input ../data/file_list_1.json
    ```

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.10.12
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取测试音频数据。

   本模型使用包含唤醒词"小云小云"的测试音频进行推理测试。用户可自行录制或使用开源代码仓库“iic/speech_charctc_kws_phone-xiaoyun/unittest/example_kws/test_wav.scp”测试数据（file_list.json）。
   当前示例使用开源代码仓库。
   1）生成精度校准数据

     ```
     python ./script/gen_groundtruth.py
     
     ```

## 模型转化<a name="section741711594517"></a>

使用ModelScope下载模型权重，将模型转换为ONNX文件，再使用ATC工具将ONNX文件转为离线推理模型文件.om文件。

1. 获取开源源码
   ```
   git clone https://github.com/modelscope/FunASR.git
   cd FunASR
   git checkout b842ff8107e1da950947ada0d11ae3c008baeb54
   git apply ../funasr.patch
   pip install -e .
   cd ../
   ```

2. 获取权重文件。


   前往[ModelScope](https://www.modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun)下载模型。

   ```
   mkdir iic
   cd iic
   git lfs install
   git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun.git
   cd speech_charctc_kws_phone-xiaoyun
   git checkout 7b61475f2b7d6b0348f624f0853303a3a374f7bc
   cd ../../
   ```

3. 导出onnx文件。

1）使用./script/export_charctc_onnx.py导出onnx文件。

     ```
     mkdir model
     python ./script/export_charctc_onnx.py
     
     ```
     转化成功后输出日志如下：
     ```
      funasr version: 1.3.1.
      WARNING:root:trust_remote_code: True
      Loading remote code successfully: model
      /home/hispark/shared/xiaoyun/modelzoo-dev/samples/built-in/audio/fsmn_kws
      input shape: torch.Size([1, 151, 400])
      ONNX model saved to: ./model/speech_charctc_kws_phone-xiaoyun.onnx

     ```
     输出说明：

     - input shape：模型输入维度。
     - ONNX model saved to：生成 onnx 文件路径。

2) 简化onnx模型

    ```
    onnxsim ./model/speech_charctc_kws_phone-xiaoyun.onnx ./model/speech_charctc_kws_phone-xiaoyun_onnxsim.onnx
    ```


4. 使用ATC工具将ONNX模型转OM模型。
    1）生成量化数据

     ```
     python ./script/gen_calibration_batch.py --target_frames=151
     ```
     输入参数说明：

     --target_frames：量化校准信息长度。来源于模型转化onnx时输出的input shape

     2) 执行ATC命令。
     
     Hi3516CV610 SVP_NNN上的om模型转换命令

         ```
          atc --framework=5 --model="./model/speech_charctc_kws_phone-xiaoyun_onnxsim.onnx" --input_type="speech_features:FP32" --output="./model/kws-610" --image_list="./data/calibration/calib_batch.txt" --soc_version=Hi3516CV610 --output_type="FP16" --online_model_type="0" --compile_mode=5
         ```
         运行成功后生成kws-610.om模型文件。

         参数说明：
       
         - --framework：5代表ONNX模型。
         - --model：为ONNX模型文件。
         - --input_shape：输入数据的shape，T为动态维度（音频帧数）。
         - --output_type：指定网络输出类型。
         - --image_list：转换模型生成量化参数时用的校准数据。
         - --output：输出的OM模型地址。
         - --soc_version：处理器型号。
         - --online_model_type：转换生成模型的类型，用于板端执行profiling或dump数据，0表示不带调试数据。
         - --compile_mode：编译模式，5表示数据和权重量化使用8bit，且仅对CUBE算子进行量化

## 精度&性能评估<a name="section741711594518"></a>

1. 精度验证，由于当前转出的om是固定shape的，输入长音频会被截断，只获取前151帧的音频，如果关键字音频刚好超出固定shape长度，被丢弃，则会识别失败。所以如果输入长音频，请参考上述流程，输入长音频文件，转化对应的onnx、om。
  1）导出可以处理10s左右音频的onnx，其他转om流程参考模型转化
 
     ```
     python ./script/export_charctc_onnx.py --input="./iic/speech_charctc_kws_phone-xiaoyun/unittest/example_kws/wav/20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_50.wav"
     
     onnxsim ./model/speech_charctc_kws_phone-xiaoyun.onnx ./model/speech_charctc_kws_phone-xiaoyun_onnxsim.onnx
     ```
2) 板端推理
    修改cfg.txt文件，指定输出目录
    ```
      # 模型推理结果
      save_result_txt=../out/result/txt
    ```
    执行板端推理
    ```
  
    ./main --model ../model/kws-610-a.om --input ../data/file_list.json
    ```

2 在开发端验证精度

    ```
    sudo chmod 777 -R ./
    python script/accuracy.py --result_dir out/result/txt
    ```

    执行结果
    ```
    === Evaluation Result ===
    Total: 12
    Correct: 12
    Accuracy: 1.0000 (100.00%)
    TP: 7, FP: 0, FN: 0, TN: 5
    Precision: 1.0000
    Recall: 1.0000
    F1: 1.0000

    ```
3）精度验证

3. 验证om模型的性能，参考命令如下：
    ```
    执行./main --model ../model/kws-610.om --input ../data/file_list_1.json
    ```

    参数说明：(此模式下，file_list_1.json只放一个数据)

    - --model：om模型路径。
    
    在板端会输出显示
    SVP_NNN平台上性能结果如下：
    
    ```
    [INFO] 1970-01-09 06:24:29.011 [model.cpp:247] execution time: 5.87ms, frame rate: 170.27fps

    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，fsmn_kws模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标1（Acc@1） |性能(fps) |
| ----------- | ---------- | -------- | ------------------ | ----------- |
| Hi3516CV610 SVP_NNN | 1          | test  | 100%            | 170.27   |

