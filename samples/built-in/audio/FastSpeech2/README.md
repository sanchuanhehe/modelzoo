# 基于FastSpeech2实现高效的端到端语音合成
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

FastSpeech2 是一种高效的端到端语音合成模型。相比 FastSpeech，FastSpeech2 引入了多尺度时长预测器和能量 / 基频预测分支，优化了时长预测模块并新增韵律特征建模，在合成速度和语音自然度上均有大幅提升。

- 参考实现：

  ```
  https://github.com/ming024/FastSpeech2
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | NPU | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | -------- | ---------------- | ------------ |
  | SVP_NNN  | text  | UINT16 | 1 x 40 | NL         |
  | SVP_NNN  | src_lens  | UINT16 | 1 x 1 | N         |

- 输出数据

  | NPU | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | -------- | ----------- | ----------- |
  | SVP_NNN  | mel  | FP32     | 1x80x320 | NCT         |
  | SVP_NNN  | wave  | FP32     | 1x1x81920 | NCT         |




## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入文本
│   ├── file_list_1.json  //输入文本, 单张循环测试FPS

├── script
│   ├── fastspeech2_pth2onnx.py       //pytorch模型转onnx
│   ├── fastspeech2_gen_file_list.py  //生成CPP程序的输入文件file_list.json
│   ├── fastspeech2_preprocess.py     //数据预处理脚本
│   ├── fastspeech2_infer.py          //模型推理和数据后处理脚本
│   ├── fastspeech2_evaluate.py       //精度评估脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── ...               //模型文件

├── CMakeLists.txt        //编译脚本，调用src目录下CMakeLists文件
├── *.json                //模型信息
├── LICENSE               //模型LICENSE
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
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | 6.10.t01spc030b660(请联系FAE获取) |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |6.10.t01spc030b660(请联系FAE获取)|  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=ie2sc9g1qk00)上进行下载。

创建`model`文件夹，并将om模型文件移动到`./model`目录下。
```
mkdir -p model
```
备注：若需要体验om模型转化过程，请参考[安装依赖](#section183221994410)和[模型转化](#section741711594517)章节。


### 获取字典文件

   ```bash
   git clone https://github.com/ming024/FastSpeech2.git
   cd FastSpeech2
   git reset --hard d4e79eb52e8b01d24703b2dfc0385544092958f3
   patch -p1 < ../fastspeech2.patch
   cd ..
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
    ./main --acl ../src/acl.json --model ../model/fastspeech_hifigan_en.om  --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```bash
# 建议使用python3.8
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>
1. 获取参考代码仓源码
 
   ```bash
   git clone https://github.com/ming024/FastSpeech2.git
   cd FastSpeech2
   git reset --hard d4e79eb52e8b01d24703b2dfc0385544092958f3
   patch -p1 < ../fastspeech2.patch
   cd ..
   ```
   
2. 获取原始数据集。（解压命令参考tar -xvf *.tar与 unzip *.zip）

   LJSpeech 是语音合成（TTS）领域的经典英文单说话人开源数据集，由 Keith Ito 基于 LibriVox 项目的公共领域有声书构建，是 FastSpeech2、Tacotron 等主流 TTS 模型的基准训练数据集。

   该模型使用 [LJSpeech数据集](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) 进行精度评估，在`modelzoo/datasets`源码目录下创建`LJSpeech-1.1`文件夹，文件结构如下：

   ```
   LJSpeech-1.1/
   ├── wavs/                # 核心：语音波形文件（单声道，16kHz采样率，16bit PCM格式）
   │   ├── LJ001-0001.wav   # 命名规则：LJ{章节ID}-{句子ID}.wav，共13100个音频文件
   │   ├── LJ001-0002.wav
   │   └── ...
   ├── metadata.csv         # 标注文件：文本-音频对应关系（核心标注）
   └── README               # 数据集说明（版本、构建方式、许可等）
   ```

   修改源作者的数据集配置路径
   ```bash
   cd FastSpeech2
   
   vi config/LJSpeech/preprocess.yaml

   # 将 corpus_path: "/home/ming/Data/LJSpeech-1.1" 修改为 corpus_path: "../../../../../datasets/LJSpeech-1.1"

   cd ..
   ```
   
3. 为精度评估准备数据

   为了评估精度，需要为FastSpeech2语音合成（TTS）模型构建高质量的 “文本 - 语音” 对齐数据。
   首先通过prepare_align.py完成数据预处理的基础配置，再借助蒙特利尔强制对齐工具（MFA）实现语音话语与音素序列的精准时间对齐（可以直接使用源代码作者提供的现成的对齐文件）。
   最后运行preprocess.py将这些对齐数据转换成模型可直接读取的格式，确保模型能读取到文本音素与语音时长、韵律等特征的对应关系，为后续评估提供标准化、对齐化的输入数据。

   源作者已经使用MFA工具处理好的对齐数据，请[点击下载LJSpeech.zip](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing) 。

   ```bash
   cd FastSpeech2
   # 该函数将原始数据集转换为 MFA 工具所需的标准化格式（音频 + 文本标签），为后续提取音素对齐信息奠定基础
   python3 prepare_align.py config/LJSpeech/preprocess.yaml

   cd preprocessed_data/LJSpeech/
   # 将下载的LJSpeech.zip放到preprocessed_data/LJSpeech目录下，解压缩
   unzip LJSpeech.zip

   # 回到源作者代码根目录，执行数据集预处理脚本，从原始的LJSpeech数据集计算得到文本、音频、梅尔频谱等数据
   cd ../../
   # 此步骤耗时较长（约20分钟），请耐心等待
   python3 preprocess.py config/LJSpeech/preprocess.yaml

   cd ..
   ```

4. 生成file_list.json

   main函数从file_list.json文件读取输入文本进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:

   ```
   data
      ├── file_list.json
      ├── file_list_1.json（单输入，但是多了loop参数，适合main函数循环推理测试性能）
      ├── cfg.txt（配置文件）
   ```
   
   执行 script/fastspeech2_gen_file_list.py 脚本，生成的file_list.json在data目录下。
   ```bash
    # 生成的文件在../data/file_list.json
    cd script
    python3 fastspeech2_gen_file_list.py ../FastSpeech2/preprocessed_data/LJSpeech/val.txt
    cd ..
   ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   先下载FastSpeech2的预训练模型，在 [谷歌云盘](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F) 中找到LJSpeech_900000.zip（英文模型）和AISHELL3_600000.pth.tar（中文模型）下载。
      ```bash
      # 执行下方命令，下载预训练模型，解压
      cd FastSpeech2
      mkdir -p output/ckpt/LJSpeech
      mkdir -p output/ckpt/AISHELL3
      
      # 将从goole云盘下载的LJSpeech_900000.zip（英文模型）放到output/ckpt/LJSpeech目录下，然后解压缩
      cd output/ckpt/LJSpeech
      unzip LJSpeech_900000.zip
   
      # 将从goole云盘下载的AISHELL3_600000.pth.tar（中文模型）放到output/ckpt/AISHELL3目录下，不需要解压缩但是需要更改下名称
      cd ../AISHELL3
      mv AISHELL3_600000.pth.tar 600000.pth.tar
   
      cd ../../../../
      ```

   然后下载Hifigan的预训练模型，在 [谷歌云盘](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y) 中找到LJ_FT_T2_V2下载。
      ```bash
      # 执行下方命令，下载预训练模型，解压
      cd FastSpeech2
      mkdir -p output/ckpt/hifigan
      cd output/ckpt/hifigan
      # 将从goole云盘下载的LJ_FT_T2_V2-{日期/编号等}.zip放到此目录下
      unzip LJ_FT_T2_V2-{日期/编号等}.zip
      cd ../../../../
      ```

2. 导出onnx文件。

      使用./script/fastspeech2_pth2onnx.py导出onnx模型

      ```bash
      # 需要在源作者FastSpeech2目录下执行脚本，否则无法读取源作者的一些相对路径的配置文件
      cd FastSpeech2
      # 导出英文的onnx模型，请执行以下命令
      python3 ../script/fastspeech2_pth2onnx.py \
         --target_len 40 \
         --language "en" \
         --npu "SVP_NNN" \
         --text "Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model" \
         --onnx_path "../model"
      # 导出中文的onnx模型，请执行以下命令
      python3 ../script/fastspeech2_pth2onnx.py \
         --target_len 40 \
         --language "zh" \
         --npu "SVP_NNN" \
         --text "湖南一餐馆推出石槽火锅生意火爆，老板称红薯叶等之前喂猪的食材现在人也在吃，这样的创新火锅你愿意尝试吗？" \
         --onnx_path "../model"
      cd ..
      ```

      运行成功后生成onnx模型到../model/fastspeech_hifigan_en.onnx，同时会生成转om模型需要的两个数据校准文件text.bin和src_lens.bin。

      参数说明：

      - --target_len：模型静态输入长度（音素），默认值为40，即输入音素序列的固定长度。
      - --language：模型支持的语言，可选值为"zh"（中文）、"en"（英文），默认值为"en"。
      - --npu：NPU类型，可选值为"SVP_NNN"、"NNN"，默认值为"SVP_NNN"。
      - --text：待合成的原始文本，仅适用于单句合成模式。
      - --onnx_path：ONNX模型导出路径，默认值为"../model"。


3. 使用ATC工具将ONNX模型转OM模型。

      执行ATC命令。
      1. Hi3403V100 SVP_NNN上的om模型转换命令
         ```bash
         atc --framework=5 --model="model/fastspeech_hifigan_en.onnx" --input_shape="text:1,40;src_lens:1,1" --output="model/fastspeech_hifigan_en" --soc_version=SS928V100 --image_list="text:model/text.bin;src_lens:model/src_lens.bin" --input_type="text:UINT16;src_lens:UINT16" --compile_mode=1 --matmul_per_channel_enable=1 --amplify_conv_alpha_enable=1 --fusion_switch_file=TransformerFusion:on
         ```

         运行成功后生成fastspeech_hifigan_en.om模型文件。
    
         参数说明：
    
         - --framework：原始框架类型，5代表ONNX模型。
         - --model：ONNX模型文件路径。
         - --input_shape：输入数据的shape。
         - --output：输出的OM模型路径。
         - --image_list：转换模型生成量化参数时用的校准数据。
         - --soc_version：处理器型号。
         - --compile_mode：编译模式，1代表数据量化使用16bit，0代表数据量化使用8bit；权重固定是8bit量化。
      
         注意：如果出现atc命令找不到，参考推理环境准备。


## 精度&性能评估<a name="section741711594518"></a>

1. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数。
    ```bash
      # 预处理后二进制文件保存路径，为空则不保存
      save_preprocess_bin="../data/preprocess/bin"

      # 预处理后二进制文件列表
      save_preprocess_txt="../data/preprocess"

      # 模型支持的语言，可选["zh", "en"]
      language="en"

      # 模型输入的音素长度
      target_phone_length=40

      # 中文和英文音素词典路径，作用是把文本词汇转换为语音合成所需的音素序列
      lexicon_en_path="../FastSpeech2/lexicon/librispeech-lexicon.txt"
      lexicon_zh_path="../FastSpeech2/lexicon/pinyin-lexicon-r.txt"

      # 中文到拼音的映射词典
      chinese_pinyin_dict="../data/chinese_pinyin_dict.txt"

      # 模型推理原始二进制结果路径（供Python精度评估用），为空则不保存
      save_result_bin="../out/result/bin"

      # 后处理生成的音频文件
      save_result_wav="../out/result/wav"
    ```

2. 登录到板端运行环境，切换到可执行文件main所在的目录，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/fastspeech_hifigan_en.om  --input ../data/file_list.json
    ```
    推理结果会保存在../out/result目录下

3. 精度验证。

   调用脚本与数据集标签person_keypoints_val2017.json比对，可以获得模型在coco2017验证集上的评估结果。

   ```bash
   # 需要在源作者FastSpeech2目录下执行脚本，否则无法读取源作者的一些相对路径的配置文件
   cd FastSpeech2
   # 更改下开发板输出文件的权限，确保脚本能访问
   sudo chmod -R 777 ../out
   python3 ../script/fastspeech2_evaluate.py \
      --output_dir "../out/result/bin" \
      --onnx_mel_len 320
   cd ..
   ```

   参数说明：

   - --output_dir：推理结果输出目录。

   - --onnx_mel_len：静态onnx模型输出的梅尔频谱的长度。

   SVP_NNN平台上精度结果：
   ```
    Validation Step 900000, Mel PostNet Loss: 1.7573
   ```

4. 验证om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/fastspeech_hifigan_en.om  --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，输入路径为一张图片)
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单输入的配置文件，注意：循环次数通过修改该文件中的loop变量即可，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 52.49ms, frame rate: 19.05fps
   ```

**（可选步骤）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 fastspeech2_preprocess.py \
      --input_file "../FastSpeech2/preprocessed_data/LJSpeech/val.txt" \
      --output_dir "../data/preprocess/bin" \
      --language en
   cd ..
   ```

   参数说明：

   - --input_file：输入文本文件路径，文件格式为“ID|x|x|文本”（竖线分隔，前三个字段为辅助信息，第四个字段为待处理文本），脚本会读取该文件中的文本内容进行预处理，默认值为"../FastSpeech2/preprocessed_data/LJSpeech/val.txt"。
   - --output_dir：预处理结果输出目录，脚本会将处理后的二进制文件（.bin）保存至该目录，若目录不存在则自动创建，默认值为"../data/preprocess/bin"。
   - --language：模型支持的语言，可选值为"zh"（中文）、"en"（英文），脚本会根据语言类型适配对应的文本预处理逻辑，默认值为"en"。


2. 使用python脚本进行模型推理
   ```bash
   cd script # 切换到script目录
   python3 fastspeech2_infer.py \
      --model_path "../model/fastspeech_hifigan_en.onnx" \
      --file_list "../data/preprocess/text_list.txt" \
      --output_dir "../out/result_pc"
   cd ..
   ```

   参数说明：

   - --model_path：ONNX模型文件路径，默认值为"../model/fastspeech_hifigan_en.onnx"。脚本会验证该文件是否存在，若不存在则报错。

   - --file_list：预处理生成的文本列表文件路径（text_list.txt），该文件记录所有待推理的预处理文件路径，脚本会从该文件读取待处理文件路径，默认值为"../data/preprocess/text_list.txt"。

   - --output_dir：推理结果输出目录，脚本会将语音合成的推理结果（如波形文件、梅尔频谱文件等）保存至该目录，若目录不存在则自动创建，默认值为"../out/result"。


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | Mel Loss | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | LJSpeech-1.1  | 1.7573        | 19.05 |