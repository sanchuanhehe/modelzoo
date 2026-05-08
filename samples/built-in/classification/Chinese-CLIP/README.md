# 基于Chinese-CLIP网络实现跨模态检索
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

Chinese-CLIP 是 CLIP 模型的中文版本。CLIP 通过对比学习方式，同时学习图像和文本的表示，并能够理解两者之间的语义关联。Chinese-CLIP 使用约 2 亿规模的中文图文对进行训练，其核心目标是解决中文场景下的跨模态检索、图像表示生成等任务。

- 参考实现：

  ```
   https://github.com/OFA-Sys/Chinese-CLIP.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 224 x 224 | NCHW         |
  | text    | FP32 | 1 x 512 |          |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | image_features    | FP32     | 1x512 |
  | text_features    | FP32     | 1x512 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── zeroshot.sh     //shell脚本，调用前后处理python文件
│   ├── accuracy.py     //精度验证脚本
    ├── pretprocess.py     //前处理脚本

├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── CMakeLists.txt    //编译脚本，调用src目录下CMakeLists文件
├── *.json			//模型信息
├── LICENSE			//模型LICENSE
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

1. 执行命令查看芯片名称。

    ```
    cat /proc/umap/sys
    #该设备芯片名为SS928V100 （自行替换）
    回显如下：
    [SYS] Version: [SS928V100XXXXXXXXX
    ```

2. 该模型需要以下环境

    **表 1** 版本配套表

| 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 |
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t14.7.b140  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022 |

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j8hn2954tk00)上进行下载。

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

4. 切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出输入图片的top5置信度的类别标识。测试图片上模型推理命令参考：
   
    ```
    ./main -a ../src/acl.json -j ../model/clip_img.om -t ../model/clip_text.om -i ../data/cfg.txt  -l 1
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

1. 安装依赖。
```
#Python 3.8.20
conda create -n myenvclip python=3.8.20 -y
pip3 install -r requirements.txt
```

2. 开源源码获取。

```
git clone https://github.com/OFA-Sys/Chinese-CLIP.git
cd Chinese-CLIP
git reset 2c38d03557e50eadc72972b272cebf840dbc34ea --hard
git apply ../clip.patch # SVP_NNN算力引擎使用clip.patch, NNN算力引擎需要修改命令为 git apply ../clip_nnn.patch
cd ..
``
```
  ## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集当前data路径下。数据集目录结构如下所示：

 data/
   |-- datasets
   |   |-- cifar-100
   |   |   |-- test
   |   |   |   |-- 000
   |   |   |   |   |-- chimp_s_000026.png
   |   |   |   |   |-- ...
   |   |   |   |-- 001
   |   |   |   |-- ...
   |   |   |-- train
   |   |   |-- index.json
   |   |   |-- label_cn.txt
   |   |   |-- label.txt
   |   ...
   ...

2. 生成文件集file_list.json，将原始数据集图片地址转换为模型的输入数据。
  
    执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。

    ```
    cp data/datasets/cifar-100/label_cn.txt data/
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```
    python3 ../../../../utils/generate_file_list.py -r data/datasets/cifar-100/test
    ```

   参数说明：
   - --dataset_path：原数据集所在路径。

3. 执行 script/target.py 脚本，生成真值文件
    ```
    python3 ./script/target.py
    
    ```
4. 数据预处理，生成量化校准文件
   
    执行 ./script/zeroshot.sh 脚本，完成数据预处理。
    
    ```
    bash ./script/zeroshot.sh ./script/preprocess.py data cifar-100 ViT-B-16 RoBERTa-wwm-ext-base-chinese ../models/clip_cn_vit-b-16.pt
    ```
    执行后会在data目录下生成txt和img文件夹，img下放置img模型转化需要的量化校准文件
    备注：如果报错，将zeroshot.sh文件的行尾序列修改为LF
## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。


1. 获取权重文件。
    点击下面链接下载模型权重文件clip_cn_vit-b-16.pt：
    [百度网盘](https://pan.baidu.com/s/19GrWVd4TVGN0BEyl9uTPiw?pwd=q2na) ，提取码为：q2na

2. 导出onnx文件。

    使用源码的onnx导出文件。

    ```
    python Chinese-CLIP/cn_clip/deploy/pytorch_to_onnx.py  --model-arch ViT-B-16 --pytorch-ckpt-path ./model/clip_cn_vit-b-16.pt --save-onnx-path ./model/vit-b-16 --context-length 512 --convert-text --convert-vision
    ```

    参数说明:

    - --pytorch-ckpt-path：权重文件。
    - --save-onnx-path：生成 onnx 文件。建议保存为./model/vit-b-16
    最终得到：
    vit-b-16.img.fp32.onnx和vit-b-16.txt.fp32.onnx

3. 使用 onnx-simplifier 简化 onnx 模型。
      1. 图像模型
      
            ```
            onnxsim model/vit-b-16.img.fp32.onnx model/vit-b-16_img_sim.onnx --overwrite-input-shape "image:1,3,224,224"
            ```
        
      2. 文本模型
      
            ```
            onnxsim model/vit-b-16.txt.fp32.onnx model/vit-b-16_txt_sim.onnx --overwrite-input-shape "text:1,512"
            ```

4. 使用ATC工具将ONNX模型转OM模型。
   
    Hi3403V100 SVP_NNN上的om模型转换命令:
    1、图像
    1. Hi3403V100 SVP_NNN上的om模型转换命令

    ```
    atc --framework=5  --model="./model/vit-b-16_img_sim_SVP_NNN.onnx" --input_shape="image:1,3,224,224" --output="./model/clip_img" --image_list="./data/img/0.bin" --compile_mode=6 --matmul_per_channel_enable=1 --softmax_optimize_enable=1 --fusion_switch_file=TransformerFusion:on --soc_version=SS928V100
    ```

    2. Hi3403V100 NNN上的om模型转换命令

    ```
    atc --framework=5 --model="./model/vit-b-16_img_sim_NNN.onnx" --input_shape="image:1,3,224,224" --output="./model/clip_img" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG
    ```

    2、文本
    1. Hi3403V100 SVP_NNN上的om模型转换命令

    ```
    atc --framework=5  --model="./model/vit-b-16_txt_sim_SVP_NNN.onnx" --input_type=text:FP32 --output="./model/clip_text" --image_list="./data/quant_text.txt" --compile_mode=6 --matmul_per_channel_enable=1 --fusion_switch_file=TransformerFusion:on --force_to_cpu=/bert/embeddings/word_embeddings/Gather --gfpq_param_file=./data/calibration_param.txt --soc_version=SS928V100
    ```

    2. Hi3403V100 NNN上的om模型转换命令

    ```
    atc --framework=5  --model="./model/vit-b-16_txt_sim_NNN.onnx" --output="./model/clip_text" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG
    ```

    运行成功后生成clip_img.om和clip_text.om模型文件。

    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_shape：输入数据的shape。
    - --input_type：输入数据类型。
    - --output：输出的OM模型。
    - --image_list: 量化校准数据。
    - --compile_mode:编译模式，参数值6代表使用16bit量化数据，使用8bit量化权重。且仅对CUBE算子进行量化
    - --matmul_per_channel_enable:权重是否每通道单独量化
    - --fusion_switch_file：融合开关配置文件
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。
    - --soc_version：处理器型号。

## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。

    ```
    ./main -a ../src/acl.json -j ../model/clip_img.om -t ../model/clip_text.om -i ../data/cfg.txt  -l 1
    备注：验证精度需要修改cfg文件中输入的accuary、img_list和txt_list。
    其中img_list修改为../data/file_list.json 。 txt_list修改为../data/label_cn.txt
    ```

2. 精度验证。

    调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在accuracy.txt中。

    ```
    python ./script/accuracy.py --output ${result_dir} --label ${gt_file} --result ${--result_file}
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/out/

    - --label：真值标签文件target.txt所在路径。

    - --result：输出精度结果所在的位置。

    例如：  `python ./script/accuracy.py --output ./out/result/out/ --label ./data/target.txt --result ./out/accuracy.txt`
    
    SVP_NNN平台上精度结果：
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "9999"}, {"key": "Number of classes", "value": "100"}, {"key": "Top1 accuracy", "value": "63.0%"}, {"key": "Top2 accuracy", "value": "76.52%"}, {"key": "Top3 accuracy", "value": "82.47%"}, {"key": "Top4 accuracy", "value": "85.78%"}, {"key": "Top5 accuracy", "value": "87.89%"}]}
    ```

    NNN平台上精度结果：
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "9999"}, {"key": "Number of classes", "value": "100"}, {"key": "Top1 accuracy", "value": "61.8%"}, {"key": "Top2 accuracy", "value": "75.31%"}, {"key": "Top3 accuracy", "value": "81.12%"}, {"key": "Top4 accuracy", "value": "84.56%"}, {"key": "Top5 accuracy", "value": "87.05%"}]}
    ```

2. 验证img的om模型的性能，参考命令如下：

    ```
    ./main -a ../src/acl.json -j ../model/clip_img.om   -i ../data/cfg.txt  -l 1
    ```

    参数说明：(此模式下，请详细查看cfg.txt说明，配置performance、txt_list和img_list)
    其中img_list修改为../data/file_list_img_1.json。txt_list修改为../data/file_list_txt_1.json
    - --imgmodel：img模型路径。
    - --acl: acl.json文件的路径，默认放在src目录下。
    - --input: 输入的图像数据列表路径
    - --loop： 循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

    在板端会输出显示，Hi3403V100 SVP_NNN平台上性能结果如下：
    ```
    [INFO]  time: 8975423, fps: 11.141536
    ```

    NNN平台上性能结果如下：
    ```
    execution time: 1049.88ms, frame rate: 0.95fps
    ```

    备注：txt模型性能
    ```
    ./main -a ../src/acl.json -t ../model/clip_text.om   -i ../data/cfg.txt  -l 1
    ```
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Chinese_CLIP模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   |  Zero-shot Image Classification |       性能(FPS)|
| ----------- | ---------- | -------- |  ------------------ |-----------------|
| Hi3403V100 SVP_NNN | 1          | CIFAR100 |  0.63            | 11.142   |
| Hi3403V100 NNN | 1          | CIFAR100 |  0.618            | 0.95   |

