# 基于CodeFormer实现盲人脸修复（该模型仅用于非商用）
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

CodeFormer 是一种基于码本查找 Transformer 的鲁棒盲人脸修复模型。相比传统方法，它通过生成对抗网络与量化编码技术，能有效处理模糊、噪声等多种退化问题，兼顾人脸修复质量与身份保真度，适用于盲人脸恢复场景。

- 参考实现：

  ```bash
  https://github.com/sczhou/CodeFormer
  https://github.com/hpc203/CodeFormer-onnxrun-cpp-py
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | input.1  | RGB_FP32 | 1 x 3 x 512 x 512 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小        | 数据排布格式        |
  | -------- | -------- | ----------- | ----------- |
  | 2936  | FP32     | 1 x 3 x 512 x 512 | NCHW |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── preprocess        //数据预处理结果
│   ├── result            //模型推理结果
│   ├── cfg.txt           //参数配置文件
│   ├── file_list.json    //输入图片路径
│   ├── file_list_1.json  //输入图片路径, 单张循环测试FPS

├── script
│   ├── codeformer_preprocess.py         //数据预处理脚本
│   ├── codeformer_infer.py              //模型推理脚本
│   ├── codeformer_postprocess.py        //数据后处理脚本
│   ├── codeformer_evaluate.py           //精度评估脚本
│   ├── codeformer_gen_lq_dataset.py     //生成低质量人脸数据脚本

├── src
│   ├── acl.json          //系统初始化的配置文件
│   ├── CMakeLists.txt    //编译脚本
│   ├── main.cpp          //模型运行CPP函数入口

├── model
│   ├── ...	      //模型文件

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
   [SYS] Version: [SS928V100XXXXXXXXX]
   ```

2. 该模型需要以下环境

  **表 1** 版本配套表

| 芯片型号  | npu  | soc_version | 环境准备指导     |
| --------- | ---- | ----------- | ---------------- |
| SS928V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/Hispark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |
| SS928V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/Hispark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |                                                     |  -                                                            |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取参考代码仓源码
 
   ```bash
   git clone https://github.com/sczhou/CodeFormer.git
   ```


2. 安装依赖。

   ```bash
   # 源作者使用的是python3.8，建议使用相同版本创建虚拟环境
   pip3 install -r requirements.txt
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   CodeFormer论文中使用的CelebA-Test数据集由作者制作但是并未公开。其制作原理是源于CelebA-HQ 数据集中选取的若干张高质量人脸图像作为参考（HQ 图像），然后按照退化策略（包括随机噪声、模糊、压缩失真等），合成对应的低质量（LQ）图像，最终形成由 “LQ 输入 - HQ 目标” 对组成的测试集，用于评估模型的人脸修复性能。

   因此我们采用类似的方式，基于CelebA-HQ数据集中的高质量图片来生成低质量图片，然后使用CodeFormer进行盲人脸修复，将修复后的结果和原始高质量图片进行对比。
   
   点击 [CelebA-512](https://pan.baidu.com/s/1INQ9ec_o_B8YYdF5_9Ue6A?pwd=6ja7) 下载数据集进行精度评估，在`modelzoo/dataset`源码目录下创建`celeba-512`文件夹，文件结构如下：
   
   ```
   celeba-512
      ├── celeba-512-300（300张图片）
         ├── hq（高质量人脸，来自CelebA-HQ数据集，作为真值数据）
            ├── 000004.jpg
            ├── 000009.jpg
             ……
         ├── lq（低质量人脸，使用codeformer_gen_lq_dataset.py生成，作为模型输入数据）
            ├── 000004.jpg
            ├── 000009.jpg
             ……
      ├── celeba-512-3000（3000张图片）
         ├── hq（高质量人脸，来自CelebA-HQ数据集，作为真值数据）
            ├── 000004.jpg
            ├── 000009.jpg
             ……
         ├── lq（低质量人脸，使用codeformer_gen_lq_dataset.py生成，作为模型输入数据）
            ├── 000004.jpg
            ├── 000009.jpg
             ……
   ...
   ```
   
2. 生成file_list.json

   main函数从file_list.json文件读取输入文件列表进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
   在data目录下提供了file_list.json的demo样例:

   ```
   data
      ├── file_list.json
      ├── file_list_1.json（单张图片输入，但是多了loop参数，适合main函数循环推理测试性能）
      ├── cfg.txt（配置文件）
   ```
   
   执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    ```bash
    # 在当前项目根目录下执行，生成的文件在data/file_list.json
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```bash
    python3 ../../../../utils/generate_file_list.py ../../../../datasets/celeba-512/celeba-512-300/lq
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。
   

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   点击 [链接](https://pan.baidu.com/s/18g918VdcyODdVcSOCLutOA?pwd=ixre) 下载codeformer.onnx(原始的codeformer.pth模型文件可以通过 [链接](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth) 下载，建议直接下载onnx文件即可) 。
   
   下载成功后，将codeformer.onnx文件放到./model路径下

2. 简化onnx文件。

      先使用script/codeformer_fix_onnx.py精简不需要的输入和输出参数；

      然后使用onnxsim命令去除冗余节点和计算，提升模型推理效率。

      ```bash
      cd script
      python3 codeformer_fix_onnx.py --input ../model/codeformer.onnx --output ../model/codeformer_fixed.onnx
      
      onnxsim ../model/codeformer_fixed.onnx ../model/codeformer_simplified.onnx
      ```

3. 使用ATC工具将ONNX模型转OM模型。

      在当前模型的代码根目录下，执行ATC命令。
      1. SS928V100 SVP_NNN上的om模型转换命令
         ```bash
         # 如果您没有现成的bin文件，您需要参考后面的python脚本使用说明，使用preprocess.py脚本先生成bin文件。如果您想使用多个bin文件进行数据校准，多个文件间请使用;分割，例如a.bin;b.bin。
         atc --framework=5 --model="model/codeformer_simplified.onnx" --input_shape="input.1:1,3,512,512" --input_type="input.1:FP32" --output="model/codeformer" --image_list="data/preprocess/bin/000004.bin" --soc_version=SS928V100
         ```
         运行成功后生成codeformer.om模型文件。
    
         参数说明：
    
         - --framework：原始框架类型，5代表ONNX模型。
         - --model：ONNX模型文件路径。
         - --input_shape：输入数据的shape。
         - --output：输出的OM模型路径。
         - --image_list：转换模型生成量化参数时用的校准数据。
         - --soc_version：处理器型号。
         - --compile_mode：编译模式，6代表数据量化使用16bit，权重量化使用8bit，且仅对CUBE算子进行量化，非CUBE算法使用fp16格式。注：SVP_NNN上选取其他编译模式可能导致精度下降
      
         注意：如果出现atc命令找不到，参考推理环境准备。


## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1. 切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```bash
    mkdir -p build
    ```

2. 切换到build目录，执行**cmake**生成编译文件。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。
    例如，当开发环境为X86架构，运行环境为ARM架构时，执行以下命令进行交叉编译。其中交叉编译器为aarch64-mix210-linux-gcc，SOC_VERSION根据使用npu的不同有SS928V100和OPTG两个选项，请根据运行环境选择使用。
	  
      ```bash
      cd build
      cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc -DSOC_VERSION=${soc_version}
      ```
    ../src表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。
    
    

3.  执行**make**命令，生成的可执行文件main在“./out“目录下。

	```bash
	make
	```

**步骤2：运行应用。**

1. 以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2. 以运行用户登录运行环境（Host）。
3. 切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，给该目录下的main文件加执行权限。

    ```bash
    chmod +x main
    ```
4. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数，cfg.txt中默认配置了推理时保存前处理和模型推理生成的原始bin文件，会占据较大的磁盘空间，您可以将该配置改为空字符串不保存节约磁盘空间。
    ```bash
    #### 两个主要消耗磁盘空间的参数如下 ####
    
    # 预处理后的二进制文件，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_preprocess_bin="../data/preprocess/bin"

    # 模型推理原始二进制结果，设置为空字符串则不保存，节约磁盘空间可以不保存
    save_result_bin="../data/result/bin"
    ```

5. 切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```bash
    ./main --acl ../src/acl.json --model ../model/codeformer.om  --input ../data/file_list.json
    ```
    修复后的图像默认会保存在data/result/img目录下，模型推理的原始bin文件默认会保存在data/result/bin目录下。

**步骤3：验证精度和性能**

1. 精度验证。

   使用codeformer_evaluate.py将模型推理的结果与数据集中原始的高清人脸图像进行对比，获取评估结果。

   ```bash
   cd script # 切换到script目录
   python3 codeformer_evaluate.py \
      --restored_dir "../data/result/img" \
      --gt_dir "../../../../../datasets/celeba-512/celeba-512-300/hq" \
      --output_file "../data/result/metrics_results.txt" 
   ```

   参数说明：

   - --restored_dir：CodeFormer修复结果目录。

   - --gt_dir：ground truth高清图像目录（512x512）。
   
   - --output_file：评估结果保存文件。

   SVP_NNN平台上精度结果：
   ```bash
   平均PSNR: 25.0426 dB
   平均SSIM: 0.6692
   平均LPIPS: 0.2151
   ```


2. 验证batch_size的om模型的性能，参考命令如下：

   ```bash
   cd out # 切换到out目录
   ./main --acl ../src/acl.json --model ../model/codeformer.om  --input ../data/file_list_1.json
   ```

   参数说明：(此模式下，输入路径为一张图片)
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单图路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可）。

   在板端会输出显示，SVP_NNN平台上性能结果如下：
   ```bash
   execution time: 726.04ms, frame rate: 1.38fps
   ```

**步骤4（可选）：使用python脚本在PC上进行数据预处理、模型推理、输出后处理（可以跟CPP版本在开发板上的推理结果进行对比）**

1. 使用python脚本进行数据预处理，将原始数据集转换为模型的输入数据。

   ```bash
   cd script # 切换到script目录
   python3 codeformer_preprocess.py \
      --file_list "../data/file_list.json" \
      --output_dir "../data/preprocess/bin" \
      --target_size 512 512
   ```

   参数说明：

   - --file_list：输入的 JSON 文件路径，内含待处理图片的路径列表（格式需符合脚本预期：包含 fileList 字段，每个元素为图片路径）。脚本会从该文件读取所有待转换的图片路径。
   - --output_dir：预处理后生成的 .bin 文件保存目录。若目录不存在，脚本会自动创建。
   - --target_size：图片缩放填充后的目标尺寸，格式为 宽 高。

2. 使用python脚本进行模型推理（可以跟CPP版本的推理结果进行对比）
   ```bash
   cd script # 切换到script目录
   python3 codeformer_infer.py \
      --preprocess_bin_dir "../data/preprocess/bin" \
      --infer_bin_dir "../data/result_pc/bin" \
      --file_list_path "../data/preprocess/file_list.txt" \
      --onnx_model_path "../model/codeformer_simplified.onnx" \
      --input_size 512 512
   ```

   参数说明：

   - --preprocess_bin_dir: 预处理后生成的二进制（.bin）文件存放目录。

   - --infer_bin_dir：推理结果（.bin 格式）的输出目录。若目录不存在，脚本会自动创建。

   - --file_list_path：python预处理阶段生成的文件列表（记录所有待推理的 .bin 文件路径）。脚本会从该文件读取待处理文件路径。

   - --onnx_model_path：待加载的 ONNX 模型文件路径。脚本会验证该文件是否存在，不存在则报错。
   
   - --input_size：模型输入张量的尺寸，格式为 高 宽（与脚本中 input_h, input_w 对应）。需与预处理阶段的目标尺寸一致，否则会导致张量形状不匹配。

3. 使用python脚本进行模型推理结果的后处理（对bin文件进行解码，保存为恢复出来的图像）
   ```bash
   cd script # 切换到script目录
   python3 codeformer_postprocess.py \
      --bin_dir "../data/result_pc/bin" \
      --img_dir "../../../../../datasets/celeba-512/celeba-512-300/hq" \
      --output_dir "../data/result_pc/img" \
      --target_size 512 512
   ```

   参数说明：

   - --bin_dir: 存放模型推理输出的 .bin 文件目录。脚本会先验证该目录是否存在，不存在则报错。

   - --img_dir：原始图片（如 JPG）的存放目录。脚本需读取图片获取原始尺寸。

   - --output_dir：后处理结果的输出目录。若目录不存在，脚本会自动创建。
   
   - --target_size：模型输入尺寸，格式为 宽 高（需与预处理、推理阶段的输入尺寸完全一致）。脚本会根据该尺寸计算图片缩放比例和填充量，用于还原边界框。

4. （可选）使用 python 脚本生成低质量图像数据集（对高清图像施加退化处理，生成对应的低质量图像用于模型测试，下载的数据集中默认已提供生成的低质量数据）
   ```bash
   cd script # 切换到脚本所在目录
   python3 generate_lq_dataset.py \
      --gt_dir "../../../../../datasets/celeba-512/celeba-512-300/hq" \
      --lq_dir "../../../../../datasets/celeba-512/celeba-512-300/lq" \
      --img_size 512
   ```

   参数说明：

   - --gt_dir: 高清图像文件夹路径。脚本会读取该目录下的图像作为原始参考图，支持多种格式（jpg、png 等）。

   - --lq_dir：低质量图像保存路径。若目录不存在，脚本会自动创建。

   - --img_size：生成的低质量图像尺寸，None 表示保持原图尺寸，默认值为 None。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 平均PSNR | 平均SSIM | 平均LPIPS |
| ----------- | ---------- | -------- | ------------------ | ------------------ | ------------------ |
| SS928V100 SVP_NNN | 1          | celeba-512-300  | 25.0426 dB      | 0.6692         | 0.2151 |

