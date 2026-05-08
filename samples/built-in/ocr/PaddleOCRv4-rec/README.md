
# 基于**PaddleOCRv4_rec**网络实现文字识别
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

PP-OCRv4识别模型在PP-OCRv3的基础上进一步升级。整体的框架保持了与PP-OCRv3识别模型相同的pipeline，分别进行了数据、网络结构、训练策略等方面的优化。
- 参考实现：

  ```
  url=https://github.com/PaddlePaddle/PaddleOCR.git
  url=https://github.com/frotms/PaddleOCR2Pytorch.git
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 48 x 320 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | 0      | FP32     | 1x40x6625 |

## 目录结构<a name="section540883920407"></a>
样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── pth2onnx.py     //python执行脚本

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
    [SYS] Version: [SS928V100XXXXXXXXX]
    ```

2、该模型需要以下环境

  **表 1** 版本配套表

| 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | 6.10.t01spc030b700  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  6.10.t01spc030b700  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 |
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t14.7.b140  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 |     

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=j918qkd0tk00)上进行下载。

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
    ./main --model ../model/rec.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.7.5
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   该模型使用 [XFUND数据集](https://github.com/doc-analysis/XFUND/releases/tag/v1.0) 中文集进行精度评估，在`PaddleOCRv4_rec`源码根目录下新建`datasets`文件夹，数据集放到`datasets`里，文件结构如下：
   
   ```
    datasets
        xfund
            ├── zh.val
                ├── zh_val_0.jpg
                ├── zh_val_1.jpg
                ……
            ├── zh.val.json
        ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
    1. 针对Hi3403V100 SVP_NNN平台上的om模型的预处理转换命令
        ```
        python3 ./script/xfund_process.py
        python3 ../../../../utils/generate_file_list.py datasets/paddleocr_rec_input/img/
        ```
    2. 生成量化数据
        ```
        python3 ./script/quant_rec.py
        ```

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取开源源码
   ```
   git clone https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR
   git checkout release/2.10  # 切换到所用版本
   
   git clone https://github.com/frotms/PaddleOCR2Pytorch.git
   cd PaddleOCR2Pytorch
   git checkout c799652dd04942240c0376cb6ade3cad94f7300e  # 切换到所用版本
   git apply ../ppocr.patch
   cd ..
   ```

2. 获取权重文件。

   在[链接](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar)中下载所需版本，下载后保存到PaddleOCR2Pytorch/ckpt目录下,并解压。也可以使用下述命令下载。
      ```
      cd PaddleOCR2Pytorch
      mkdir ckpt
      cd ckpt
      wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
      wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
      tar -xvf ch_PP-OCRv4_det_train.tar
      tar -xvf ch_PP-OCRv4_rec_train.tar
      cd ../
      ```
   
3. 导出onnx文件。

    1. 使用开源源码中的导出方法，在PaddleOCR2Pytorch目录下：

         ```
         cp ../script/onnx_model_sim.py ./
         python ./converter/ch_ppocr_v4_det_converter.py --yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml --src_model_path ckpt/ch_PP-OCRv4_det_train/
         
         python ./converter/ch_ppocr_v4_rec_converter.py --yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml --src_model_path ckpt/ch_PP-OCRv4_rec_train/
         
         python ./tools/infer/predict_system.py --image_dir ./doc/imgs/11.jpg --det_model_path ./ch_ptocr_v4_det_infer.pth --det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml --rec_image_shape "3,48,320" --rec_model_path ./ch_ptocr_v4_rec_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml
         
         python ./onnx_model_sim.py
         ```

    获得ch_ptocr_v4_det_simplified.onnx和ch_ptocr_v4_rec_simplified.onnx文件,将这二个文件复制到model目录下。

4. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. Hi3403V100 SVP_NNN上的om模型转换命令
        ```
        atc --model="./model/ch_ptocr_v4_rec_simplified.onnx" --online_model_type="0" --framework=5 --input_format="NCHW" --save_original_model="false" --output=./model/rec --input_type="input.1:FP32" --compile_mode=5 --soc_version=SS928V100 --image_list="./data/quant/data_rec.txt" --input_shape="input.1:1,3,48,320"
        ```

    2. Hi3403V100 NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/ch_ptocr_v4_rec_simplified.onnx" --input_format="NCHW" --input_shape="input.1:1,3,48,320" -output="./model/rec" --enable_single_stream=true --soc_version=OPTG
        ```
        运行成功后生成rec.om模型文件。

        参数说明：
        - --framework：原始框架类型，5代表ONNX模型。
        - --model：ONNX模型文件路径。
        - --input_shape：输入数据的shape。
        - --insert_op_conf：插入图像预处理的配置
        - --output：输出的OM模型路径。
        - --image_list：转换模型生成量化参数时用的校准数据。数据获取参考[准备数据集](#section183221994411)章节。
        - --enable_single_stream：推理时使用一条stream。
        - --soc_version：处理器型号。
        - --compile_mode：编译模式


## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，运行可执行文件。本例中，模型执行后，基于推理结果，输出的选框会保存再./out/result/txt/目录下

    ```
    ./main --model ../model/rec.om --input ../data/file_list.json
    ```

2. 精度验证。

    调用脚本可以获得精度数据。

    ```
    python3 ./script/rec_accuray.py --label_path "./datasets/paddleocr_rec_input/zh_val_labels.txt" --pre_file "./out/result/txt/"
    ```

    参数说明：

    - --pre_file：推理结果所在路径，默认为./out/result/txt/

    - --label_path：真值标签文件所在路径。
      
    运行rec_accuray.py脚本会输出文件，该文件中保存的是每一个图片的结果，平均结果为上述所有值求和输出：
    
    Hi3403V100 SVP_NNN平台上精度结果：
    ```
    精度结果:  0.6213833011848994 
    ```

    Hi3403V100 NNN平台上精度结果：
    ```
    精度结果:  0.65693028382474
    ```
3. 验证om模型的性能，参考命令如下：

    ```
    执行./main --model ../model/rec.om --input ../data/file_list_1.json
    ```

    参数说明：(此模式下，file_list_1.json只放一张图片)

    - --acl:  ACL 配置文件路径
    - --model：om模型路径。
    - --input：指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可,循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值）

    在板端会输出显示，Hi3403V100 SVP_NNN平台上性能结果如下：
    ```
    execution time: 11.0448ms, frame rate: 90.54fps
    ```

    在板端会输出显示，Hi3403V100 NNN平台上性能结果如下：
    ```
    execution time: 22.16ms, frame rate: 45.12fps
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，rec模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | acc | 性能（fps） |
| ----------- | ---------- | -------- | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | XFUND | 62.14%    | 90.54 |
| Hi3403V100 NNN | 1          | XFUND | 65.69%    | 45.12 |