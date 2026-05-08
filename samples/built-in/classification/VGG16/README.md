# 基于VGG16网络实现图片分类

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

VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的深度卷积神经网络，它探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3*3的小型卷积核和2*2的最大池化层，成功地构筑了16~19层深的卷积神经网络。VGGNet相比之前state-of-the-art的网络结构，错误率大幅下降，VGGNet论文中全部使用了3*3的小型卷积核和2*2的最大池化核，通过不断加深网络结构来提升性能。
VGG16包含了16个隐藏层（13个卷积层和3个全连接层）

- 参考实现：

  ```
  url=https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
  branch=master
  commit_id=78ed10cc51067f1a6bac9352831ef37a3f842784
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ----------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 224 x 224 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | class    | FP32     | 1x1000 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── pth2onnx.py     //pth转onnx脚本
│   ├── accuracy.py     //精度验证脚本

├── src
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── model_cfg
│   ├── SS928V100_NNN	//模型配置文件
|   |	  ├── insert_op.cfg		//aipp配置文件
│   ├── SS928V100_SVP_NNN	//模型配置文件
|   |	  ├── insert_op.cfg		//aipp配置文件

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
    | Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022  |
    | Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  Linux |  SS928 V100R001C02SPC022 |


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始（推荐）<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=ht96f9b50o00)上进行下载。

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
    ./main --model ../model/vgg16.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。

## 安装依赖<a name="section183221994410"></a>

```
# 建议使用 Python 3.7.5
pip3 install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[ImageNet](https://gitee.com/link?target=https%3A%2F%2Fimage-net.org%2Fdownload.php)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传数据集modelzoo/datasets/ImageNet路径下。数据集目录结构如下所示：

   ```
   ImageNet/
   |-- val
   |   |-- ILSVRC2012_val_00000001.JPEG
   |   |-- ILSVRC2012_val_00000002.JPEG
   |   |-- ILSVRC2012_val_00000003.JPEG
   |   ...
   |-- val_label.txt
   ```

2. 生成文件集file_list.json，将原始数据集图片地址转换为模型的输入数据。
  
    执行 ../../../utils/generate_file_list.py 脚本，完成数据预处理，生成的file_list.json在data目录下。
    
    ```
    python3 ../../../../utils/generate_file_list.py ${dataset_path}
    ```
    例如:
    ```
    python3 ../../../../utils/generate_file_list.py ../../../../datasets/ImageNet/val
    ```
  
   参数说明：
   - --dataset_path：原数据集所在路径。 


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。


1. 获取权重文件。

    [vgg16-397923af.pth](https://download.pytorch.org/models/vgg16-397923af.pth)

2. 导出onnx文件。

    使用./scrpit/pth2onnx.py导出onnx文件。

    ```
    python3 ./scrpit/pth2onnx.py ${pth_file} ${onnx_file}
    ```

    参数说明:

    - --pth_file：权重文件。
    - --onnx_file：生成 onnx 文件。建议保存为./model/vgg16.onnx

    比如：
    ```
    python3 ./script/pth2onnx.py ./model/vgg16-397923af.pth ./model/vgg16.onnx
    ```

3. 使用ATC工具将ONNX模型转OM模型。
    
    Hi3403V100 SVP_NNN上的om模型转换命令:
    ```
    atc --framework=5 --model="./model/vgg16.onnx" --input_shape="input_0:1,3,224,224" --insert_op_conf=./model_cfg/SS928V100_SVP_NNN/insert_op.cfg --output="./model/vgg16" --image_list="./data/image_ref_list.txt" --soc_version=SS928V100
    ```
    Hi3403V100 NNN上的om模型转换命令:
    ```
    atc --framework=5 --model="./model/vgg16.onnx" --input_shape="input_0:1,3,224,224" --insert_op_conf="./model_cfg/SS928V100_NNN/insert_op.cfg" --output="./model/vgg16" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG
    ```
    运行成功后生成vgg16.om模型文件。

    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_shape：输入数据的shape。
    - --insert_op_conf：aipp算子配置，用于输入数据处理。
    - --output：输出的OM模型。
    - --image_list: 量化校准数据。数据获取参考[准备数据集](#section183221994411)章节。
    - --compile_mode:编译模式，参数值1代表使用16bit量化数据，使用8bit量化权重。
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。
    - --soc_version：处理器型号。


## 精度&性能评估<a name="section741711594518"></a>

1. 登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。
    ```
    ./main --model ../model/vgg16.om --input ../data/file_list.json
    ```

2. 精度验证。

    执行脚本将结果文件与数据集标签val_label.txt比对，可以获得精度数据，结果保存在accuracy.txt中。

    ```
    python3 ./script/accuracy.py --output ${result_dir} --label ${gt_file} --result ${--result_file}
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/txt/

    - --label：真值标签文件val_label.txt所在路径。

    - --result：输出精度结果所在的位置。

    例如 `python3 ./script/accuracy.py --output ./out/result/txt/ --label ../../../../datasets/ImageNet/val_label.txt --result ./out/accuracy.txt`

    SVP_NNN平台上精度结果：
      ```
      {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "71.58%"}, {"key": "Top2 accuracy", "value": "82.44%"}, {"key": "Top3 accuracy", "value": "86.56%"}, {"key": "Top4 accuracy", "value": "88.8%"}, {"key": "Top5 accuracy", "value": "90.38%"}]}
      ```
    NNN平台上精度结果：

     ```
     {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "71.56%"}, {"key": "Top2 accuracy", "value": "82.33%"}, {"key": "Top3 accuracy", "value": "86.48%"}, {"key": "Top4 accuracy", "value": "88.7%"}, {"key": "Top5 accuracy", "value": "90.32%"}]}
     ```

3. 验证batch_size的om模型的性能，参考命令如下：

    ```
    ./main --model ../model/vgg16.om --input ../data/file_list_1.json
    ```

    参数说明：
    - --model：om模型文件路径。
    - --input: 输入的图像列表文件, 将file_list_1.json的loop参数设置为100

    file_list_1.json中的配置代表对一张输入图片重复推理100次，程序执行时会在板端会输出打印推理的平均时间和帧率。

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，vgg16模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   |  精度指标1（Acc@1） | 精度指标2（Acc@5） |      性能(FPS)|
| ----------- | ---------- | -------- |  ------------------ | ------------------ |-----------------|
| Hi3403V100 SVP_NNN | 1          | ImageNet |  71.58%            | 90.38%            | 72.27   |
| Hi3403V100 NNN     | 1          | ImageNet |  71.56%            | 90.32%            | 32.53   |