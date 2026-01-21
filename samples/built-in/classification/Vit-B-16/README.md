# vit-base-patch16模型-推理指导
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

`Transformer` 架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

- 参考实现：

```
url=https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
mode_name = [
   vit_base_patch16_224,
]
```

## 输入输出数据<a name="section540883920406"></a>

- vit_base_patch16_224 的输入数据

    | 输入数据 | 数据类型  | 大小                       | 数据排布格式  |
    | -------- | -------- | ------------------------- | ------------ |
    | input    | FLOAT32  | 1 x 3 x 224 x 224 | NCHW         |

-  vit_base_patch16_224 的输入数据

    | 输出数据 | 数据类型 | 大小                | 数据排布格式 |
    | -------- | -------- | --------          | ------------ |
    | output   | FLOAT32  | 1 x num_class     | ND           |


## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── accuracy.py     // 精度验证脚本
│   ├── vit_pth2onnx.py     // pytorch模型转onnx模型脚本
│   ├── vit_preprocess.py   // 前处理脚本
│   ├── vit_postprocess.py  // 后处理脚本


├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件
│   ├── main_dlite.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── model_cfg
│   ├── SS928V100_NNN	//模型配置文件
|   |	├── insert_op.cfg		//aipp配置文件

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

| 芯片型号  | npu     | soc_version | 环境准备指导  | cann包版本 | 编译工具链 | os  | sdk  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [openharmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  linux  | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  5.30.t11.7.b110  |  aarch64-mix210-linux-gcc |  linux  | SS928 V100R001C02SPC022                                                       |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 安装依赖。

   ```
   # 建议使用 Python 3.7.5版本
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
   ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

    2.1. Hi3403V100 SVP_NNN上的数据预处理命令
	```
	python3 ./script/vit_preprocess.py --data_path ../../../../datasets/ImageNet/val --store_path data/img --image_size 224
	```
	参数说明：
    - --data_path：原数据集所在路径。
    - --store_path：转化完后的数据保存路径， 默认在./data路径下
    - --image_size： 图像缩放后的尺寸

    2.2. Hi3403V100 NNN上的数据预处理命令

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

   下载链接可参考：https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

   模型变体较多，可按需下载。根据下表通过搜索文件名找到对应的权重文件下载地址，下载到当前目录下。

   |             模型变体|                                                                                                 文件名|
   |---------------------|------------------------------------------------------------------------------------------------------|
   | vit_base_patch16_224| B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz|

   然后将权重文件重命名为```模型变体名称.npz```
   ```bash
   # 以 vit_base_patch16_224 为例
   mv B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz vit_base_patch16_224.npz
   ```

2. 导出onnx模型

	```bash
	# bs为Batch Size，可根据需要设置，此处以1为例
	# model_name为模型变体名称，可根据需要设置，此处以 vit_base_patch16_224 为例
	python3 ./script/vit_pth2onnx.py --model_path ./model/vit_base_patch16_224.npz --save_dir model/ --model_name vit_base_patch16_224
	```
	参数说明：
	- --model_path: 模型权重npz文件路径
	- --save_dir: 保存onnx文件的目录
	- --model_name: 模型变体名称
	---
	获得```vit_base_patch16_224.onnx```文件。


3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. Hi3403V100 SVP_NNN上的om模型转换命令

        ```
        atc --framework=5                                   \
        --online_model_type=2                               \
        --output="./model/vit_base_patch16_224"             \
        --model="./model/vit_base_patch16_224.onnx"         \
        --matmul_per_channel_enable=1                       \
        --compile_mode=5                                    \
        --quant_mode=1                                      \
        --softmax_optimize_enable=1                         \
        --image_list="data/img/ILSVRC2012_val_00000101.bin" \
        --soc_version=SS928V100                             \
        --fusion_switch_file=TransformerFusion:on 
        ```

    2. Hi3403V100 NNN上的om模型转换命令

        ```
        atc --framework=5                                   \
        --output="./model/vit_base_patch16_224"             \
        --model="./model/vit_base_patch16_224.onnx"         \
        --soc_version=OPTG                                  \
        --insert_op_conf="./model_cfg/SS928V100_NNN/insert_op.cfg" \
        --enable_small_channel=1                            \
        --enable_single_stream=true                         \
        --input_shape="input:1,3,224,224"                   
        ```
    
    运行成功后生成 vit_base_patch16_224.om 模型文件。
    参数说明：
    - --model：为ONNX模型文件。
    - --framework：5代表ONNX模型。
    - --output：输出的OM模型。
    - --input_format：输入数据的格式。
    - --input_shape：输入数据的shape。
    - --compile_mode：编译模式，参数值5代表使用8bit量化数据，使用8bit量化权重,仅对CUBE算子量化，非CUBE算子使用f16格式。
    - --soc_version：处理器型号。
    - --insert_op_conf：aipp算子配置，用于输入数据处理
    - --image_list: 量化校准数据
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。


## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1.  切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```
    mkdir -p build
    ```

2.  切换到“build“目录，执行**cmake**生成编译文件。
    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    例如，开发环境为X86架构、运行环境为ARM架构时，执行以下命令进行交叉编译。交叉编译工具链按运行环境操作系统，可选toolchain_aarch64_linux.cmake或toolchain_aarch64_ohos.cmake；SOC_VERSION按算力引擎可选SS928V100或OPTG，请根据运行环境和算力引擎平台选择。
	  
	```
    cd build
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${toolchain.cmake} -DSOC_VERSION=${soc_version}
	```
    比如
    ```
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake -DSOC_VERSION=SS928V100
    ```

3.  执行**make**命令，生成的可执行文件main在“./out“目录下。

	```
	make
	```

**步骤2：运行应用。**

1.  以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2.  以运行用户登录运行环境（Host）。
3.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。
    1. Hi3403V100 SVP_NNN上的命令

    ```
    ./main --acl ../src/acl.json --model ../model/vit_base_patch16_224.om --input ../data/file_list.txt
    ```

    2. Hi3403V100 NNN上的命令

    ```
    ./main --acl ../src/acl.json --model ../model/vit_base_patch16_224.om --input ../data/file_list.json
    ```

**步骤3：输出后处理**

1. 精度验证。
    1. Hi3403V100 SVP_NNN上的命令

    调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在result_acc.json中。

    ```
	python ./script/vit_postprocess.py --save_path out/result_acc.json --input_dir ./out/result/bin --label_path ../../../../datasets/ImageNet/val_label.txt
    ```
	参数说明： 
	- --input_dir：为生成推理结果所在路径
	- --label_path：为标签数据路径
	- --save_path: 结果保存路径
     
    精度结果如下：

    SVP NNN精度验证如下：
    ```
	{'Top1 Acc': '82.48%', 'Top5 Acc': '96.60%'}
    ```

    2. Hi3403V100 NNN上的命令

    ```
    python ./script/accuracy.py --output ./out/result/txt/ --label ../../../../datasets/ImageNet/val_label.txt --result ./out/accuracy.txt
    ```

    参数说明： 
	- --input_dir：为生成推理结果所在路径
	- --label_path：为标签数据路径
	- --save_path: 结果保存路径
     
    精度结果如下：

    NNN精度验证如下：
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "84.49%"}, {"key": "Top2 accuracy", "value": "93.0%"}, {"key": "Top3 accuracy", "value": "95.47%"}, {"key": "Top4 accuracy", "value": "96.64%"}, {"key": "Top5 accuracy", "value": "97.3%"}]}    
    ```

2. 验证batch_size的om模型的性能，参考命令如下：
    1. Hi3403V100 SVP_NNN上的命令

	```
	./main --acl ../src/acl.json --model ../model/vit_base_patch16_224.om --input ../data/file_list_1.txt --loop 100
	```

    2. Hi3403V100 NNN上的命令, file_list_1.json 中loop参数设置为 100

    ```
	./main --acl ../src/acl.json --model ../model/vit_base_patch16_224.om --input ../data/file_list_1.json
	```

	参数说明：(此模式下，file_list_1.txt/.json中为一张图片路径)
	- --acl：acl.json文件的路径，默认放在src目录下。
	- --input_path:  后处理后结果所在位置
	- --model: 模型所在位置
	- --loop：循环执行多少次取结果， loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

	SS928V100 SVP_NNN 性能结果如下：
	```
    [INFO] time: 2351459, fps: 42.5268
	```

    SS928V100 NNN 性能结果如下：
	```
    [INFO] execution time: 145.93ms, frame rate: 6.85fps
	```


# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，vit-base-patch16模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标1（Acc@1） | 精度指标2（Acc@5）   | 性能（FPS）|
| ----------- | ---------- | --------| ------------------ | ------------------ |---------- |
| Hi3403V100 SVP_NNN | 1  | ImageNet  | 82.48%              | 96.60%             |   42.53  |
| Hi3403V100 NNN | 1  | ImageNet  | 84.49%              | 97.3%             |   6.85  |

