# 基于SqueezeNet1_1网络实现图片分类

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

Squeezenet的设计采用了卷积替换、减少卷积通道数和降采样操作后置等策略，旨在在不大幅降低模型精度的前提下，最大程度的提高运算速度。

- 参考论文：

  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

- 参考实现：

  ```
  https://github.com/pytorch/vision/blob/v0.14.0/torchvision/models/squeezenet.py#L193
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
│   ├── pth2onnx.py     //python执行文件

├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp     //资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	//模型文件

├── model_cfg
│   ├── SS928V100_NNN	//模型配置文件
|   |	├── insert_op.cfg		//aipp配置文件
│   ├── SS928V100_SVP_NNN	//模型配置文件
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

| 芯片型号  | 算力引擎   | soc_version | 环境准备指导  | CANN包版本 | 编译工具链 | 板端OS  | SDK  |
| --------- | ------- | -----------| ------------ | ---------- | ---------- | --- | ---- |
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 
| Hi3403V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  SS928 V100R001C02SPC022  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022                                                       |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 安装依赖。

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
   ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。

   1. Hi3403V100 SVP_NNN上的数据预处理命令

      ```
      python3 ./script/transformPic.py --input_path ../../../../datasets/ImageNet/val/ --output_path ./data 
      ```

   2. Hi3403V100 NNN上的的数据预处理命令

      ```
      python3 ../../../../utils/preprocess.py --input_path ../../../../datasets/ImageNet/val/ --output_path ./data --resize 256 --center_crop 224 --transpose 1
      ```
   参数说明：
   - --input_path：原数据集所在路径。
   - --output_path：转化完后的数据保存路径， 默认在./data路径下

3. 准备量化校准数据集
    按照./data/image_ref_list.txt准备好图片文件，并放置在对应目录下
    ```
    python3 ./script/transformPic_quant.py ${input_path} ${output_path}
    ```
    参数说明：
    - --input_path：原数据集所在路径。
    - --output_path：转化完后的数据保存路径， 默认在./data路径下

    例如： `python ./script/transformPic_quant.py --input_path ../../../../datasets/ImageNet/quickstart/ --output_path ./data`

## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

  1. 获取权重文件。

      [squeezenet1_1-f364aa15.pth](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Squeezenet1_1/PTH/squeezenet1_1-f364aa15.pth)

  2. 导出onnx文件。

      使用./scrpit/pth2onnx.py导出动态batch的onnx文件。

      ```
      python ./script/pth2onnx.py ${pth_file} ${onnx_file}
      ```

      参数说明：

      - --pth_file：权重文件。
      - --onnx_file：生成 onnx 文件。建议保存为./model/squeezenet.onnx
      
      比如：python ./script/pth2onnx.py ./model/squeezenet1_1-f364aa15.pth ./model/squeezenet.onnx

3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    
    1. Hi3403V100 SVP_NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/squeezenet.onnx" --input_shape="image:1,3,224,224" --output="./model/squeezenet" --image_list="./data/quant/data.txt" --soc_version=SS928V100 
        ```
    2. Hi3403V100 NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/squeezenet.onnx" --input-shape="image:1,3,224,224" --insert_op_conf="./model_cfg/SS928V100_NNN/insert_op.cfg" --output="./model/squeezenet" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG 
        ```
    运行成功后生成squeezenet.om模型文件。
  
    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_shape：输入数据的shape。
    - --insert_op_conf：aipp算子配置，用于输入数据处理。
    - --output：输出的OM模型。
    - --image_list: 量化校准数据。
    - --enable_small_channel:使能small channel优化。
    - --enable_single_stream:推理时使用一条stream。
    - --soc_version：处理器型号。


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
​
    ```
    make
    ```
​    


**步骤2：运行应用。**

1.  以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2.  以运行用户登录运行环境（Host）。
3.  切换到可执行文件main所在的目录，例如“../out”，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

4.  切换到可执行文件main所在的目录，例如“$HOME/acl\_sample/out”，运行可执行文件。

    ```
    ./main --acl ../src/acl.json --model ../model/squeezenet.om --input ../data/file_list.txt
    ```

**步骤3：输出后处理**

本例中，模型执行后，基于推理结果，输出各输入图片的top5置信度的类别标识。

1. 调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在accuracy.txt中。

    ```
    python ./script/accuracy.py --output ${result_dir} --label ${gt_file} --result ${--result_file}
    ```
    
    参数说明：
   
    - --output：推理结果所在路径，默认为./out/result/txt/
    
    - --label：真值标签文件val_label.txt所在路径。
    
    - --result：输出精度结果所在的位置。
   
    例如 `python3 ./script/accuracy.py --output ./out/result/txt/ --label ../../../../datasets/ImageNet/val_list.txt --result ./out/accuracy.txt`

    SVP_NNN平台上精度结果：
     ```
     {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "57.99%"}, {"key": "Top2 accuracy", "value": "69.43%"}, {"key": "Top3 accuracy", "value": "74.93%"}, {"key": "Top4 accuracy", "value": "78.22%"}, {"key": "Top5 accuracy", "value": "80.5%"}]}
     ```
    NNN平台上精度结果:
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "58.16%"}, {"key": "Top2 accuracy", "value": "69.65%"}, {"key": "Top3 accuracy", "value": "75.07%"}, {"key": "Top4 accuracy", "value": "78.29%"}, {"key": "Top5 accuracy", "value": "80.58%"}]}
    ```


2. 验证batch_size的om模型的性能，参考命令如下：

   ```
   ./main --acl ../src/acl.json --model ../model/squeezenet.om --input ../data/file_list_1.txt --loop 100
   ```

   参数说明：(此模式下，file_list_1.txt中为单个图片路径)

   - --model：om模型路径。
   
   - --acl: acl.json文件的路径，默认放在src目录下。
   
   - --input: 输入的图像数据列表路径
   
   - --loop： 循环执行多少次取结果，loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值
   
    在板端会输出显示，SVP_NNN平台上性能结果如下：
    ```
    [INFO] time: 50199, fps: 1992.07
    ```
    NNN平台上性能结果如下：
    ```
    [INFO] time: 124747, fps: 801.622483
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Squeezenet1_1模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   |  精度指标1（Acc@1） | 精度指标2（Acc@5） | 性能（fps） |
| ----------- | ---------- | -------- |  ------------------ | ------------------ | ------------------ |
| Hi3403V100 SVP_NNN | 1          | ImageNet |  57.99%             | 80.5%             | 1992.07     |
| Hi3403V100 NNN | 1          | ImageNet | 58.16%             | 80.58%             | 801.62       |