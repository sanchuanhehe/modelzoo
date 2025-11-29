# 基于EfficientNetV2网络实现图片分类
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

EfficientNetV2是一系列图像分类模型，与现有技术相比，其实现了更好的参数效率和更快的训练速度。基于EfficientNetV1，Efficient NetV2模型使用神经架构搜索（NAS）来联合优化模型大小和训练速度，并以更快的训练和推理速度进行扩展。
- 参考实现：

  ```
  url=https://github.com/rwightman/pytorch-image-models
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 288 x 288 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | output    | FP32     | 1x1000 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── file_list_1.json          //测试数据输入样例

├── script
│   ├── pth2onnx.py     // python执行脚本
|   ├── preprocess.py   // python版本前处理
|   ├── accuracy.py     // 精度评估脚本

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

| 芯片型号  | npu  | soc_version | 环境准备指导     |
| --------- | ---- | ----------- | ---------------- |
| SS928V100 | SVP_NNN | SS928V100 | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |
| SS928V100 | NNN     | OPTG        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/SS928V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |


# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码

2. 安装依赖。

   ```
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

   前往[模型下载](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth)下载对应权重，参考下载权重如下：

   [权重](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnetv2_t_agc-3620981a.pth)

2. 导出onnx文件。

    使用./script/pth2onnx.py导出动态batch的onnx文件。

    ```
    python ./script/pth2onnx.py ${pth_file} ${onnx_file}
    ```

    参数说明：

    - pth_file：权重文件。
    - onnx_file：生成 onnx 文件。建议保存为./model/efficientnetv2.onnx

    比如：`python ./script/pth2onnx.py ./model/efficientnetv2_t_agc-3620981a.pth ./model/efficientnetv2.onnx`

3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. SS928V100 SVP_NNN上的om模型转换命令

        ```
        atc --framework=5 --model="./model/efficientnetv2.onnx" --input_shape="image:1,3,288,288" --output="./model/efficientnetv2" --image_list="./data/img/ILSVRC2012_val_00000002.bin" --soc_version=SS928V100 
        ```
    2. SS928V100 NNN上的om模型转换命令

        ```
        atc --framework=5 --model="./model/efficientnetv2.onnx" --input_shape="image:1,3,288,288" --output="./model/efficientnetv2_dlite" --enable_small_channel=1 --enable_single_stream=true --soc_version=OPTG 
        ```
   
        运行成功后生成efficientnetv2.om模型文件。

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
        
        注意：如果出现命令找不到，配置环境变量。
        ```
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        ```

## 模型推理<a name="section741711594518"></a>

**步骤1：编译代码。**

1.  切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```
    mkdir -p build
    ```

2.  切换到“build“目录，执行**cmake**生成编译文件。

    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

    当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

    例如，当开发环境为X86架构，运行环境为ARM架构时，执行以下命令进行交叉编译。其中交叉编译器为aarch64-mix210-linux-gcc，SOC_VERSION根据使用npu的不同有SS928V100和OPTG两个选项，请根据运行环境选择使用。
    
    ```
    cd build
    cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gcc -DSOC_VERSION=${soc_version}
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

    ```
    ./main --acl ../src/acl.json --model ../model/efficientnetv2.om --input ../data/file_list.txt
    ```

**步骤3：输出后处理**

本例中，模型执行后，基于推理结果，输出各输入图片的top5置信度的类别标识。

1. 精度验证。

    调用脚本与数据集标签val_label.txt比对，可以获得Accuracy数据，结果保存在accuracy.txt中。

    ```
    python ./script/accuracy.py --output ${result_dir} --label ${gt_file} --result ${--result_file}
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/txt/

    - --label：真值标签文件val_label.txt所在路径。

    - --result：输出精度结果所在的位置。

    例如：  `python ./script/accuracy.py --output ./out/result/txt/ --label ../../../../datasets/ImageNet/val_label.txt --result ./out/accuracy.txt`
      
    SVP_NNN平台上精度结果：
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "81.77%"}, {"key": "Top2 accuracy", "value": "90.65%"}, {"key": "Top3 accuracy", "value": "93.57%"}, {"key": "Top4 accuracy", "value": "95.02%"}, {"key": "Top5 accuracy", "value": "95.9%"}]}
    ```
    NNN平台上精度结果：
    ```
    {"title": "Overall statistical evaluation", "value": [{"key": "Number of images", "value": "50000"}, {"key": "Number of classes", "value": "1000"}, {"key": "Top1 accuracy", "value": "82.34%"}, {"key": "Top2 accuracy", "value": "91.05%"}, {"key": "Top3 accuracy", "value": "93.94%"}, {"key": "Top4 accuracy", "value": "95.4%"}, {"key": "Top5 accuracy", "value": "96.19%"}]}
    ```
2. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/efficientnetv2.om --input ../data/file_list_1.txt --loop 100
    ```

    参数说明：(此模式下，file_list_1.txt只放一张图片)

    - --model：om模型路径。
    - --output:  后处理后结果所在位置
    - --model: 模型所在位置
    - --loop：循环执行多少次取结果， loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

    在板端会输出显示
    SVP_NNN平台上性能结果如下：
    ```
    [INFO]  time: 2698983, fps: 37.050993
    ```
    NNN平台上性能结果如下：
    ```
    [INFO]  time: 3657867, fps: 27.3383
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，EfficientNetV2模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标1（Acc@1） | 精度指标2（Acc@5） |性能(fps) |
| ----------- | ---------- | -------- | ------------------ | ------------------ |----------- |
| SS928V100 SVP_NNN | 1          | ImageNet  | 81.77%            | 95.9%             |37.051   |
| SS928V100 NNN | 1          | ImageNet  | 82.34%            | 96.19%             |27.338  |