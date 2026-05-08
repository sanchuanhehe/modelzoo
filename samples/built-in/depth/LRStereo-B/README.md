# 基于LRStereo-B网络实现双目深度
- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)
    - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型推理](#ZH-CN_TOPIC_0000001126281700)

  - [快速开始(推荐)](#section4622531142816)
  - [安装依赖](#section183221994410)
  - [准备数据集](#section183221994411)
  - [精度&性能评估](#section741711594518)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

LRStereo-B是一个轻量且鲁棒的双目立体匹配模型。它在开源模型(Raft-Stereo)的基础上做了大量的模型结构改进和重训。具体功能为输入标定好的左右目图像以及相关的相机参数，获得左目图像对应的深度图。


## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 384 x 1248 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | 0    | FP32     | 384 x 1248 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── script
│   ├── evaluate.py    //python执行脚本

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
| Hi3403V100 | SVP_NNN | SS928V100   | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  [clang 15.0.4](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md#241%E5%AE%89%E8%A3%85clang%E4%BA%A4%E5%8F%89%E7%BC%96%E8%AF%91%E5%99%A8)  | [OpenHarmony](https://gitee.com/HiSpark/pegasus/blob/Beta-v0.9.1/docs/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97/Hi3403V100%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)   | [ss928v100_clang](https://gitee.com/HiSpark/ss928v100_clang/tree/Beta-v0.9.1/) |
| Hi3403V100 | SVP_NNN    | SS928V100        | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) |  [SVP_NNN_PC_V1.0.6.0](https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz)  |  aarch64-mix210-linux-gcc |  Linux | SS928 V100R001C02SPC022 


# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

备注：以下所有命令均在模型目录下执行

### 获取om模型文件

网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelDetail?id=i9urm5td6k00)上进行下载。

创建`model`文件夹，并将om模型文件移动到`./model`目录下。
```
mkdir -p model
```
备注：本模型只提供om模型文件。

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
    ./main --acl ../src/acl.json --model ../model/LRStereo-B_384x1248_release.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。


## 安装依赖<a name="section183221994410"></a>
```
# 建议使用python 3.9.11版本
pip install -r requirements.txt
```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型使用[KITTI15](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)验证集进行推理测试 ，用户自行获取数据集后，将文件解压并上传到当前路径的datasets路径下。数据集目录结构如下所示：

   ```
   datasets/
   |-- kitti15
   |   |-- testing
   |   |-- training
   |   |   |-- image_2
   |   |   |   |-- 000000_10.png
   |   |   |   |-- ...
   |   |   |-- image_3
   |   |   |   |-- 000000_10.png
   |   |   |   |-- ...
   |   |   |-- disp_noc_0
   |   |   |   |-- 000000_10.png
   |   |   |   |-- ...
   |   ...
   ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
  
    执行下面命令获取输入数据列表。
    
    ```
    python ./script/generate_file_list.py datasets/kitti15
    ```

## 精度&性能评估<a name="section741711594518"></a>

1.  登录到板端运行环境，切换到可执行文件main所在的目录，执行数据集上推理命令。

    如果想要跑LRStereo-B_480x640_release.om，需要同步修改./data/cfg.txt文件里面的默认尺寸和视差转距离模型路径名称。

    ```
    ./main --model ../model/LRStereo-B_384x1248_release.om --input ../data/file_list.json
    ```

2. 精度验证。

    调用脚本与数据集标签比对，可以获得Accuracy数据。

    ```
    python ./script/evaluate.py
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/bin/

    例如：  `python ./script/evaluate.py --output ./out/result/bin/`
      
    Hi3403V100 SVP_NNN平台上精度结果：
    ```
    accuray epe_meam:  1.4939332485198975  dl_mean:  8.239584974646569
    ```

3. 验证om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/LRStereo-B_384x1248_release.om --input ../data/file_list_1.json
    ```
    注意：需要将cfg.txt中的type设置为0，测试性能的时候不需要视差转距离
    参数说明：(此模式下，输入路径为一张图片)
    
    - --acl:  ACL 配置文件路径
    
    - --model: 指定待验证性能的 OM 模型文件路径。
    
    - --input： 指定输入数据的列表文件路径（此场景下为单图片路径的配置文件，注意：循环次数通过修改该文件中的loop变量即可）。

    在板端会输出显示，Hi3403V100 SVP_NNN平台上性能结果如下：
    ```
    execution time: 43.46ms, frame rate: 23.01fps
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 精度指标D1 |性能(fps) |
| ----------- | ---------- | -------- | ------------------ |----------- |
| Hi3403V100 SVP_NNN | 1          | KITTI15  |  8.239           |23.01    |