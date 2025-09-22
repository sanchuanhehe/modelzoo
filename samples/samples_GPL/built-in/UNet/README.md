# 基于**UNet**网络实现图片分类
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

UNet是由FCN改进而来的图像分割模型，其网络结构像U型，分为特征提取部分和上采样特征融合部分。
- 参考实现：

  ```
  url=https://github.com/milesial/Pytorch-UNet
  commit_id=6aa14cbbc445672d97190fec06d5568a0a004740
  model_name=UNet
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | image    | RGB_FP32 | 1 x 3 x 572 x 572 | NCHW         |

- 输出数据

  | 输出数据 | 数据类型 | 大小   |
  | -------- | -------- | ------ |
  | class    | FRGB_FP32     | 1 x 1 x  572 x 572 |



## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据

├── inc
│   ├── ...            //声明头文件

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
3. 获取开源源码
   ```
   git clone https://github.com/milesial/Pytorch-UNet.git
   cd Pytorch-UNet
   git reset --hard 6aa14cb
   cd ..
   ```

## 准备数据集<a name="section183221994411"></a>

1. 获取原始数据集。（解压命令参考tar –xvf *.tar与 unzip *.zip）

   本模型支持carvana数据集，[下载链接](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)。数据集train.zip以及train_masks.zip分别作为训练和标签文件，上传并解压到源码包路径下，目录结构如下：

   ```
   carvana/
   |-- train_hq
   |   |-- 0cdf5b5d0ce1_01.jpg
   |   ...
   |-- train_masks
   |   |-- 0cdf5b5d0ce1_01.mask.gif
   |   ...
   ```

2. 数据预处理，将原始数据集转换为模型的输入数据。
  
    执行 ../../../../utils/preprocess.py 脚本，完成数据预处理。
    
    ```
    python3 ../../../../utils/preprocess.py ${input_path} ${output_path}
    ```
   2.1 SS928V100 SVP_NNN上的数据预处理命令
   
   ```
   python ../../../../utils/preprocess.py --input_path ../../../../datasets/carvana/train_hq/ --output_path ./data/ --resize 572 --resize_mode 1 --type float32 --mean 255
   ```
   
   参数说明：
   
   - --input_path：原数据集所在路径。
   - --output_path：转化完后的数据保存路径， 默认在./data路径下


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   UNet.pth权重文件[下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/1_PyTorch_PTH/Unet/PTH/UNet.pth)。

2. 导出onnx文件。

    1. 移动pth2onnx.py至Pytorch_Unet目录，使用pth2onnx.py导出onnx文件。

         ```
         python ./Pytorch-UNet/pth2onnx.py ./model/UNet.pth ./model/UNet_dynamic_bs.onnx
         ```
         
         获得UNet_dynamic_bs.onnx文件。

    2. 使用onnxsim精简onnx文件。
         ```
         python -m onnxsim --dynamic-input-shape --input-shape="1,3,572,572" ./model/UNet_dynamic_bs.onnx ./model/UNet_dynamic_sim.onnx
         ```
         获得UNet_dynamic_sim.onnx文件。

    参数说明：

    - resume：权重文件。
    - cfg：配置文件

3. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    1. SS928V100 SVP_NNN上的om模型转换命令
        ```
        atc --framework=5 --model="./model/UNet_dynamic_sim.onnx" --input_shape="actual_input_1:1,3,572,572" --output="./model/unet" --image_list="./data/img/0cdf5b5d0ce1_01.bin" --soc_version=SS928V100
        ```
   
        运行成功后生成unet.om模型文件。

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
    需要先将opensource/opencv/lib/中的so加入到LD_LIBRARY_PATH。比如：
    LD_LIBRARY_PATH=XXX/samples/opensource/opencv/lib/:$LD_LIBRARY_PATH

    ```
    ./main --acl ../src/acl.json --model ../model/unet.om --input ../data/file_list.txt
    ```

**步骤3：输出后处理**

本例中，模型执行后，基于推理结果，输出各输入图片的top5置信度的类别标识。

1. 精度验证。

    调用脚本可以获得Accuracy数据，结果保存在accuracy.txt中。

    ```
    cp accuracy.py ./Pytorch_UNet
    python ./script/accuracy.py --output ${result_dir} --label ${gt_file} --result ${--result_file}
    ```

    参数说明：

    - --output：推理结果所在路径，默认为./out/result/txt/

    - --label：真值标签文件val_label.txt所在路径。

    - --result：输出精度结果所在的位置。

    例如：  `python ./Pytorch-UNet/accuracy.py --output ./out/result/bin/ --label ../../../../datasets/carvana/train_masks/ --result ./Pytorch_UNet/accuracy.txt`
      
    SVP_NNN平台上精度结果：
     文件中保存的是每一个图片的结果，平均结果为上述所有值求和输出：
    ```
    IOU Average: 0.9863201297157492
    ```
2. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行./main --acl ../src/acl.json --model ../model/unet.om --input ../data/file_list_1.txt --loop 100
    ```

    参数说明：(此模式下，file_list_1.txt只放一张图片)

    - --model：om模型路径。
    - --output:  后处理后结果所在位置
    - --model: 模型所在位置
    - --loop：循环执行多少次取结果， loop为1的时候第一次加载，耗时比多次执行长，建议loop取100次求平均值

    在板端会输出显示，SVP_NNN平台上性能结果如下：
    ```
     [INFO] time: 10374899, fps: 9.638646
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，Unet模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集   | 开源精度                                              | 精度指标IOU |性能 |
| ----------- | ---------- | -------- | ----------------------------------------------------- | ------------------ |------------------ |
| SS928V100_SVP_NNN | 1          | carvana | [链接](https://pytorch.org/vision/stable/models.html) | 0.9863       | 9.6386|