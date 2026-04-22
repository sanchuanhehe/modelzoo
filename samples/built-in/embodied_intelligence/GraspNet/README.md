
# 基于GraspNet网络实现通用物体抓取（该模型仅用于非商用）
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
GraspNet是一种基于点云输入的多阶段抓取姿态预测模型，由抓取视角估计和抓取姿态生成两个阶段组成，通过特征提取、视角估计、局部特征提取、抓取参数估计和预测解码一系列处理，最终生成包含抓取评分、抓取宽度、抓取高度、抓取深度、旋转矩阵、抓取中心点和物体ID的预测结果，旨在解决机器人抓取任务中的6D抓取姿态估计问题。
- 参考实现：

  ```
  url=https://github.com/graspnet/graspnet-baseline
  ```

## 输入输出数据<a name="section540883920406"></a>

- 输入数据

  | 输入数据           | 数据类型         | 大小                 | 数据排布格式  |
  | ------------------ | --------------- | ------------------  | ------------ |
  | color.png          | RGB_uint8       | 1 x 3 x 720 x 1280  | NCHW         |
  | depth.png          | I_int32         | 1 x 1 x 720 x 1280  | NCHW         |
  | workspace_mask.png | 1_bool          | 1 x 1 x 720 x 1280  | NCHW         |

- 输出数据

  | 输出数据 | 数据类型  | 大小        |
  | --------| -------  | ----------  |
  | 0       | FP32     | 1 x 50 x 17 |


## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── ...            // 测试数据
├── script
│   ├── graspnet_infer.py        // python执行脚本
│   ├── graspnet_pth2onnx.py     // python执行脚本
│   ├── result_visualization     // python执行脚本

├── src
│   ├── acl.json         // 系统初始化的配置文件
│   ├── CMakeLists.txt   // 编译脚本
│   ├── main.cpp         // 资源初始化/销毁相关函数的实现文件

├── model
│   ├── ...	             // 模型文件

├── CMakeLists.txt    // 编译脚本，调用src目录下CMakeLists文件
├── *.json			  // 模型信息
├── LICENSE			  // 模型LICENSE
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>

1、本模型sample编译和运行依赖3591P CANN包，请按照板端CANN包安装说明在板端安装CANN包，并执行以下指令加载npu环境变量：

    ```
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```
2、onnx模型导出和om转换依赖cylinder_query、furthest_point_sampling和group_pointsnpu三个自定义算子，请按照 `operators/ascend_npu/README.md` 中的指引完成自定算子编译和安装，并添加环境变量(若已获取om模型文件可忽略此步骤)：

    ```
    # onnx导出和om转换前需要添加环境变量：
    export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/latest/opp/vendors/customize/
    export LD_LIBRARY_PATH=/usr/local/Ascend/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    ```

    若指定了算子run包安装路径，则通过以下指令添加环境变量：
    ```
    export ASCEND_CUSTOM_OPP_PATH=/xxx/xxx/xxx/vendors/customize/
    export LD_LIBRARY_PATH=/xxx/xxx/xxx/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    ```
3、onnx模型导出和om转换建议基于conda环境运行，conda安装步骤如下：

    ```
    # 1）下载miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py313_25.5.1-1-Linux-aarch64.sh
    chmod +x Miniconda3-py313_25.5.1-1-Linux-aarch64.sh
    bash Miniconda3-py313_25.5.1-1-Linux-aarch64.sh
    # 2）激活conda：
    source ~/.bashrc
    # 若产生/root/.bashrc: No such file or directory 报错则执行：
    cp /etc/skel/.bashrc ~/.bashrc
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
    # 3）创建新环境
    conda create -n graspnet_py38 python=3.8
    # 4）激活python 3.8 环境
    conda activate graspnet_py38
    # 单板重启之后需要执行以下指令激活miniconda
    source ~/.bashrc
    ```

# 快速上手<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 获取源码<a name="section4622531142816"></a>

1. 获取本仓源码
2. 安装依赖。
    ```
    # 建议使用 Python 3.8
    dnf install hdf5 hdf5-devel libX11 libGL
    pip3 install -r requirements.txt
    ```
3. 获取开源源码
    ```
    git clone https://github.com/graspnet/graspnet-baseline.git
    cd graspnet-baseline
    git checkout 280c215129f759ed8649cb4e89fc5dfee55f4f80
    # 应用patch
    cp /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/patch/GraspNet.patch ./
    git apply --stat GraspNet.patch    # 检查patch文件内容
    git apply --check GraspNet.patch   # 检查是否能成功应用（无报错表明可以应用patch）
    git apply --whitespace=nowarn GraspNet.patch # 应用patch
    # 安装knn_pytorch:
    cd knn
    python setup.py install
    cd ..
    # 安装graspnetAPI：
    pip3 install graspnetAPI
    ```
    若graspnetAPI安装过程中发生/tmp/xxx/python3.8/site-packages/Cython/Utils.cpython-38-aarch64-linux-gnu.so: failed to map segment from shared object 报错：
    问题原因： /tmp 目录没有可执行权限，可通过 mount -o remount, exec /tmp 查看权限；
    解决方法： vi /etc/fstab ，删除 /tmp 后面的noexec，然后重启，并重新安装graspnetAPI。

## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

    1）该模型使用 [GraspNet模型样例数据](https://github.com/graspnet/graspnet-baseline/blob/main/doc/example_data) 进行精度评估，在`GraspNet`源码的`data`目录下新建`color`、`depth`、`workspace_mask`文件夹:

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/data
    mkdir color
    mkdir depth
    mkdir workspace_mask
    ```

    2）将GraspNet模型样例数据中的RGB图`color.png`存放到`data/color`目录，深度图`depth.png`存放到`data/depth`目录，有效区掩码图`workspace_mask.png`存放到`data/workspace_mask`目录，camera参数文件`meta.mat`存放到`data`目录，文件结构如下：
   
    ```
    data
    ├── meta.mat                     // camera参数
    ├── color
    │       ├── color.png            // RGB图
    │       ……
    ├── depth
    │       ├── depth.png            // 深度图
    │       ……
    ├── workspace_mask
    │       ├── workspace_mask.png   // 有效区掩码图
            ……
     ...
    ```

    meta.mat关键参数说明：

    ```
    intrinsic_matrix[0][0]           # fx - 相机在x轴方向的焦距（像素单位），计算公式：fx = f * sx，其中f是物理焦距，sx是像素尺寸
    intrinsic_matrix[1][1]           # fy - 相机在y轴方向的焦距（像素单位），计算公式：fy = f * sy
    intrinsic_matrix[0][2]           # cx - 主点在x轴方向的坐标（像素），通常接近width/2，表示光学中心在图像中的水平位置
    intrinsic_matrix[1][2]           # cy - 主点在y轴方向的坐标（像素），通常接近height/2，表示光学中心在图像中的垂直位置
    factor_depth                     # 将深度图像像素值转换为真实世界距离的缩放因子，深度值除以这个值得到真实深度
    ```


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取权重文件。

   使用[链接](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A)下载，预训练模型权重（checkpoint-rs.tar），并将其放置于`graspnet-baseline`项目目录下

2. 导出onnx文件和om转换前的环境准备
    ```
    # 1）拷贝onnx导出脚本到graspnet-baseline项目目录下
    cd graspnet-baseline
    cp /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/script/* ./
    # 2）导出ONNX进行OM转换需要安装MagicONNX
    git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
    cd MagicONNX
    pip3 install .
    # 3）下载ais-bench工具
    # 从https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench 下载ais-bench安装包及其依赖包
    wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl
    wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl
    pip3 install aclruntime-0.0.2-cp38-cp38-linux_aarch64.whl
    pip3 install ais_bench-0.0.2-py3-none-any.whl
    ```

3. 导出onnx文件。
    注意：onnx文件导出依赖，自定义算子库run包，请务必在此之前按照`operators/ascend_npu/README.md` 中的指引完成自定算子编译和安装，并添加环境变量。
    上述准备工作完成后，在graspnet-baseline根目录下执行以下以下指令：
    ```
    python3 graspnet_pth2onnx.py
    ```
    获得graspnet.onnx文件。

4. 使用ATC工具将ONNX模型转OM模型。

    执行ATC命令。
    ```
    atc --framework=5 --model=graspnet.onnx --output=graspnet --input_format=NCHW --input_shape="point_clouds:1,20000,3" --log=error --soc_version=Ascend310P1 --op_select_implmode=high_precision --precision_mode=force_fp32
    ```

    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --input_format: 输入数据的格式。
    - --input_shape：输入数据的shape。
    - --output：输出的OM模型。
    - --soc_version：处理器型号。
    - --enable_single_stream:选择算子是高精度实现还是高性能实现。
    - --precision_mode:设置网络模型的精度模式。

    注意：
    1）如果出现命令找不到，需要配置环境变量：
    ```
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```
    2）若遇到报错fatal error: 'cstdint' file not found，可通过以下指令解决：
    ```
    dnf install gcc g++ cmake  # 若已安装gcc和g++则不必执行此指令，执行下一条指令即可
    export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux:$CPLUS_INCLUDE_PATH"
    # /usr/include/c++/12 应替换为实际g++版本和安装路径
    ```

    此sample也提供了om模型python推理脚本，指令如下：
    ```
    python3 graspnet_infer.py --model_path graspnet_linux_aarch64.om
    ```
    3）若遇到ModuleNotFoundError: No module named 'tbe'报错，可通过以下方法解决：
    ```
    cd /usr/local/Ascend/latest/aarch64-linux/lib64
    pip3 install te-0.4.0-py3-none-any.whl
    pip3 install opc_tool-0.1.0-py3-none-any.whl
    pip3 install op_compile_tool-0.1.0-py3-none-any.whl
    ```

## 模型推理<a name="section741711594518"></a>

**步骤1：准备工作。**

1.  在`GraspNet`源码的根目录下新建`model`，然后将模型转化步骤中获取的 graspnet_linux_aarch64.om 模型文件拷贝至`model`目录：

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet
    mkdir model
    cp /xxx/xxx/graspnet-baseline/graspnet_linux_aarch64.om ./model/
    ```

2. 切换到`GraspNet/script`目录下执行python脚本解析camera参数文件`meta.mat`，解析结果会作为输入关键参数：
    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/script
    python3 graspnet_parse_cam_param.py
    ```

**步骤2：编译代码。**

1.  切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet
    mkdir -p build
    ```

2.  切换到“build“目录，执行**cmake**生成编译文件。

    “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。
    
    ```
    cd build
    # 添加环境变量指定CANN包路径
    export NPU_INCLUDE_PATH=/usr/local/Ascend/latest/aarch64-linux/include/acl
    export NPU_LIB_PATH=/usr/local/Ascend/latest/aarch64-linux/lib64/
    cmake .. -DCMAKE_TOOLCHAIN_FILE=../../../common/cmake/toolchain_aarch64_openeuler.cmake
    ```
    
3.  执行**make**命令，生成的可执行文件main在“./out“目录下。

    ```
    make -j$(nproc)
    ```

**步骤3：运行应用。**

1.  以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/acl\_sample”。
2.  以运行用户登录运行环境（Host）。
3.  切换到可执行文件build所在的目录，例如“/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/build”，运行可执行文件。

    ```
    chmod +x out/main
    ./out/main --acl ../src/acl.json --model ../model/graspnet_linux_aarch64.om --input ../data/file_list_1.json
    ```

**步骤4：输出后处理**

本例中，模型执行后，基于推理结果，输出结果会保存在“/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/out/txt/“目录下

1. 精度验证。

    由于GraspNet模型在点云采样时采用随机采样，为比对模型精度结果，需要修改为固定采样，本样例提供固定采样的配置，可将“/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/data/cfg.txt“中的`is_test=0`修改为`is_test=1`，并将模型的输出结果与GPU推理结果“data/use_for_precision_align.txt“ 进行比对。具体步骤如下：

    ```
    # 1）将/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/GraspNet/data/cfg.txt中的 is_test=0 修改为 is_test=1 
    # 2）重新运行应用获取推理结果
    ./out/main --acl ../src/acl.json --model ../model/graspnet_linux_aarch64.om --input ../data/file_list_1.json
    # 3）重新运行应用获取推理结果
    python3 ../script/graspnet_test.py
    ```
      
    运行accuracy.py脚本会输出推理的精度信息：

    精度结果：
    ```
        平均绝对误差: 0.04393862560391426
        均方根误差: 0.18322496116161346
    ```
2. 验证batch_size的om模型的性能，参考命令如下：

    ```
    执行 ./out/main --acl ../src/acl.json --model ../model/graspnet_linux_aarch64.om --input ../data/file_list_1.json
    ```

    参数说明：(此模式下，file_list_1.json只放一张图片, 文件中 loop设为100)

    - --model：om模型路径。

    在板端会输出显示，性能结果如下：
    ```
     execution time: 1337.75ms, frame rate: 0.75fps
    ```

# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，GraspNet模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集         | MAE (平均绝对误差) | RMSE (均方根误差) | 性能 (FPS) |
| ----------- | ---------- | ------------  | ----------------- | ---------------- | ---------- |
| Hi3591P     | 1          | example_data  | 0.0439            | 0.1832           | 0.75       |