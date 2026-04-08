
# 基于Pi0网络实现物体抓取
- [概述](#ZH-CN_TOPIC_0000001172161501)

    - [输入输出数据](#section540883920406)
    - [目录结构](#section540883920407)

- [环境准备](#ZH-CN_TOPIC_0000001126281702)

- [模型推理](#ZH-CN_TOPIC_0000001126281700)

  - [快速开始](#section4622531142816)
  - [安装依赖](#section183221994410)
  - [准备数据集](#section183221994411)
  - [模型转化](#section741711594517)
  - [精度&性能评估](#section741711594518)
  - [模型仿真](#section741711594519)

- [模型推理性能&精度](#ZH-CN_TOPIC_0000001172201573)

  ------

# 概述<a name="ZH-CN_TOPIC_0000001172161501"></a>

  Pi0是一款视觉-语言-动作(VLA)通用机器人大模型，它基于预训练视觉语言模型(VLM)和流匹配(Flow Matching)机制，能够将自然语言指令直接转换为机器人可执行的连续动作序列，从而精准控制机器人完成复杂、高灵巧度的操作任务。

- 参考实现：
  ```
  url=https://github.com/Physical-Intelligence/openpi
  ```

## 输入输出数据<a name="section540883920406"></a>

本样例使用的示例模型为：https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube

输入输出格式定义在模型文件中的config.json中，样例示例为一路摄像头，模型输入输出格式定义如下：

- 输入数据

  | 输入数据               | 数据类型 | 大小                 | 数据排布格式  |
  | ---------------------- | ------- | ------------------  | ------------ |
  | lang_tokens            | INT64   | 1 x 48              | ND           |
  | lang_masks             | UINT8   | 1 x 48              | ND           |
  | observation.state      | FP32    | 1 x 14              | ND           |
  | observation.images.top | FP32    | 1 x 3 x 480 x 640   | NCHW         |

- 输出数据

  | 输出数据 | 数据类型  | 大小        |
  | --------| -------  | ----------  |
  | action  | FP16     | 1 x 50 x 14 |


## 目录结构<a name="section540883920407"></a>

样例代码结构如下所示。

```
├── data
│   ├── cfg.txt                    // 参数配置文件
│   ├── file_list.json             // 输入图片和token路径
│   ├── file_list_1.json           // 输入图片和token路径, 单组数据循环测试TPS
│   ├── action_stats.json          // 反归一化参数路径，包含模型action输出结果的均值和方差，若模型更新需同步更新该文件中的参数
├── script
│   ├── pi0_infer.py               // 模型离线推理python脚本
│   ├── pi0_pth2onnx.py            // onnx模型导出python脚本
│   ├── pi0_simulation_eval.py     // 模型仿真python脚本
│   ├── pi0_dataset_preprocess.py  // 数据集预处理python脚本
│   ├── pi0_accuracy_test.py       // 模型输出结果精度验证python脚本

├── src
│   ├── acl.json         // 系统初始化的配置文件
│   ├── CMakeLists.txt   // 编译脚本
│   ├── main.cpp         // 资源初始化/销毁相关函数的实现文件

├── model
│   ├── pi0.om	         // 模型文件

├── CMakeLists.txt       // 编译脚本，调用src目录下CMakeLists文件
├── *.json			     // 模型信息
├── LICENSE			     // 模型LICENSE
```

# 推理环境准备<a name="ZH-CN_TOPIC_0000001126281702"></a>
1. 该模型需要以下环境

    **表 1** 版本配套表

    | 芯片型号  | CANN包版本 | 板端OS  | SDK  |
    | --------- | ------------ | ---------- | ---------- |
    | Hi3591PV100 | 7.7.0.1.238-linux.aarch64-spc001(请联系FAE获取) | OpenEuler | 7.7.0.1.231-openEuler24.03.aarch64-rc-spc001(请联系FAE获取)  |

2. 本模型sample编译和运行依赖3591P CANN包，请按照板端CANN包安装说明在板端安装CANN包，并执行以下指令加载npu环境变量：

    ```
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```
3. (可选)onnx模型导出和om转换建议基于conda环境运行，conda安装步骤如下：

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
    conda create -n pi0_py310 python=3.10
    # 4）激活 pi0_py310 环境
    conda activate pi0_py310
    # 单板重启之后需要执行以下指令激活miniconda
    source ~/.bashrc
    ```

# 模型推理<a name="ZH-CN_TOPIC_0000001126281700"></a>

## 快速开始<a name="section4622531142816"></a>

### 获取本仓源码

    ```
    git clone https://gitee.com/HiSpark/modelzoo.git
    cd modelzoo/samples/built-in/embodied_intelligence/Pi0
    ```
    备注：以下所有命令均在模型目录下执行

### 获取om模型文件

    网站上提供转化成功的om模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelZoo)上进行下载。

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

2. 切换到`build`目录，执行**cmake**生成编译文件，生成的可执行文件main在“./out“目录下。

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

#### 运行应用

1. 运行可执行文件。本例中，模型执行后，输出推理结果。测试图片上模型推理命令参考：
    
    ```
    ./out/main --acl ../src/acl.json --model ../model/pi0.om --input ../data/file_list_1.json
    ```
    备注：若需要在数据集上进行精度评估，需要参考[安装依赖](#section183221994410)、[准备数据集](#section183221994411)和[精度&性能评估](#section741711594518)章节。


## 安装依赖<a name="section183221994410"></a>
    ```
    # 建议使用python3.8
    pip3 install -r requirements.txt
    ```


## 准备数据集<a name="section183221994411"></a>
1. 获取原始数据集。

    本样例使用 [lerobot/aloha_sim_transfer_cube_scripted数据集](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted) 进行精度评估和仿真演示:

    ```
    # 下载数据集
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/data
    pip install modelscope
    modelscope download --dataset lerobot/aloha_sim_transfer_cube_scripted --local_dir ./aloha_sim_transfer_cube_scripted
    ```

2. 执行数据集预处理脚本，从数据集中提取state、action以及image数据以便main程序使用，生成action和state数据存放在`Pi0/data/dataset/`路径下，image数据存放在`Pi0/data/images/`路径下
   
    ```
    python3 ../script/pi0_dataset_preprocess.py
    ```

3. 生成file_list.json
    main函数从file_list.json文件读取输入文本进行推理，因此我们对要推理的数据集生成匹配的file_list.json。
    在data目录下提供了file_list.json的demo样例:

    ```
    data
        ├── file_list.json（样例取lerobot/aloha_sim_transfer_cube_scripted数据集前500组数据）
        ├── file_list_1.json（单输入，但是多了loop参数，适合main函数循环推理测试性能）
        ├── cfg.txt（配置文件）
    ```


## 模型转化<a name="section741711594517"></a>

使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

1. 获取开源源码
    本样例基于Lerobot框架实现，Lerobot对Openpi的Pi0代码进行了迁移，并集成了多种机器人模型、数据集和工具。模型推理使用的Lerobot版本和模型训练使用的Lerobot版本不匹配时可能会出现config类报错，因此本样例主要基于Lerobot v0.3.4版本实现，同时指定了代码tag。

    ```
    # 1) 拉取lerobot源码并指定tag
    cd /xxx/xxx/modelzoo/samples/opensource/lerobot
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout b0923ab74b7fb7ed688ef2abbe79607f3dee390a

    # 2) 打入lerobot代码patch
    cp ../../../built-in/embodied_intelligence/Pi0/patch/Pi0.patch ./
    git apply --stat Pi0.patch    # 检查patch文件内容
    git apply --check Pi0.patch   # 检查是否能成功应用（无报错表明可以应用patch）
    git apply --whitespace=nowarn Pi0.patch # 应用patch

    # 3) 安装 lerobot
    pip install -e .

    # 4) 安装仿真依赖（若需要跑仿真）
    pip install -e ".[aloha]"

    # 5) 打入transformers库patch, 可以通过 pip show transformers 查看transformers安装路径
    export TRANSFORMERS_PATH=/xxx/xxx/pi0_py310/lib/python3.10/site-packages/transformers
    cp ../../../built-in/embodied_intelligence/Pi0/patch/modeling_gemma.patch $TRANSFORMERS_PATH/models/gemma/modeling_gemma.patch
    cd $TRANSFORMERS_PATH/models/gemma
    git apply --check -p1 modeling_gemma.patch
    git apply -p1 modeling_gemma.patch
    ```

    本样例部分python代码参照https://gitcode.com/cann/cann-recipes-embodied-intelligence实现，已在文件中标注copyright及LICENSE。

2. 获取权重文件。
    1）下载预训练VLM模型`google/paligemma-3b-pt-224`模型推理不依赖safetensors文件，可以删除模型的safetensors文件节省空间）
    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0
    mkdir -p weight
    cd weight
    # 需先通过 pip3 install modelscope 先安装 modelscope，再执行以下下载指令：
    modelscope download --model google/paligemma-3b-pt-224 --local_dir ./paligemma-3b-pt-224
    ```

    2）从[链接](https://huggingface.co/BrunoM42/pi0_aloha_transfer_cube/tree/main)下载示例模型权重，并将其放置于`/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/weight`目录下，文件结构如下：

    ```
    weight
    │    ├── paligemma-3b-pt-224
    │    ├── pi0_aloha_transfer_cube
    ```

    备注：也可以用 huggingface-cli 下载到本地:
    ```
    pip install -U huggingface_hub
    huggingface-cli download BrunoM42/pi0_aloha_transfer_cube --local-dir ./pi0_aloha_transfer_cube
    ```

3. 导出onnx文件和om转换前的准备工作
    1）运行pi0_pth2onnx.py时，会触发`google/paligemma-3b-pt-224`模型下载，由于网络环境下载可能较慢或卡顿，需要修改文件`lerobot/src/lerobot/policies/pi0/modeling_pi0.py`中第247行、`lerobot/src/lerobot/policies/pi0/modeling_pi0_all.py`中第306行的`google/paligemma-3b-pt-224`，将其替换为本地路径`/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/weight/paligemma-3b-pt-224`。
    2）确保numpy版本为1.26.0，否则执行atc转换指令将onnx转换成om文件时会产生报错
    ```
    pip install numpy==1.26.0
    ```

4. 导出onnx文件
    上述准备工作完成后，在`/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0`目录下执行以下以下指令：
    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0
    python3 ./script/pi0_pth2onnx.py --pretrained-policy-path ./weight/pi0_aloha_transfer_cube --output ./model/pi0.onnx 
    ```
    获得pi0.onnx文件。

5. 使用ATC工具将ONNX模型转OM模型

    执行ATC命令，生成om模型存放到`./model`路径下。
    ```
    atc --model=./model/pi0.onnx \
        --framework=5 \
        --output=./model/pi0 \
        --soc_version=Ascend310P1 \
        --precision_mode_v2=origin
    ```

    参数说明：
    - --framework：5代表ONNX模型。
    - --model：为ONNX模型文件。
    - --output：输出的OM模型。
    - --soc_version：处理器型号。
    - --precision_mode:设置网络模型的精度模式。

    注意：
    1）如果出现命令找不到，需要配置环境变量：
    ```
    source /usr/local/Ascend/latest/bin/setenv.bash
    ```
    2）若遇到报错`fatal error: 'cstdint' file not found`，可通过以下指令解决：
    ```
    dnf install gcc g++ cmake  # 若已安装gcc和g++则不必执行此指令，执行下一条指令即可
    export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux:$CPLUS_INCLUDE_PATH"
    # /usr/include/c++/12 应替换为实际g++版本和安装路径
    ```
    3）若遇到Numpy报错`AttributeError: 'np.float_' was removed in the NumPy 2.0 release. Use 'np.float64' instead`，需要指定numpy版本为1.26.0：
    ```
    pip install numpy==1.26.0
    ```
    4）若遇到`ModuleNotFoundError: No module named 'tbe'`报错，可通过以下方法解决：
    ```
    cd /usr/local/Ascend/latest/aarch64-linux/lib64
    pip3 install te-0.4.0-py3-none-any.whl
    pip3 install opc_tool-0.1.0-py3-none-any.whl
    pip3 install op_compile_tool-0.1.0-py3-none-any.whl
    ```


## 精度&性能评估<a name="section741711594518"></a>

1. 修改配置文件cfg.txt。

    main函数执行时，前后处理函数会读取data/cfg.txt获取一些配置参数。
    ```bash
        # 输入图像高度
        img_height=480

        # 输入图像宽度
        img_width=640

        # 模型输入最大token长度
        token_len=48

        # tokenizer模型路径
        token_model_path="../weight/paligemma-3b-pt-224/tokenizer.model"

        # 模型输入最大token长度
        action_len=14

        # 反归一化参数路径，用于模型推理结果反归一化处理，包含模型action输出结果的均值和方差，可从模型训练数据集中的meta/stats.json文件中获取。
        # 注意：若模型更新需同步更新该文件中的参数
        action_stats_path="../data/action_stats.json"

        # 模型推理结果路径，为空则不保存
        save_out_txt="../out/action/"
    ```

2. 切换到build目录，运行可执行文件:

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/build
    ./out/main --acl ../src/acl.json --model ../model/pi0.om --input ../data/file_list.json
    ```
    推理结果会保存在“../out/action”目录下

3. 精度验证。

    调用脚本与数据集结果比对，可以获得模型在lerobot/aloha_sim_transfer_cube_scripted验证集上的评估结果。

    ```
    # 例如: 将模型输出的前500组结果与数据集结果进行比对
    python ../script/pi0_accuracy_test.py --dataset-action-path ../data/dataset/ --out-action-path ../out/action/ --data-num 500
    ```

    参数说明：

    - --dataset-action-path：经pi0_dataset_preprocess.py预处理的数据集结果存放路径。

    - --out-action-path：模型输出结果存放路径。

    - --data-num：用于精度验证的数据组数。

    Hi3591P平台上精度结果:
    ```
    平均绝对误差MAE: 0.002771893749013543
    均方根误差RMSE: 0.009318939410150051
    ```

4. 验证om模型的性能，参考命令如下：

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/build
    ./out/main --acl ../src/acl.json --model ../model/pi0.om --input ../data/file_list_1.json
    ```

   参数说明：((此模式下，file_list_1.json只放一组输入数据, 文件中 loop设为1000))
   
   - --acl:  ACL 配置文件路径
   
   - --model: 指定待验证性能的 OM 模型文件路径。
   
   - --input： 指定输入数据的列表文件路径（此场景下为单输入的配置文件，注意：循环次数通过修改该文件中的loop变量即可，loop为1的时候第一次加载，耗时比多次执行长，建议loop取1000）。

   在板端会输出显示，Hi3591P平台上性能结果如下：

   ```bash
   execution time: 221.48ms, frame rate: 4.52fps
   ```

**（可选步骤）：本样例也提供了om模型python离线推理脚本**

1. om模型离线推理依赖ais_bench工具，可按照下述步骤下载和安装
    ```
    # 从 https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench 下载ais-bench安装包及其依赖包
    wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl
    wget https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl

    pip3 install aclruntime-0.0.2-cp310-cp310-linux_aarch64.whl
    pip3 install ais_bench-0.0.2-py3-none-any.whl
    ```

2. 执行推理脚本

    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0
    # 例如: 离线推理输出前500组结果
    python ./script/pi0_infer.py --model-path ./model/pi0.om --output-dir ./out/action/ --dataset-dir ./data/aloha_sim_transfer_cube_scripted --video-dir ./data/aloha_sim_transfer_cube_scripted/videos/observation.images.top/chunk-000/file-000.mp4 --extract-num 500
    ```

    参数说明：
    - --model_path：om模型文件路径。

    - --output_dir：推理结果输出目录。

    - --extract-num：模型推理从数据集中提取的数据组数，默认取前500组数据。

3. 执行精度比对脚本

    ```
    # 例如: 将模型输出的前500组结果与数据集结果进行比对
    python ./script/pi0_accuracy_test.py --dataset-action-path ./data/dataset/ --out-action-path ./out/action/ --data-num 500
    ```

## 模型仿真<a name="section741711594519"></a>
1. 安装仿真环境依赖
    ```
    dnf install mesa-libOSMesa mesa-libOSMesa-devel
    ```
2. 执行仿真脚本
    ```
    cd /xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0
    export MUJOCO_GL=osmesa
    # 执行仿真脚本
    python ./script/pi0_simulation_eval.py --policy.path=./weight/pi0_aloha_transfer_cube --env.type=aloha --env.task=AlohaTransferCube-v0 --env.episode_length=600 --eval.n_episodes=1 --eval.batch_size=1 --seed=42 --model-path ./model/pi0.om
    ```

    仿真结果保存在`./outputs/eval`目录下。

    注意：
    1）运行pi0_simulation_eval.py时，会触发`google/paligemma-3b-pt-224`模型下载，由于网络环境下载可能较慢或卡顿，需要修改文件`lerobot/src/lerobot/policies/pi0/modeling_pi0.py`中第247行的`google/paligemma-3b-pt-224`，将其替换为本地路径`/xxx/xxx/modelzoo/samples/built-in/embodied_intelligence/Pi0/weight/paligemma-3b-pt-224`，否则可能会发生`Max retries exceeded with url: /google/paligemma-3b-pt-224/resolve/main/config.json`报错，

### 注意事项

1. 模型推理输出结果需要进行反归一化处理，若仅运行示例模型，无需获取mean.pt和std.pt文件，直接使用代码中的默认值即可；

2. 若修改模型后需要重新获取反归一化参数，需要重新获取mean.pt和std.pt文件，并且修改`action_stats.json`文件中的mean和std值，可通过以下方式获取均值和方差：
    方式1：通过运行以下命令获取mean.pt和std.pt文件：

    ```
    cd /xxx/xxx/modelzoo/samples/opensource/lerobot
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=osmesa
    python ./src/lerobot/scripts/eval.py  --policy.path=/path/to/your/safetensors/file  --env.type=aloha  --env.task=AlohaTransferCube-v0  --env.episode_length=10
    ```

    方式2：mean和std值也可从模型训练数据集中的meta/stats.json文件中获取。例如本样例中使用的是aloha_sim_transfer_cube_scripted数据集中的统计参数。

# 注意：若模型更新需同步更新该文件中的参数
# 模型推理性能&精度<a name="ZH-CN_TOPIC_0000001172201573"></a>

调用ACL接口推理计算，GraspNet模型的性能和精度参考下列数据。

| 芯片型号    | Batch Size | 数据集         | MAE (平均绝对误差) | RMSE (均方根误差) | 性能 (FPS) |
| ----------- | ---------- | ------------  | ----------------- | ---------------- | ---------- |
| Hi3591P     | 1          | lerobot/aloha_sim_transfer_cube_scripted  | 0.00277           | 0.00932           | 4.52       |