# 基于ACT网络实现模仿学习
## 概述
ACT（Action Chunking with Transformers）是面向机器人学习场景的高性能端到端动作控制模型。相比传统模块化机器人控制模型，ACT采用轻量化Transformer架构作为核心骨干进行动作表征学习，结合多模态感知融合模块和时序动作优化网络，在控制精度和实时响应速度上均有显著提升。

- 参考实现：
  ```
  https://gitcode.com/openeuler/lerobot_ros2/tree/master/src/lerobot/policies/act
  ```

- 输入数据
  | 输入数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | observation.state | FP32 | 1 x 6 | - |
  | observation.images.top | RGB_FP32 | 1 x 3 x 240 x 320 | NCHW |
  | observation.images.wrist | RGB_FP32 | 1 x 3 x 240 x 320 | NCHW |

- 输出数据
  | 输出数据 | 数据类型 | 大小 | 数据排布格式 |
  | -------- | -------- | ----------- | ----------- |
  | feature_map_1 | FP32 | 1x100x6 | [N, action_n, motor_n] |

## 原理介绍
本样例涉及的关键功能点如下：
- **初始化**
  - 调用`aclInit`接口初始化ACL配置；
  - 调用`aclFinalize`接口实现ACL去初始化。
- **Device管理**
  - 调用`aclrtSetDevice`接口指定运算Device；
  - 调用`aclrtGetRunMode`接口获取运行模式，按模式差异化处理流程；
  - 调用`aclrtResetDevice`接口复位Device，回收资源。
- **Context管理**
  - 调用`aclrtCreateContext`接口创建Context；
  - 调用`aclrtDestroyContext`接口销毁Context。
- **Stream管理**
  - 调用`aclrtCreateStream`接口创建Stream；
  - 调用`aclrtDestroyStream`接口销毁Stream。
- **内存管理**
  - 调用`aclrtMalloc`接口申请Device内存；
  - 调用`aclrtFree`接口释放Device内存。
- **数据传输**
  - 调用`aclrtMemcpy`接口通过内存复制实现数据传输。
- **模型推理**
  - 调用`aclmdlLoadFromMem`接口从`*.om`文件加载模型；
  - 调用`aclmdlExecute`接口执行同步模型推理；
  - 调用`aclmdlUnload`接口卸载模型。

## 目录结构
样例代码结构如下：
```
├── ACT  // 项目根目录（基于ACT网络的机械臂模仿学习）
│
├── build  // 编译构建目录（存放中间文件、可执行程序）
│
├── data  // 输入数据与推理结果目录
│   ├── observation.images.top_240_320.bin  // 顶部摄像头图像数据（二进制）
│   ├── observation.images.wrist_240_320.bin  // 腕部摄像头图像数据（二进制）
│   ├── observation.state_240_320.bin  // 机械臂状态数据（关节角度、位姿等，二进制）
│
├── inc  // 头文件目录（存放CPP头文件）
├── model  // 模型文件目录（存放OM离线模型、配置文件）
├── out  // 输出目录（存放推理结果、日志）
│
├── script  // 辅助脚本目录
│   ├── model_test.py  // ACT模型推理测试脚本（验证功能、调试输入输出）
│
├── src  // 核心代码目录（CPP实现推理、数据处理逻辑）
│   ├── acl.json  // ACL系统初始化配置文件（指定设备、上下文等）
│   ├── CMakeLists.txt  // src目录编译脚本（定义编译规则）
│   ├── main.cpp  // 主函数入口（调度：数据加载→模型推理→结果处理）
│   ├── model_process.cpp  // 模型处理模块（OM加载、异步推理提交）
│   ├── result.txt  // 推理结果临时存储文件（记录机械臂动作输出）
│   ├── sample_process.cpp  // 样本处理模块（二进制数据加载、解析、预处理）
│
├── utils  // 工具模块目录（通用辅助功能）
│   ├── CMakeLists.txt  // utils目录编译脚本
│
├── config.yaml  // 全局配置文件（定义数据路径、模型参数、推理参数）
├── README.md  // 项目说明文档（环境依赖、编译运行步骤、目录解释）
```

## 推理环境准备
1. 查看芯片名称：
   ```bash
   cat /proc/umap/sys
   # 示例回显（芯片名为SS928V100，需自行替换实际芯片名）
   [SYS] Version: [SS928V100XXXXXXXXX]
   ```

2. 环境版本配套要求：
   | 芯片型号 | 算力引擎 | soc_version | 环境准备指导 | CANN包版本 | 编译工具链 | 板端OS | SDK |
   | -------- | ------- | ----------- | ------------ | ---------- | ---------- | --- | ---- |
   | Hi3403V100 | NNN | SS928V100 | [推理环境准备](https://gitee.com/HiSpark/modelzoo/blob/master/docs/Hi3403V100%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md) | [5.20.t6.2.b060] | aarch64-openeuler-linux-gnu-g++ | [openEuler](https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/bsp/arm64/hisilicon/hieulerpi/update.html) | SS928 V100R001C02SPC022 |

   系统驱动安装，参考：https://gitee.com/HiSpark/ss928v100_gcc/tree/Beta-v0.9.2
   
   SDK编译工具链的安装，参考：https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/getting_started/index.html#install-openeuler-embedded-sdk


## 快速上手
### 获取源码
1. 克隆参考代码仓：
   ```bash
   git clone https://gitcode.com/openeuler/lerobot_ros2.git
   cd lerobot_ros2
   ```
2. 安装依赖：参考代码仓README文档完成环境依赖安装。
3. 训练数据集下载：https://huggingface.co/datasets/lwh2017/grab_banana/tree/main/banana_grasp_100_320x240

### 模型转化
通过PyTorch将多`.safetensors`权重文件夹转为`.onnx`文件，再用ATC工具转为`.om`离线推理模型：
1. 准备权重文件：
   模型下载链接：https://huggingface.co/datasets/lwh2017/grab_banana
   ```bash
   mkdir model  # 创建模型目录
   # 将下载的模型权重文件夹（含.safetensors文件）放入model目录
   ```
2. 导出ONNX文件：
   ```bash
   cd lerobot_ros2
   # pretrained_model：权重文件夹路径；act：模型类型
   python src/lerobot/oee/export_onnx.py ../model/pretrained_model/ act
   # 移动生成的ONNX模型到model目录
   mv ./model/pretrained_model/act_ros2_simplified.onnx ../model/act_ros2_simplified.onnx
   cd ..
   ```
   **export_onnx.py参数说明**：
   | 参数/位置参数 | 说明 |
   |---------------|------|
   | 位置参数1（pretrained_model） | 必选，ACT预训练权重文件夹路径（内含.safetensors文件） |
   | 位置参数2（model_type） | 必选，模型类型，固定为"act" |
3. ATC工具转OM模型（Hi3403V100 SVP_NNN平台）：
   ```bash
   # 若无数值校准bin文件，需先通过preprocess.py生成；多文件用;分隔
   atc --model="./act_ros2_simplified.onnx" \
   --framework="5" \
   --input_format="NCHW" \
   --save_original_model="true" \
   --output="./act_ros2_simplified" \
   --soc_version=OPTG \
   --release=0
   ```
   成功后生成`act_ros2_simplified.om`文件，通过以下命令，将模型进行重命名，供main文件加载。
   ```
   mv ./model/act_ros2_simplified.om ./model/act_distill_fp32_for_mindcmd_simp_release.om
   ```
   **ATC命令核心参数说明**：
   | 参数 | 说明 |
   |------|------|
   | --model | 必选，待转换的ONNX模型文件路径 |
   | --framework | 必选，原始框架类型（5=ONNX） |
   | --output | 必选，输出OM模型的路径（无需后缀） |
   | --image_list | 必选，量化校准数据路径，格式为“输入名:文件路径;输入名:文件路径” |
   | --soc_version | 必选，处理器型号（如SS928V100） |
|
   注意：若找不到atc命令，参考“推理环境准备”配置环境。

### 模型推理
#### 步骤1：编译代码
1. 创建编译目录：
   ```bash
   mkdir -p build
   cd build
   ```
2. 生成编译文件（交叉编译示例：X86→ARM）：
   ```bash
   cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=aarch64-mix210-linux-g++ -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-I/home/Ascend/ascend-toolkit/5.20.t6.2.b060/arm64-lmixlinux200/aarch64-linux/include" -DCMAKE_CXX_LINK_FLAGS="-L/home/Ascend/ascend-toolkit/5.20.t6.2.b060/arm64-lmixlinux200/aarch64-linux/devlib -lascendcl -lpthread -ldl" -DCMAKE_CXX_COMPILER_WORKS=1
   ```
   | 参数 | 说明 |
   |------|------|				
   |-Dtarget=board |必选,指定编译目标为板端运行				
   |-DCMAKE_BUILD_TYPE=Release |可选，编译模式（Release = 生产模式，Debug = 调试模式）				
   |-DCMAKE_CXX_COMPILER=aarch64-mix210-linux-gnu-g++ | 必选，指定 C++ 交叉编译工具链为 aarch64-mix210-linux-gnu-g++				
   |-DCMAKE_C_COMPILER=/usr/bin/cc | 必选,指定 C 语言编译器路径为系统默认的 /usr/bin/cc				
   |-DCMAKE_SKIP_RPATH=TRUE |可选,禁用运行时库路径（RPATH）的生成，避免编译产物依赖特定库路径				
   |-DCMAKE_CXX_FLAGS="-I/home/Ascend/ascend-toolkit/5.20.t6.2.b060/arm64-lmixlinux200/aarch64-linux/include" |必选,C++ 编译选项：添加 Ascend（昇腾）工具链的头文件搜索路径
   |-DCMAKE_CXX_LINK_FLAGS="-L/home/Ascend/ascend-toolkit/5.20.t6.2.b060/arm64-lmixlinux200/aarch64-linux/devlib -lascendcl -lpthread -ldl" | 必选,C++ 链接	
   |-DCMAKE_CXX_COMPILER_WORKS=1 |必选,强制指定 C++ 编译器可用

3. 编译生成可执行文件：
   ```bash
   make  # 生成的main在./out目录
   ```

#### 步骤2：运行推理应用
在运行环境（板端）通过`model_test.py`调用C++可执行文件，完成数据预处理、推理、结果解析：
1. 部署文件：将样例目录上传至运行环境（Host），如`$HOME/ACT`；
2. 授权可执行文件：
   ```bash
   cd $HOME/ACT/out
   chmod +x main
   ```
3. 配置Python环境：
   ```bash
   # 创建并激活conda虚拟环境
   conda create -n act_om python=3.8 -y
   conda activate act_om
   # 安装依赖
   pip install numpy==1.24.3
   ```
4. 准备推理数据：将输入`.bin`文件放入`./data`目录；
5. 运行推理脚本：
   ```bash
   python model_test.py --image_list "bin文件路径1;bin文件路径2;..."
   ```
   **model_test.py参数说明**：
   | 参数 | 类型 | 必选 | 说明 |
   |------|------|------|------|
   | --image_list | str | 是 | 以分号分隔的bin文件路径列表，顺序需与模型输入（state/top/wrist）对应 |
6. 推理结果：保存于`./result.txt`。

#### 步骤3：验证精度和性能
1. 精度验证：
   - 测试方法
       1. 数据准备：选取20个有效测试样本，构建标准化输入批次（batch_path: ../ACT/data/batches.json）；
       2. 基准推理：通过PyTorch原生框架加载ACT模型，完成20个样本的推理，保存首个1×6 action向量作为基准值；
       3. OM推理：通过昇腾OM模型加载器执行相同20个样本的推理，提取首个1×6 action向量；
       4. 误差计算：逐样本计算OM输出与PyTorch输出的L1 Loss，统计平均值、极值等关键指标。
   - 开发环境生成目标动作：
     ```bash
     cd lerobot_ros2
     python ./src/lerobot/oee/ascend/utils/loss_compare.py \
     --device cpu --generate-target \
     --batch_path ../ACT/data/batches.json \
     --target_path ../ACT/data/target.json \
     --policy_path_act model/pretrained_model/
     ```
   - 运行环境对比误差：
     ```bash
     cd lerobot_ros2
     python ./src/lerobot/oee/ascend/utils/loss_compare.py \
     --device cpu \
     --batch_path ../ACT/data/batches.json \
     --target_path ../ACT/data/target.json \
     --policy_path_act model/pretrained_model/
     ```
   **loss_compare.py核心参数说明**：
   | 参数 | 类型 | 必选 | 说明 |
   |------|------|------|------|
   | --device | str | 是 | 运行设备，可选"cpu"/"cuda"/"npu" |
   | --generate-target | 开关 | 否 | 若指定，基于PyTorch模型生成目标动作并保存至target_path |
   | --batch_path | str | 是 | 输入batches.json文件路径（机械臂观测数据） |
   | --target_path | str | 是 | 目标动作文件（target.json）路径，用于精度对比 |
   | --policy_path_act | str | 是 | ACT预训练模型权重文件夹路径 |
   
   NNN平台精度结果：（根据实际测试补充）

    1. 测试对象
       - 模型类型：ACT动作预测模型，推理输出维度为`1×100×6`的动作矩阵（对应100个动作步，每个步为6维动作向量）；
       - 对比维度：取模型输出矩阵中首个`1×6`的action向量作为核心对比对象（实际部署中优先执行该向量）；
       - 测试样本：共20个独立测试样本，覆盖不同观测输入场景。

    2. 测试指标
        采用**L1 Loss（平均绝对误差）** 作为精度差异量化指标，计算公式为：
        $$L1_{loss} = \sum_{i=1}^6 |a_{OM,i} - a_{PyTorch,i}|$$
        其中，$a_{OM,i}$ 为OM模型输出的1×6向量第$i$维值，$a_{PyTorch,i}$ 为PyTorch原生模型输出的1×6向量第$i$维值。

    3. 测试结果
       - 单样本L1 Loss明细
  
       | 样本序号 | L1 Loss值       | 样本序号 | L1 Loss值       |
       |----------|-----------------|----------|-----------------|
       | 0        | 0.000008265178 | 10       | 0.000013351440 |
       | 1        | 0.000010410945 | 11       | 0.000003337860 |
       | 2        | 0.000002702077 | 12       | 0.000007390976 |
       | 3        | 0.000007867813 | 13       | 0.000007947286 |
       | 4        | 0.000006357829 | 14       | 0.000004688899 |
       | 5        | 0.000006675720 | 15       | 0.000004847845 |
       | 6        | 0.000008344650 | 16       | 0.000008821487 |
       | 7        | 0.000011364619 | 17       | 0.000007947286 |
       | 8        | 0.000008583069 | 18       | 0.000002940496 |
       | 9        | 0.000006198883 | 19       | 0.000000715256 |

       - 统计指标
        
       | 统计项       | 数值            |
       |--------------|-----------------|
       | 平均L1 Loss  | 0.000006937981 |
       | 最小L1 Loss  | 0.000000715256（样本 19）|
       | 最大L1 Loss  | 0.000013351440（样本 10）|
       | 中位数L1 Loss| 0.000007029395 |
       | 标准差       | 0.0000028943|

2. 性能验证
    ```bash
    cd lerobot_ros2
    python ./src/lerobot/oee/ascend/utils/loss_compare.py \
    --device cpu --generate-target \
    --batch_path ../ACT/data/batches.json \
    --target_path ../ACT/data/target.json \
    --policy_path_act model/pretrained_model/
    ```

    **性能验证说明**
    脚本运行时会输出多维度性能指标和数据规格信息，用于全面评估ACT模型推理性能，核心信息如下：
    1. **输入数据规格**（单次推理输入）：
    - 共3路输入数据，具体维度/类型/大小：
        - 输入0：形状`(1, 6)`，float32类型，元素数6，字节数24；
        - 输入1：形状`(1, 3, 240, 320)`（3通道240×320图像），float32类型，元素数230400，字节数921600；
        - 输入2：形状`(1, 3, 240, 320)`（3通道240×320图像），float32类型，元素数230400，字节数921600；
    1. **推理耗时指标**：
    - 模型核心推理时间：单次推理约**2.69秒**（仅模型前向计算耗时，不含数据传输/解析）；
    - 端到端推理时间：单次推理约**8秒**（含数据打包、C++进程通信、输出解析、张量转换全流程耗时），可通过流水线设计增加推理吞吐；
