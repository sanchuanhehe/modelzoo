# 基于 fire_detection 网络实现目标检测

## 目录
- [1. 概述](#1-概述)
  - [输入输出数据](#输入输出数据)
  - [目录结构](#目录结构)
- [2. 快速开始](#2-快速开始)
- [3. 模型训练与优化](#3-模型训练与优化)
  - [训练环境准备](#训练环境准备)
  - [训练数据集准备](#训练数据集准备)
  - [模型训练](#模型训练)
  - [模型剪枝](#模型剪枝)
  - [模型量化](#模型量化)
- [4. 模型转换 (ONNX & OM)](#4-模型转换-onnx--om)
- [5. 模型推理验证](#5-模型推理验证)
  - [推理环境准备](#推理环境准备)
  - [准备推理数据集](#准备推理数据集)
  - [编译运行与后处理](#编译运行与后处理)
- [6. 模型推理性能与精度](#6-模型推理性能与精度)

---

## 1. 概述

火焰检测网络模型是基于 YOLO11n 网络开发的适配端侧芯片的检测网络，在继承了原有 YOLO 网络模型优点的基础上，增加部分 attention 机制帮助火焰特征提取。使用剪枝以及量化的小型化操作，在保持精度情况下，大大降低网络的资源消耗与单帧推理延时。

- 参考 YOLO11 实现：
  ```
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml
  ```

### 输入输出数据

- **输入数据**

  | 输入数据 | 数据类型 | 大小             | 数据排布格式 |
  | -------- | -------- | ---------------- | ------------ |
  | images   | NV21     | 1 x 3 x 384 x 640 | YVU420SP     |

- **输出数据**

  | 输出数据 | 数据类型 | 大小        |
  | -------- | -------- | ----------- |
  | output0  | FP32     | 5 x 5040    |

### 目录结构

样例代码结构如下所示：

```text
├── data
│   ├── ...                      // 测试数据
├── script
│   ├── drawRectangle.py         // 画框验证脚本
│   ├── accuracy_optg.py         // 精度评测脚本
│   ├── pth2onnx.py              // 模型导出脚本
├── src
│   ├── acl.json                 // 系统初始化的配置文件
│   ├── CMakeLists.txt           // src下编译脚本
│   ├── main.cpp                 // 板端推理 sample 的实现文件
├── train
|   ├── fire_data.yaml           // 数据集配置文件
|   ├── fire_model.yaml          // 模型配置文件
|   ├── fire_train.yaml          // 训练配置文件
|   ├── prune.ipynb              // 剪枝 notebook
|   ├── yolo.patch               // yolo补丁文件
├── model
│   ├── fire_detectionV1.om      // 火焰检测模型文件（需自行下载或转换产生）
├── CMakeLists.txt               // 外层编译脚本
├── *.json                       // 模型其他信息与配置文件
├── LICENSE                      // 许可文件
```

---

## 2. 快速开始

> **注**：本章节包含无需自己训练和转换，直接使用现有 OM 模型并在板端运行推理的完整流程。

**步骤一：获取模型**

提供转化成功的 `om` 模型文件，可以从[网站](https://modelzoo.hispark.hisilicon.com/#/ModelZoo)上进行下载。

创建 `model` 文件夹，并将下载好的 om 模型文件移动到 `./model` 目录下。
```bash
mkdir -p model
```
*(注：若需要体验从 pth 到 om 模型的转化过程，请参考后文的[模型转换](#4-模型转换-onnx--om)章节。)*

**步骤二：编译代码 (在PC交叉编译环境执行)**

1. **环境准备**：在 PC 端安装 CANN 包，依赖 SDK 头文件和动态库在 CANN 包安装目录下，以 Hi3516CV610 为例（假设 CANN 包安装路径为 `$HOME/Ascend/`），配置环境变量：
   ```bash
   export NPU_INCLUDE_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/include/acl
   export NPU_LIB_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/lib32/stub
   ```
   在 PC 端安装交叉编译工具链（在版本包中找到 `gcc-20250305-arm-v01c02-linux-musleabi`）：
   ```bash
   cd gcc-20250305-arm-v01c02-linux-musleabi
   ./install_gcc_toolchain.sh
   export PATH=/opt/linux/x86-arm/arm-v01c02-linux-musleabi/bin:$PATH
   ```

2. 切换到 `fire_detection` 目录，创建目录用于存放编译生成的文件：
   ```bash
   mkdir build
   cd build
   ```

3. 执行 cmake 命令生成编译文件：
   ```bash
   cmake ../src -DCMAKE_TOOLCHAIN_FILE=../../../common/cmake/toolchain_aarch64_610_linux.cmake -DSOC_VERSION=Hi3516CV610
   ```

4. 执行 make 命令生成可执行文件，可执行文件 `main` 将生成于 `./out` 目录下：
   ```bash
   make
   ```

**步骤三：运行应用 (在板端环境中执行)**

1. 将整个 `modelzoo` 代码目录上传到板端运行环境。（或者使用mount命令挂载PC的modelzoo目录到板端）
2. 以运行用户登录板端运行环境。
3. 切换到可执行文件 `main` 所在的目录并赋予执行权限：
   ```bash
   chmod +x main
   ```
4. 配置可执行文件依赖动态库的搜索路径：
   ```bash
   export LD_LIBRARY_PATH=$HOME/modelzoo/samples/samples_GPL/opensource/opencv/lib/aarch64_610_linux:$LD_LIBRARY_PATH
   ```
5. 运行可执行文件进行测试推理：
   ```bash
   ./main --model ../model/fire_detectionV2.om  --input ../data/file_list.json
   ```
   *参数说明*：
   - `--model`：om 模型路径
   - `--input`：输入数据列表配置文件路径。（通过修改该文件中的 `loop` 变量可以控制循环次数。`loop` 为 1 包含加载耗时，建议设为 100 以求平均性能）。

   **结果查看**：
   推理结果（bin态）会保存在 `out/result/bin` 目录下，后处理 bbox 结果保存在 `result/txt` 目录下。
   板端输出性能示例 (SVP_NNN平台)：`[INFO]  time: 20.38ms, fps: 49.06`

---

## 3. 模型训练与优化

### 训练环境准备
1. 安装 YOLO 基础环境与依赖：
   ```bash
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   git checkout 94fac3903612fb03cab007734a8d1ce86de5376e
   
   # Python >= 3.8（建议 python3.9.0）
   pip3 install -e .
   ```
   *(注：该命令会自动安装所需的其他依赖库)*

2. 打入本仓库下针对 fire detection 的定制 Patch：
   ```bash
   # 命令执行在ultralytics目录下
   cp ~/workspace/modelzoo/samples/samples_GPL/built-in/fire_detection/train/yolo.patch .
   git apply yolo.patch
   ```

### 训练数据集准备
训练需要正负样本，以下为本模型相关的数据集参考（仅保留火焰相关标注）：
1. **fasdd 数据集** (正负样本): [链接](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)
2. **DFS 数据集** (正样本): [链接](https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset)
3. **D-fire 数据集** (正负样本): [链接](https://github.com/gaia-solutions-on-demand/DFireDataset)
4. **S2TLD 数据集** (负样本): [链接](https://github.com/Thinklab-SJTU/S2TLD)
5. **Lamp_detection** (负样本): [链接](https://universe.roboflow.com/michael-shearer/lamp-detector)

### 模型训练
相关配置文件存在于 `train` 目录下：
- **模型结构**：`train/fire_model.yaml`
- **数据配置**：`train/fire_data.yaml`
- **训练配置**：`train/fire_train.yaml`

**开启训练**：
```bash
cd train
yolo cfg=fire_train.yaml
```

### 模型剪枝
当前 `fire_model.yaml` 已经是大模型剪枝后得到的小型化结构。若需要体验或调整大模型剪枝小模型的流程，可运行配套的 `prune.ipynb`。完整的剪枝步骤包含：
1. 可剪枝模块依赖性分析
2. 剪枝敏感度分析
3. 确定各个模块的剪枝比例
4. 执行全局剪枝
5. 剪枝后的 Fine-tune (微调) 恢复精度

### 模型量化
利用 PTQ（训练后量化）以及 ATC 工具，可选择不同的量化模式实现性能提升。若全局量化精度下降严重，可利用 `mindcmd` 伪量化分析工具，挑选出相似度较低（建议阈值 < 0.95）的层保留在 FP16 计算，以此开启混精度。

针对火焰检测大模型的高精度敏感层配置示例：
```text
--hight_precision_later="/model.10/m/m.0/attn/MatMul;/model.10/m/m.0/attn/Softmax;/model.10/m/m.0/attn/MatMul_1;/model.10/m/m.0/attn/Reshape_1;/model.10/m/m.0/attn/Add;/model.10/m/m.0/Add;/model.10/m/m.0/ffn/ffn.0/conv/Conv;/model.10/m/m.0/Add_1;/model.26/cv2.2/cv2.2.2.0/conv/Conv;/model.26/cv2.2/cv2.2.1/conv/Conv;/model.26/cv2.2/cv2.2.2/Conv;/model.26/Concat_2;/model.26/Reshape_2;/model.26/Concat_3;/model.26/Split;/model.26/Sigmoid;/model.26/dfl/Reshape;/model.26/dfl/Transpose;/model.26/dfl/Softmax;/model.26/dfl/conv/Conv;/model.26/dfl/Reshape_1;/model.26/Slice;/model.26/Slice_1;/model.26/Sub_1;/model.26/Mul_2;/model.26/Concat_5"
```

---

## 4. 模型转换 (ONNX & OM)

此流程介绍如何将自行训练的 `*.pt` 模型转换为可于板端执行的 `*.om` 模型。

1. **获取待转化权重文件**
   假设已准备好 `fire_detectionV1.pt`，将其置于 `model` 文件夹内：
   ```bash
   mkdir -p model
   # 移动或下载模型到 ./model/fire_detectionV1.pt
   ```

2. **导出 ONNX 模型**
   使用提供的代码脚本导出，或通过 ultralytics 原生支持导出：
   ```bash
   cd script
   python pth2onnx.py
   cd ../
   ```
   *(或者由 yolo 直接执行：`yolo export model="fire_detectionV1.pt" format=onnx opset=13 project=export name="fire_detectionV1.onnx"`)*

3. **ATC 工具转 OM 模型**
   - **对于 Hi3516CV610 (SVP_NNN)**：
     ```bash
     atc --framework=5 --model="fire.onnx" --input_shape="images:1,3,384,640" --insert_op_conf="../model_cfg/HI3516CV610/insert_op.conf" --output="fire_detectionV1" --images_list="../data/image_ref_list.txt" --soc_version=Hi3516CV610 --compile_mode=0 --hight_precision_later="/model.10/m/m.0/attn/MatMul;/model.10/m/m.0/attn/Softmax;/model.10/m/m.0/attn/MatMul_1;/model.10/m/m.0/attn/Reshape_1;/model.10/m/m.0/attn/Add;/model.10/m/m.0/Add;/model.10/m/m.0/ffn/ffn.0/conv/Conv;/model.10/m/m.0/Add_1;/model.26/cv2.2/cv2.2.2.0/conv/Conv;/model.26/cv2.2/cv2.2.1/conv/Conv;/model.26/cv2.2/cv2.2.2/Conv;/model.26/Concat_2;/model.26/Reshape_2;/model.26/Concat_3;/model.26/Split;/model.26/Sigmoid;/model.26/dfl/Reshape;/model.26/dfl/Transpose;/model.26/dfl/Softmax;/model.26/dfl/conv/Conv;/model.26/dfl/Reshape_1;/model.26/Slice;/model.26/Slice_1;/model.26/Sub_1;/model.26/Mul_2;/model.26/Concat_5" --online_model_type=0
     ```
   *(关于ATC参数详情请查阅对应环境的手册资料；编译模式 0 为推荐量化等级以达到最佳的端侧性能)*
   *(image_ref_list中为量化校准图片，可以从数据集中挑选10-20张典型场景图片作为量化校准图片)*

---

## 5. 模型推理验证

如果您跳过了“快速开始”或者更换了板端/SOC类型，可以通过以下步骤使用评测数据集走完推理全流程。

### 推理环境准备

验证芯片名称以选用正确的包版本：
```bash
cat /proc/umap/sys
# 预期回显示例形如: [SYS] Version: Hi3516CV610
```

**版本配套表**
| 芯片型号    | 算力引擎   | CANN包版本          | 编译工具链                     | SDK版本                    |
| ----------- | ---------- | ------------------- | ------------------------------ | ------------------------- |
| Hi3516CV610 | SVP_NNN    | SVP_NN_PC_V5.0.2.3  | arm-v01c02-linux-musleabi-gcc  | Hi3516CV610R001C01SP020   |

### 准备推理数据集

1. **获取原始测试数据集**：
   下载 **fasdd 数据集** : [链接](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)，并在当前源码根目录下创建 `fasdd` 文件夹：
   ```
   fasdd
    ├── train
    │    ├── images
    │    └── labels
    └── val
    |    ├── images
    |    └── labels
    └── test
         ├── images
         └── labels
   ```

2. **数据集目录声明 (NNN 必须)**：
   ```bash
   python3 ../../../../utils/generate_file_list.py fasdd/test
   ```

### 编译运行与后处理

1. **PC端编译**：
   按您的实际运行操作系统和 SOC_VERSION 选择对应的 toolchain 执行 CMake 编译：
   ```bash
   mkdir -p build && cd build
   
   cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_610_linux.cmake -DSOC_VERSION=Hi3516CV610
   
   make
   ```

2. **板端运行**：
   登入板端，确保含有二进制执行权限。根据平台执行：
   
   - **Hi3516CV610 SVP_NNN**
     ```bash
     ./main ../../model/fire_detectionV1.om ../../fasdd/test/file_list.txt
     ```

3. **精度与可视化验证**：
   推理完毕之后，计算 mAP：
   ```bash
   # SVP_NNN 平台
   python accuracy.py --bin_dir ../coco/result/bin --img_dir ../coco/val2017 --output_json ../coco/result.json --gt_annotations ../coco/annotations/instances_val2017.json
   ```
   利用脚本可视化画出边界框（仅 SVP_NNN）：
   ```bash
   python script/drawRectangle.py --image xx.jpg --annotation result/txt/xx_result.txt
   ```

---

## 6. 模型推理性能与精度

下方表格展示了在不同芯片引擎上的 `yolo11s` 参考打底指标，您可以根据此表验证环境与配置是否正确。

| 芯片型号            | Batch Size | 测试数据集 | AP（IoU=0.50） | AP（IoU=0.50:0.95） | 性能（fps） |
| ------------------- | ---------- | ---------- | -------------- | ------------------- | ----------- |
| Hi3516CV610 SVP_NNN  | 1          | fasdd   | 81.4%          | 53.0%               | 49.06      |
