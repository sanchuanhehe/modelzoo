# ACT 模型

ACT（Action Chunking with Transformers）是面向机器人学习场景的端到端动作控制模型，采用 Transformer 架构进行动作表征学习。

## 模型特性

| 项目 | 说明 |
|------|------|
| 参考实现 | https://gitcode.com/openeuler/lerobot_ros2/tree/master/src/lerobot/policies/act |
| 训练数据集 | https://huggingface.co/datasets/lwh2017/grab_banana/tree/main/banana_grasp_100_320x240 |
| SVP NNN 性能 | 27 fps (Hi3403V100) |

## 输入输出

**输入数据**
| 名称 | 类型 | 维度 |
|------|------|------|
| observation.state | FP32 | 1 x 6 |
| observation.images.top | RGB_FP32 | 1 x 3 x 240 x 320 |
| observation.images.wrist | RGB_FP32 | 1 x 3 x 240 x 320 |

**输出数据**
| 名称 | 类型 | 维度 |
|------|------|------|
| feature_map_1 | FP32 | 1 x 100 x 6 |

## 子目录

- [SVP_NNN/](SVP_NNN/) - 基于 SVP NNN 算力引擎的推理实现（已适配 Hi3403V100）
- NNN/ - 基于 NNN 算力引擎的推理实现（计划中）

## 快速开始

请根据目标算力引擎选择对应的子目录，查看详细的模型转换、编译和运行说明：

- **SVP NNN 引擎**：进入 [SVP_NNN/](SVP_NNN/) 目录，参考该目录下的 README 文档
