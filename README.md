# ModelZoo

# 简介
ModelZoo，HiSpark下的开源AI模型平台，涵盖计算机视觉、自然语言处理、语音、推荐、多模态、大语言模型等方向的AI模型及其基于海思实操案例。平台的每个模型都有详细的使用指导，为方便更多开发者使用ModelZoo，我们将持续增加典型网络和相关预训练模型。如果您有任何需求，请在[Gitee]([Issues · HiSpark/ModelZoo - Gitee.com](https://gitee.com/HiSpark/modelzoo/issues))提交issue，我们会及时处理。


# 目录

| 目录                                                         | 说明                       |
| ------------------------------------------------------------ | -------------------------- |
| docs | 文档说明 |
| datasets | 数据集 |
| samples | 模型 |
| utils | 工具 |

# 如何贡献

本仓子模块参考目录，可以直接克隆子仓，也可以克隆主仓，在开始贡献之前，请先阅读[NOTICE]([contribute/社区参与贡献指南.md · HiSpark/docs - Gitee.com](https://gitee.com/HiSpark/docs/blob/master/contribute/社区参与贡献指南.md))，谢谢！

# 模型列表

## 运行用户建议

**说明：**
**因使用版本差异，模型性能可能存在波动，性能仅供参考**

<table align="center">
    <tr>
    <th rowspan=1>模型</th>
    <th rowspan=1>数据集</th>
    <th rowspan=1>3403性能</th>
    <th rowspan=1>3403性能</th>
    <th rowspan=1>3591性能</th>
    <th rowspan=1>输入shape</th>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification">Squeezenet1_1 </a>
    </td>
    <td>ImageNet</td>
    <td>1ms</td>
    <td></td>
    <td></td>
    <td>1 x 3 x 224 x 224</td>
    </tr>
</table>


# 免责声明

## 致ModelZoo使用者
1. HiSpark ModelZoo提供的模型仅供您用于非商业目的。
2. HiSpark ModelZoo仅提供公共数据集下载、模型下载和预处理脚本。这些数据集和模型不属于ModelZoo，ModelZoo也不对其质量或维护负责。请确保您具有这些数据集和模型的使用许可，如您因使用数据集和模型产生侵权纠纷，海思不承担任何责任。
3. 如您在使用ModelZoo模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集、模型所有者
如果您不希望您的数据集、模型公布在ModelZoo上或希望更新ModelZoo中属于您的数据集、模型，请在Gitee提交issue，我们将根据您的issue删除或更新您的数据集、模型。衷心感谢您对ModelZoo的理解和贡献。

## License声明
HiSpark ModelZoo提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅HiSpark ModelZoo根目录。
