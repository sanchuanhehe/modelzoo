# ModelZoo


# 简介
ModelZoo，HiSpark下的开源AI模型平台，涵盖计算机视觉、自然语言处理、语音、推荐、多模态等方向的AI模型。我们仅对模型做适配和格式转化，未重新训练模型和对模型进行功能性修改。我们提供模型基于海思实操案例demo和性能表现，供开发者学习。平台的每个模型都有详细的使用指导，为方便更多开发者使用ModelZoo，我们将持续增加典型网络和相关预训练模型。如果您有任何需求，请在 **Gitee** 提交 [**issue**](https://gitee.com/HiSpark/modelzoo/issues) ，我们会及时处理。


# 目录

| 目录     | 说明    |
| -------- | ------- |
| docs     | 文档说明 |
| datasets | 数据集   |
| samples  | 模型     |
| utils    | 工具     |


# 模型列表

## 运行用户建议

**说明：**
**因使用版本差异，模型性能可能存在波动，性能仅供参考**

<table align="center">
    <tr>
    <th rowspan=1>模型</th>
    <th rowspan=1>数据集</th>
    <th rowspan=1>SVP NNN性能fps</th>
    <th rowspan=1>NNN性能fps</th>
    <th rowspan=1>输入</th>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SqueezeNet1_1">SqueezeNet1_1 </a>
    </td>
    <td>ImageNet</td>
    <td>2052.42</td>
    <td>801.62</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ShuffleNetV2">ShuffleNetV2 </a>
    </td>
    <td>ImageNet</td>
    <td>403.39</td>
    <td>255</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet50">ResNet50 </a>
    </td>
    <td>ImageNet</td>
    <td>336.66</td>
    <td>133.10</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/MobileNetV2">MobileNetV2 </a>
    </td>
    <td>ImageNet</td>
    <td>1317.222</td>
    <td>311.80</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/EfficientNetV2">EfficientNetV2 </a>
    </td>
    <td>ImageNet</td>
    <td>37.051</td>
    <td></td>
    <td>288 x 288</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/DenseNet121">DenseNet121 </a>
    </td>
    <td>ImageNet</td>
    <td>170.63</td>
    <td>108.01</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Swin-Transformer">Swin-Transformer </a>
    </td>
    <td>ImageNet</td>
    <td>38.24</td>
    <td></td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Vit-B-16">Vit-B-16</a>
    </td>
    <td>ImageNet</td>
    <td>28.56</td>
    <td></td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet101">ResNet101</a>
    </td>
    <td>ImageNet</td>
    <td>210.675</td>
    <td>83.102</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet18">ResNet18</a>
    </td>
    <td>ImageNet</td>
    <td>753.02</td>
    <td></td>
    <td>500 x 375</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Chinese-CLIP">Chinese-CLIP</a>
    </td>
    <td>CIFAR100</td>
    <td>11.142</td>
    <td></td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/Depth-Anything-v2">Depth-Anything-v2</a>
    </td>
    <td>DA-2K</td>
    <td>3.756</td>
    <td></td>
    <td>518 x 518</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/PFLD">PFLD</a>
    </td>
    <td>WFLW</td>
    <td>1631.88</td>
    <td></td>
    <td>112 x 112</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/point/SuperPoint">SuperPoint</a>
    </td>
    <td>HPatches</td>
    <td>320.60</td>
    <td></td>
    <td>240 x 320</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/UNet">UNet</a>
    </td>
    <td>carvana</td>
    <td>9.6372</td>
    <td></td>
    <td>572 x 572</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5">yolov5s</a>
    </td>
    <td>coco2017</td>
    <td>85.41</td>
    <td>35.54</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8l">yolov8l</a>
    </td>
    <td>coco2017</td>
    <td>10.393</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s">yolov8s</a>
    </td>
    <td>coco2017</td>
    <td>42.914</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s">yolo11s</a>
    </td>
    <td>coco2017</td>
    <td>42.753</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-seg">yolo11s-seg</a>
    </td>
    <td>coco2017</td>
    <td>32.732</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
</table>


# 模型适配AI引擎计划
**注：**
**适配计划如下所示，具体时间节点，以实际发布为准**

## Hi3403 
<div align="center">

|*AI引擎* |                                                        **2025Q3**                                                       |      **2025Q4**     |        **2026Q1**      |      **2026Q2**     |       **2026Q3**       |
|:-------:|:-----------------------------------------------------------------------------------------------------------------------:|:-------------------:|:----------------------:|:-------------------:|:----------------------:|
| SVP NNN | [**SqueezeNet1_1**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SqueezeNet1_1)       | **vgg16**           | **YOLOV8s-OBB**        |                     |                        |
|         | [**ShuffleNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ShuffleNetV2)         | **inception v3**    | **YOLOV9s**            |                     |                        |
|         | [**ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet50)                 | **seresnet50**      | **Yolov8s-world**      |                     |                        |
|         | [**MobileNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/MobileNetV2)           | **YOLOV3**          | **centernet**          |                     |                        |
|         | [**EfficientNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/EfficientNetV2)     | **YOLOV4**          | **deepsort**           |                     |                        |
|         | [**DenseNet121**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/DenseNet121)           | **YOLOV6s**         | **x-Stereo**           |                     |                        |
|         | [**Swin-Transformer**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Swin-Transformer) | **YOLOV7**          | **Yolov8s-seg**        |                     |                        |
|         | [**Vit-B-16**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Vit-B-16)                 | **YOLOV10s**        | **tinysan(MobileSam)** |                     |                        |
|         | [**ResNet101**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet101)               | **YOLO11s-pose**    | **Siamese network**    |                     |                        |
|         | [**ResNet18**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet18)                 | **HRNet**           | **Facenet**            |                     |                        |
|         | [**Chinese-CLIP**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Chinese-CLIP)         | **CrowdCount**      | **PaddleOCR-rec**      |                     |                        |
|         | [**Depth-Anything-v2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/Depth-Anything-v2)        | **CodeFormer**      | **PaddleOCR-det**      |                     |                        |
|         | [**PFLD**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/PFLD)                              |                     | **CRNN**               |                     |                        |
|         | [**SuperPoint**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/point/SuperPoint)                      |                     | **VDSR**               |                     |                        |
|         | [**UNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/UNet)                            |                     | **FastSpeech2**        |                     |                        |
|         | [**yolov5s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5)                       |                     | **minicpm(VLM 0.5B)**  |                     |                        |
|         | [**yolov8l**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8l)                      |                     |                        |                     |                        |
|         | [**yolov8s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s)                      |                     |                        |                     |                        |
|         | [**yolo11s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s)                      |                     |                        |                     |                        |
|         | [**yolo11s-seg**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-seg)              |                     |                        |                     |                        |
|   NNN   | [**SqueezeNet1_1**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SqueezeNet1_1)       | **vgg16**           | **CN-CLIP**            | **YOLOV8s-OBB**     | **YOLOV10s**           |
|         | [**ShuffleNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ShuffleNetV2)         | **inception v3**    | **YOLOV3**             | **centernet**       | **Yolov8s-world**      |
|         | [**ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet50)                 | **seresnet50**      | **YOLOV4**             | **deepsort**        | **tinysan(MobileSam)** |
|         | [**MobileNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/MobileNetV2)           | **EfficientNet-V2** | **YOLOV8s**            | **DepthAnythingV2** |                        |
|         | [**DenseNet121**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/DenseNet121)           | **SwinT**           | **YOLOV9s**            | **Siamese network** |                        |
|         | [**ResNet101**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet101)               | **Vit-B-16**        | **YOLOV11s**           | **Facenet**         |                        |
|         | [**yolov5s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5)                       | **YOLOV6s**         | **YOLO11s-seg**        | **PaddleOCR-rec**   |                        |
|         |                                                                                                                         | **YOLOV7**          | **YOLO11s-pose**       | **PaddleOCR-det**   |                        |
|         |                                                                                                                         | **UNET**            | **Yolov8s-seg**        | **CRNN**            |                        |
|         |                                                                                                                         | **HRNet**           | **PFLD**               | **VDSR**            |                        |
|         |                                                                                                                         | **CrowdCount**      | **CodeFormer**         | **FastSpeech2**     |                        |
|         |                                                                                                                         |                     | **SuperPoint**         |                     |                        |

</div>

## Hi3591P   
<div align="center">

| **2026Q1** | **pi0** | **graspnet** |
|:----------:|:-------:|:------------:|

</div>


# 如何贡献

本仓子模块参考目录，可以直接克隆子仓，也可以克隆主仓，在开始贡献之前，请先阅读[NOTICE](https://gitee.com/HiSpark/docs/blob/master/contribute/%E7%A4%BE%E5%8C%BA%E5%8F%82%E4%B8%8E%E8%B4%A1%E7%8C%AE%E6%8C%87%E5%8D%97.md)，谢谢！


# 免责声明

## 致ModelZoo使用者
* HiSpark ModelZoo提供的模型仅供您用于非商业目的。
* HiSpark ModelZoo仅提供公共数据集下载、模型下载和预处理脚本。这些数据集和模型不属于ModelZoo，ModelZoo也不对其质量或维护负责。请确保您具有这些数据集和模型的使用许可，如您因使用数据集和模型产生侵权纠纷，海思不承担任何责任。
* 如您在使用ModelZoo模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集、模型所有者
如果您不希望您的数据集、模型公布在ModelZoo上或希望更新ModelZoo中属于您的数据集、模型，请在Gitee提交issue，我们将根据您的issue删除或更新您的数据集、模型。衷心感谢您对ModelZoo的理解和贡献。

## License声明
HiSpark ModelZoo提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅HiSpark ModelZoo根目录。
