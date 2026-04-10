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

(说明：因使用版本差异，模型性能可能存在波动，性能仅供参考)

## 具身模型

<table align="center">
    <tr>
    <th rowspan=1>模型</th>
    <th rowspan=1>数据集</th>
    <th rowspan=1>SVP NNN性能fps</th>
    <th rowspan=1>NNN性能fps</th>
    <th rowspan=1>输入</th>
    </tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/contribute">ACT</a>
    </td>
    <td><a href="https://huggingface.co/datasets/lwh2017/grab_banana/tree/main/banana_grasp_100_320x240">个人公开数据集</a></td>
    <td>27</td>
    <td></td>
    <td>1 x 6; 240 x 320 * 3; 240 x 320 * 3</td>
    </tr>
</table>

## 基础模型

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
    <td>27.338</td>
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
    <td>9.81</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Vit-B-16">Vit-B-16</a>
    </td>
    <td>ImageNet</td>
    <td>42.53</td>
    <td>22.11</td>
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
    <td>0.95</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/Depth-Anything-v2">Depth-Anything-v2</a>
    </td>
    <td>DA-2K</td>
    <td>3.756</td>
    <td>0.07</td>
    <td>518 x 518</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/PFLD">PFLD</a>
    </td>
    <td>WFLW</td>
    <td>1631.88</td>
    <td>611.95</td>
    <td>112 x 112</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/point/SuperPoint">SuperPoint</a>
    </td>
    <td>HPatches</td>
    <td>320.60</td>
    <td>65.76</td>
    <td>240 x 320</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/UNet">UNet</a>
    </td>
    <td>carvana</td>
    <td>9.6372</td>
    <td>2.14</td>
    <td>572 x 572</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5">YOLOv5s</a>
    </td>
    <td>coco2017</td>
    <td>85.41</td>
    <td>35.54</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8l">YOLOv8l</a>
    </td>
    <td>coco2017</td>
    <td>10.393</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s">YOLOv8s</a>
    </td>
    <td>coco2017</td>
    <td>42.914</td>
    <td>26.22</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s">YOLO11s</a>
    </td>
    <td>coco2017</td>
    <td>42.753</td>
    <td>23.02</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-seg">YOLO11s-seg</a>
    </td>
    <td>coco2017</td>
    <td>32.732</td>
    <td>18.18</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/InceptionV3">InceptionV3</a>
    </td>
    <td>ImageNet</td>
    <td>192.42</td>
    <td>95.48</td>
    <td>299 x 299</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov3">YOLOv3</a>
    </td>
    <td>coco2017</td>
    <td>17.327</td>
    <td>7.69</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov7">YOLOv7</a>
    </td>
    <td>coco2017</td>
    <td>13.45</td>
    <td>7.65</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/yolov4">YOLOv4</a>
    </td>
    <td>coco2017</td>
    <td>3.84</td>
    <td>2.41</td>
    <td>608 x 608</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-pose">YOLO11s-pose</a>
    </td>
    <td>coco2017</td>
    <td>38.6</td>
    <td>22.02</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov6s">YOLOv6s</a>
    </td>
    <td>coco2017</td>
    <td>39.51</td>
    <td>30.24</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov10s">YOLOv10s</a>
    </td>
    <td>coco2017</td>
    <td>34.44</td>
    <td></td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SEResNet50">SE-ResNet50</a>
    </td>
    <td>ImageNet</td>
    <td>13.56</td>
    <td>63.83</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/VGG16">VGG16</a>
    </td>
    <td>ImageNet</td>
    <td>72.27</td>
    <td>32.53</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/super_resolution/CodeFormer">CodeFormer</a>
    </td>
    <td>CelebA-512</td>
    <td>1.38</td>
    <td>0.58</td>
    <td>512 x 512</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-obb">YOLOv8s-OBB</a>
    </td>
    <td>DOTAv1</td>
    <td>18.85</td>
    <td>3.65</td>
    <td>1024 x 1024</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/LRStereo-B">LRStereo-B</a>
    </td>
    <td>KITTI15</td>
    <td>23.01</td>
    <td></td>
    <td>384 x 1248</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-seg">YOLOv8s-seg</a>
    </td>
    <td>coco2017</td>
    <td>21.46</td>
    <td>20.69</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/segmentation/TinySam">TinySAM</a>
    </td>
    <td>coco2017</td>
    <td>3.99</td>
    <td></td>
    <td>448 x 448</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/SiameseNetwork">SiameseNetwork</a>
    </td>
    <td>Faces</td>
    <td>217.44</td>
    <td>34.71</td>
    <td>100 x 100</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/FaceNet">Facenet</a>
    </td>
    <td>LFW</td>
    <td>245.73</td>
    <td>159.41</td>
    <td>160 x 160</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-rec">PaddleOCRv4-rec</a>
    </td>
    <td>XFUND</td>
    <td>90.54</td>
    <td>45.12</td>
    <td>48 x 320</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-det">PaddleOCRv4-det</a>
    </td>
    <td>XFUND</td>
    <td>14.35</td>
    <td>3.57</td>
    <td>960 x 960</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/CRNN">CRNN</a>
    </td>
    <td>Synthetic-Chinese-String-Dataset</td>
    <td>81.26</td>
    <td>9.94</td>
    <td>32 x 160</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/super_resolution/VDSR">VDSR</a>
    </td>
    <td>Set5</td>
    <td>15.09</td>
    <td>4.38</td>
    <td>516 x 516</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/count/CrowdCount">CrowdCount</a>
    </td>
    <td>CrowDensity</td>
    <td>4.61</td>
    <td>2.86</td>
    <td>224 x 224</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/pose/HRNet">HRNet</a>
    </td>
    <td>coco2017</td>
    <td>13.84</td>
    <td>5.52</td>
    <td>512 x 768</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/audio/FastSpeech2">FastSpeech2</a>
    </td>
    <td>LJSpeech</td>
    <td>19.05</td>
    <td></td>
    <td>1 x 40</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/vlm/MiniCPM">MiniCPM-4v-0.5B</a>
    </td>
    <td></td>
    <td>21.3TPS</td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/tracking/deepsort">DeepSort</a>
    </td>
    <td>MOTA16</td>
    <td>693.28</td>
    <td>70.49</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/embodied_intelligence/GraspNet">GraspNet</a>
    </td>
    <td>example_data</td>
    <td></td>
    <td>0.75</td>
    <td>720 x 1280</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/embodied_intelligence/Pi0">Pi0</a>
    </td>
    <td>lerobot/aloha_sim_transfer_cube_scripted</td>
    <td></td>
    <td>4.52</td>
    <td>480 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-world">Yolov8s-world</a>
    </td>
    <td>COCO2017</td>
    <td>41.71</td>
    <td>20.64</td>
    <td>640 x 640</td>
    </tr>
    <tr>
    <td>
    <a href="https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov9s">YOLOv9s</a>
    </td>
    <td>COCO2017</td>
    <td>35.48</td>
    <td>18.07</td>
    <td>640 x 640</td>
    </tr>
</table>






# 模型适配AI引擎计划
**注：**
**适配计划如下所示，具体时间节点，以实际发布为准**

## Hi3403 
<div align="center">

| *AI引擎* |                          **2025Q3**                          |                          **2025Q4**                          |                          **2026Q1**                          |                          **2026Q2**                          |                          **2026Q3**                          |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SVP NNN  | [**SqueezeNet1_1**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SqueezeNet1_1) | [**VGG16**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/vgg16) | [**YOLOv8s-OBB**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-obb) |                                                              |                                                              |
|          | [**ShuffleNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ShuffleNetV2) | [**InceptionV3**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/InceptionV3) | [**YOLOv9s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov9s) |                                                              |                                                              |
|          | [**ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet50) | [**SE-ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SEResNet50) | [**Yolov8s-world**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-world) |                                                              |                                                              |
|          | [**MobileNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/MobileNetV2) | [**YOLOv3**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov3) |                        **centernet**                         |                                                              |                                                              |
|          | [**EfficientNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/EfficientNetV2) | [**YOLOv4**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/yolov4) | **[DeepSort](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/tracking/deepsort)** |                                                              |                                                              |
|          | [**DenseNet121**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/DenseNet121) | [**YOLOv6s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov6s) | [**LRStereo-B**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/LRStereo-B) |                                                              |                                                              |
|          | [**Swin-Transformer**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Swin-Transformer) | [**YOLOv7**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov7) | **[Yolov8s-seg](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-seg)** |                                                              |                                                              |
|          | [**Vit-B-16**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Vit-B-16) | [**YOLOv10s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov10s) | **[TinySAM](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/segmentation/TinySam)** |                                                              |                                                              |
|          | [**ResNet101**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet101) | [**YOLO11s-pose**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-pose) | **[SiameseNetwork](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/SiameseNetwork)** |                                                              |                                                              |
|          | [**ResNet18**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet18) | [**HRNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/pose/HRNet) | [**Facenet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/FaceNet) |                                                              |                                                              |
|          | [**Chinese-CLIP**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Chinese-CLIP) | [**CrowdCount**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/count/CrowdCount) | **[PaddleOCR-rec](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-rec)** |                                                              |                                                              |
|          | [**Depth-Anything-v2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/Depth-Anything-v2) | [**CodeFormer**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/super_resolution/CodeFormer) | [**PaddleOCR-det**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-det) |                                                              |                                                              |
|          | [**PFLD**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/PFLD) |                                                              | [**CRNN**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/CRNN) |                                                              |                                                              |
|          | [**SuperPoint**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/point/SuperPoint) |                                                              | [**VDSR**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/super_resolution/VDSR) |                                                              |                                                              |
|          | [**UNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/UNet) |                                                              | **[FastSpeech2](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/audio/FastSpeech2)** |                                                              |                                                              |
|          | [**YOLOv5s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5) |                                                              | **[MiniCPM-4v-0.5B](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/vlm/MiniCPM)** |                                                              |                                                              |
|          | [**YOLOv8l**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8l) |                                                              |                                                              |                                                              |                                                              |
|          | [**YOLOv8s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s) |                                                              |                                                              |                                                              |                                                              |
|          | [**YOLO11s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s) |                                                              |                                                              |                                                              |                                                              |
|          | [**YOLO11s-seg**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-seg) |                                                              |                                                              |                                                              |                                                              |
|   NNN    | [**SqueezeNet1_1**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SqueezeNet1_1) | [**VGG16**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/vgg16) | [**Chinese-CLIP**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Chinese-CLIP) | [**YOLOv8s-OBB**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-obb) |                         **YOLOV10s**                         |
|          | [**ShuffleNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ShuffleNetV2) | [**InceptionV3**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/InceptionV3) | [**YOLOv3**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov3) |                        **centernet**                         | **[Yolov8s-world](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-world)** |
|          | [**ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet50) | [**SE-ResNet50**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/SEResNet50) | [**YOLOv4**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/yolov4) | **[DeepSort](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/tracking/deepsort)** |                    **tinysan(MobileSam)**                    |
|          | [**MobileNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/MobileNetV2) | [**EfficientNetV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/EfficientNetV2) | [**YOLOv8s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s) | [**DepthAnythingV2**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/depth/Depth-Anything-v2) |                                                              |
|          | [**DenseNet121**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/DenseNet121) | [**Swin-Transformer**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Swin-Transformer) | **[YOLOv9s](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov9s)** | [**SiameseNetwork**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/SiameseNetwork) |                                                              |
|          | [**ResNet101**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/ResNet101) | [**Vit-B-16**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/classification/Vit-B-16) | **[YOLO11s](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s)** | [**Facenet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/recognition/FaceNet) |                                                              |
|          | [**YOLOv5s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov5) | [**YOLOV6s**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov6s) | **[YOLO11s-seg](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-seg)** | [**PaddleOCR-rec**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-rec) |                                                              |
|          |                                                              | [**YOLOv7**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov7) | [**YOLO11s-pose**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolo11s-pose) | [**PaddleOCR-det**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/PaddleOCRv4-det) |                                                              |
|          |                                                              | [**UNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/UNet) | [**Yolov8s-seg**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/samples_GPL/built-in/yolov8s-seg) | [**CRNN**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/ocr/CRNN) |                                                              |
|          |                                                              | [**HRNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/pose/HRNet) | [**PFLD**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/detection/PFLD) | [**VDSR**](https://gitee.com/HiSpark/modelzoo-dev/tree/master/samples/built-in/super_resolution/VDSR) |                                                              |
|          |                                                              | [**CrowdCount**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/count/CrowdCount) | [**CodeFormer**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/super_resolution/CodeFormer) |                       **FastSpeech2**                        |                                                              |
|          |                                                              |                                                              | [**SuperPoint**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/point/SuperPoint) |                                                              |                                                              |

</div>

## Hi3591P   
<div align="center">

|                          **2026Q1**                          |                          **2026Q2**                          |       **2026Q3**       |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------: |
| [**GraspNet**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/embodied_intelligence/GraspNet) |                        |                        | 
| [**Pi0**](https://gitee.com/HiSpark/modelzoo/tree/master/samples/built-in/embodied_intelligence/Pi0) |                        |                        | 


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
