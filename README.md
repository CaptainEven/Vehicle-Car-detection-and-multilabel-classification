# Vehicle-Car-detection-and-multilabel-classification 车辆检测和多标签属性识别
## 使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别 (Using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize)

## 效果如下: detect and classify results are as follows： </br>
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_5.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_14.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_6.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_11.jpg)

</br>

### 程序简介 program introductions
#### (1). 程序包含两大模块: 车辆检测模块(只提供检测, 训练代码可以参考[pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)); 车辆属性的多标签识别(包含车辆颜色、车辆朝向、车辆类型)，将两个模块结合在一起，实现车辆的检测和识别，对智能交通信息，进行了一定的结构化提取。
