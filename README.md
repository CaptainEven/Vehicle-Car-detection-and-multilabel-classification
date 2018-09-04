# Vehicle-Car-detection-and-multilabel-classification 车辆检测和多标签属性识别
## 使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别 (Using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize)

## 效果如下: Vehicle detection and recognition results are as follows： </br>
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_5.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_17.jpg)


</br>

### 程序简介 program introductions
#### (1). 程序包含两大模块:  The program consists of two parts: first, car detection(only provides model loading and inference code, if you need training code, you can refer to [pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)); the car attributes classiyfing(provide both training and testing code, it will predict a vehicle's body color, body direction and car type)
##### <1>. 车辆检测模块： 只提供检测, 训练代码可以参考[pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train); </br>
##### <2>. 标签识别模块：包含车辆颜色、车辆朝向、车辆类型
将两个模块结合在一起，实现车辆的检测和识别，对室外智能交通信息，进行了一定的结构化提取。
#### (2). 程序模块详解 program modules detailed introduction </br>

