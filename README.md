# Vehicle-Car-detection-and-multilabel-classification 车辆检测和多标签属性识别
## 使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别 (Using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize)

## 效果如下: Vehicle detection and recognition results are as follows： </br>
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_5.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_17.jpg)

</br>

## 使用方法
python Vehicle_DC -src_dir your_imgs_dir -dst_dir your_result_dir

### 程序简介 brief introductions
#### (1). 程序包含两大模块:  The program consists of two parts: first, car detection(only provides model loading and inference code, if you need training code, you can refer to [pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)); the car attributes classiyfing(provide both training and testing code, it will predict a vehicle's body color, body direction and car type)
##### <1>. 车辆检测模块： 只提供检测, 训练代码可以参考[pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train); </br>
##### <2>. 标签识别模块：包含车辆颜色、车辆朝向、车辆类型
将两个模块结合在一起，实现车辆的检测和识别，对室外智能交通信息，进行了一定程度的结构化提取。
#### (2). 程序模块详解 modules detailed introduction </br>
##### <1>. VehicleDC.py </br>
此模块主要是对汽车检测和多标签识别进行封装，输入测试目录和存放结果目录。主类Car_DC, 函数__init__主要负责汽车检测、汽车识别两个模型的初始化。
函数detect_classify负责逐张图像的检测和识别：首先对输入图像进行预处理，统一输入格式，然后输出该图像所有的车的检测框。通过函数process_predict做nms, 
坐标系转换，得到所有最终的检测框。然后程序会调用函数cls_draw_bbox。在cls_draw_bbox中逐一处理每个检测框， 首先取出检测框区域的ROI， 送入车辆多标签分类器中。分类器采用B-CNN算法，参考[模型文件链接](https://arxiv.org/pdf/1709.09890.pdf)，B-CNN主要用于训练端到端的细粒度分类。

