# Vehicle-Car-detection-and-multilabel-classification 车辆检测和多标签属性识别
## 一个基于Pytorch精简的框架，使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别。 </br> (A precise pytorch based framework for using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize)

## 效果如下: Vehicle detection and recognition results are as follows： </br>
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_5.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_17.jpg)
</br>

## 使用方法 Usage
python Vehicle_DC -src_dir your_imgs_dir -dst_dir your_result_dir

## 训练好的模型文件(包括车辆检测模型和多标签分类模型) trained models on baidu drive
[Tranied models-vehicle detection](https://pan.baidu.com/s/1OhtyRVDcodWpSR2HyhnGTw) </br>
[Tranied models-vehicle classification](https://pan.baidu.com/s/1XmzjvCgOrrVv0NWTt4Fm3g) </br>
在运行Vehicle_DC脚本之前，先下载上面的模型文件或者使用自己预先训练好的模型文件，将car_detect.weights（用于检测）放在项目根目录，将epoch_39.pth（用于多标签识别）放在根目录下的checkpoints目录下，即可使用Vehicle_DC运行。</br> 
Before running Vehicle_DC, you should download provided model files provided above or use your own pretrained models. If using models provided, you need to place car_detect.weights on root directory of this project, and place epoch_39.pth on root/checkpoints/.

### 程序简介 brief introductions
#### (1). 程序包含两大模块: </br> The program consists of two parts: first, car detection(only provides model loading and inference code, if you need training code, you can refer to [pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)); the car attributes classiyfing(provide both training and testing code, it will predict a vehicle's body color, body direction and car type)
##### <1>. 车辆检测模块： 只提供检测, 训练代码可以参考[pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train); </br>
##### <2>. 多标签识别模块：包含车辆颜色、车辆朝向、车辆类型
将这两个模块结合在一起，可以同时实现车辆的检测和识别。以此为基础，可以对室外智能交通信息，进行一定程度的结构化信息提取。 </br>
Combining these two modules together, you can do vehicle detection and multi-label recognization at the same time. Based on this info, you can extract some structured infos in outdoor scenes.
#### (2). 程序模块详解 modules detailed introduction </br>
##### <1>. VehicleDC.py </br>
此模块主要是对汽车检测和多标签识别进行封装，输入测试目录和存放结果目录。主类Car_DC, 函数__init__主要负责汽车检测、汽车识别两个模型的初始化。
函数detect_classify负责逐张对图像进行检测和识别：首先对输入图像进行预处理，统一输入格式，然后，输出该图像所有的车的检测框。通过函数process_predict做nms, 
坐标系转换，得到所有最终的检测框。然后，程序调用函数cls_draw_bbox，在cls_draw_bbox中，逐一处理每个检测框。首先，取出原图像检测框区域检测框对应的的ROI(region of interest)， 将ROI送入车辆多标签分类器。分类器调用B-CNN算法对ROI中的车辆进行多标签属性分类。参考[paper链接](https://arxiv.org/pdf/1709.09890.pdf)。B-CNN主要用于训练端到端的细粒度分类。本程序对论文中的网络结构做了一定的适应性修改：为了兼顾程序的推断速度和准确度，不同于论文中采用的Vgg-16，这里的B-CNN的基础网络采用Resnet-18。</br>
##### 耗时统计
车辆检测模块： 单张图像推断，在单个GTX 1050TI GPU上单张约18ms。 </br>
车辆多标签识别模块：单张图像推断耗时，在单个GTX TITAN GPU上约7ms，在单个GTX 1050TI GPU上单张约10ms。 </br>

##### <2>. train_vehicle_multilabel.py </br>
