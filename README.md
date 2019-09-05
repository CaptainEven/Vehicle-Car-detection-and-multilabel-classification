# Vehicle-Car-detection-and-multilabel-classification 车辆检测和多标签属性识别
## 一个基于Pytorch精简的框架，使用YOLO_v3_tiny和B-CNN实现街头车辆的检测和车辆属性的多标签识别。 </br> (A precise pytorch based framework for using yolo_v3_tiny to do vehicle or car detection and attribute's multilabel classification or recognize)

## 效果如下: Vehicle detection and recognition results are as follows： </br>
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_5.jpg)
![](https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification/blob/master/test_result/test_17.jpg)
</br>

## 使用方法 Usage
python Vehicle_DC -src_dir your_imgs_dir -dst_dir your_result_dir

## 训练好的模型文件(包括车辆检测模型和多标签分类模型) trained models on baidu drive
[Tranied models-vehicle detection](https://pan.baidu.com/s/1HwTCVGTmdqkeLnqnxfNL8Q) </br>
[Tranied models-vehicle classification](https://pan.baidu.com/s/1XmzjvCgOrrVv0NWTt4Fm3g) </br>
在运行Vehicle_DC脚本之前，先下载上面的模型文件或者使用自己预先训练好的模型文件，将car_540000.weights（用于检测）放在项目根目录，将epoch_39.pth（用于多标签识别）放在根目录下的checkpoints目录下，即可使用Vehicle_DC运行。</br> 
Before running Vehicle_DC, you should download provided model files provided above or use your own pretrained models. If using models provided, you need to place car_540000.weights on root directory of this project, and place epoch_39.pth on root/checkpoints/.

### 程序简介 brief introductions
#### (1). 程序包含两大模块: </br> The program consists of two parts: first, car detection(only provides model loading and inference code, if you need training code, you can refer to [pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)); the car attributes classiyfing(provide both training and testing code, it will predict a vehicle's body color, body direction and car type)
##### <1>. 车辆检测模块： 只提供检测, 训练代码可以参考[pytorch_yolo_v3](https://github.com/eriklindernoren/PyTorch-YOLOv3#train); </br>
##### <2>. 多标签识别模块：包含车辆颜色、车辆朝向、车辆类型
将这两个模块结合在一起，可以同时实现车辆的检测和识别。以此为基础，可以对室外智能交通信息，进行一定程度的结构化信息提取。 </br>
Combining these two modules together, you can do vehicle detection and multi-label recognization at the same time. Based on this info, some structured infos in outdoor traffic scenes can be extracted.
#### (2). 程序模块详解 modules detailed introduction </br>
##### <1>. VehicleDC.py </br>
此模块是车辆检测和车辆多标签识别接口的封装，需要指定测试源目录和结果输出目录。主类Car_DC, 函数__init__主要负责汽车检测、汽车识别两个模型的初始化。
函数detect_classify负责逐张对图像进行检测和识别：首先对输入图像进行预处理，统一输入格式，然后，输出该图像所有的车的检测框。通过函数process_predict做nms, 坐标系转换，得到所有最终的检测框。然后，程序调用函数cls_draw_bbox，在cls_draw_bbox中，逐一处理每个检测框。首先，取出原图像检测框区域检测框对应的的ROI(region of interest)， 将ROI送入车辆多标签分类器。分类器调用B-CNN算法对ROI中的车辆进行多标签属性分类。参考[paper link](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf)。B-CNN主要用于训练端到端的细粒度分类。本程序对论文中的网络结构做了一定的适应性修改：为了兼顾程序的推断速度和准确度，不同于论文中采用的Vgg-16，这里的B-CNN的基础网络采用Resnet-18。</br>
This module is responsible for interface encapsulation of vehicle detection and multi-label classification. You need to specify source directory and result directory. The main class is Car_DC. The pretrained models are loaded and initiated in function init(). In function detect_classify, each input image is pre-processed to get uniformed format, then output the raw bounding boxes for further NMS calculation and coordinates tranformation. We do classification and bounding box drawing in function cls_draw_box based on bounding box ROIs. Bilinear CNN is used for fine-grained classification, and we use resnet-18 as backbone insted of vgg-16 for trade-off of accuracy and speed.
##### 耗时统计耗时 Time consuming
车辆检测： 单张图像推断耗时，在单个GTX 1050TI GPU上约18ms。 </br>
车辆多标签识别：单张图像推断耗时，在单个GTX TITAN GPU上约7ms，在单个GTX 1050TI GPU上约10ms。 </br>
Vehicle detection: sigle image inference cost 18ms on single GTX1050TI. </br>
Vehicle classification: single image inference cost 10ms on single GTX1050TI.

##### <2>. 车辆多标签数据模块（由于保密协议等原因暂时不能公开数据集） dataset.py </br>
训练、测试数据类别按照子目录存放，子目录名即label，Color_Direction_type，如Yellow_Rear_suv。 </br>
Vehicle类重载了data.Dataset的init, getitem, len方法： </br>
函数__init__负责初始化数据路径，数据标签，由于数据标签是多标签类型，故对输出向量分段计算交叉熵loss即可。 </br>
函数__getitem__负责迭代返回数据和标签，返回的数据需要经过标准化等预处理；函数__len__获取数据的总数量。

##### <3>. 车辆多标签训练、测试模块 train_vehicle_multilabel.py
此模块负责车辆多标签的训练和测试。训练过程选择交叉熵作为损失函数，需要注意的是，由于是多标签分类，故计算loss的时候需要累加各个标签的loss，其中loss = loss_color + loss_direction + 2.0 * loss_type，根据经验，将车辆类型的loss权重放到到2倍效果较好。
</br>
另一方面，训练分为两步：（1）. 冻结除了Resnet-18除全连接层之外的所有层，Fine-tune训练到收敛为止；（2）.打开第一步中冻结的所有层，进一步Fine-tune训练，调整所有层的权重，直至整个模型收敛为止。
