# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import pickle
import shutil
import random
import argparse

# from darknet_util import *
# from darknet import Darknet
# from preprocess import prep_image, process_img, inp_to_image

# import torch
# import torchvision
# import paramiko

# import cv2
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.image import AxesImage
# from scipy.spatial.distance import cityblock
# from tqdm import tqdm

# 为了使用matplotlib正确显示中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

# use_cuda = True  # True
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device(
#     'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

# if use_cuda:
#     torch.manual_seed(0)
#     torch.cuda.manual_seed_all(0)
# print('=> device: ', device)


# 全局变量
# root = 'e:/pick_car_roi'                           # 测试数据路径

# model_path = 'e:/epoch_96.pth'
# attrib_path = 'e:/vehicle_attributes.pkl'          # 属性文件路径


def letterbox_image(img, inp_dim):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) //
           2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


class Cropper(object):
    """
    GUI交互, 通过鼠标键盘交互, 实现矩形抠图和拷贝
    """

    def __init__(self,
                 root,
                 dst_dir,
                 is_resume=False):
        """
        初始化资源
        @param root: 原图所在目录路径
        """
        object.__init__(self)

        if not os.path.exists(root):
            print('[Err]: empty src dir.')
            return

        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

        self.root = root
        self.imgs_path = [os.path.join(self.root, x)
                          for x in os.listdir(self.root)]
        self.dst_dir = dst_dir  # 选取的ROI存放目录

        self.ROI = None

        self.clip_id = 0

        # 加载断点
        print('=> is resume: ', is_resume)
        if is_resume == 1:
            self.idx = pickle.load(open('clip_idx.pkl', 'rb'))
            self.label_dict = pickle.load(open('label_dict.pkl', 'rb'))
            print('=> resume from @%d, remain %d files to be classified.' %
                  (self.idx, len(self.imgs_path) - self.idx - 1))
        elif is_resume == 0:
            self.idx = 0  # 初始化序号
            self.label_dict = {}
            print('=> resume from @%d, remain %d files to be classified.' %
                  (self.idx, len(self.imgs_path)))
        else:
            print('=> [Err]: unrecognized flag.')
            return

        # 初始化车辆多标签分类管理器
        # self.manager = Manager(model_path=model_path,
        #                        attrib_path=attrib_path)

        # 创建绘图
        self.fig = plt.figure(figsize=(14.0, 8.0))
        self.ax = self.fig.add_subplot(111)

        # 为绘图添加鼠标和键盘callback
        self.cid_scroll = self.fig.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        self.cid_btn_press = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_btn_press)
        self.cid_btn_release = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_btn_release)
        self.cid_mouse_move = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_mouse_motion)
        self.cid_key_release = self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

        # 初始化鼠标按键为False
        self.is_btn_press = False

        # 初始化鼠标点击次数为0
        self.is_rect_ready = False

        # 读取图像
        try:
            img_path = self.imgs_path[self.idx]
            print(img_path)
        except Exception as e:
            print(e)
            return
        self.img = Image.open(img_path)

        # 绘制光标定位
        self.cursor = Cursor(self.ax,
                             useblit=True,
                             color='red',
                             linewidth=1)

        # 初始化矩形框
        self.init_rect()

        # 绘制第一张图
        ax_img = self.ax.imshow(self.img, picker=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.title(img_path)
        plt.tight_layout()
        plt.show()

        self.fig.canvas.draw()

    def init_rect(self):
        """
        初始化矩形框
        """
        self.is_btn_press = False
        self.is_rect_ready = False
        self.rect = Rectangle((0, 0), 1, 1,
                              edgecolor='b',
                              linewidth=1,
                              facecolor='none')
        self.x_0, self.y_0, self.x_1, self.y_1 = 0, 0, 0, 0
        self.ax.add_patch(self.rect)

    def exit(self):
        """
        退出处理
        """
        # 关闭图像
        self.ax.cla()
        self.fig.clf()
        plt.close()

        # 保存断点
        pickle.dump(self.idx, open('clip_idx.pkl', 'wb'))
        pickle.dump(self.label_dict, open('label_dict.pkl', 'wb'))
        print('=> save checkpoint idx @%d, and exit.' % self.idx)

    def update_fig(self):
        """
        更新绘图
        """
        if self.idx < len(self.imgs_path):
            # 释放上一帧缓存
            self.ax.cla()

            # 重绘一帧图像
            self.img = Image.open(self.imgs_path[self.idx])  # 读取图像
            ax_img = self.ax.imshow(self.img, picker=True)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            plt.title(str(self.idx) + ': ' + self.imgs_path[self.idx])
            plt.tight_layout()

            # 重新初始化矩形框
            self.init_rect()

            self.fig.canvas.draw()

    def draw_rect(self, event):
        self.x_1 = event.xdata
        self.y_1 = event.ydata
        if self.x_1 > 0 and self.x_1 < self.img.width:
            self.rect.set_width(self.x_1 - self.x_0)
            self.rect.set_height(self.y_1 - self.y_0)
            self.rect.set_xy((self.x_0, self.y_0))
        self.fig.canvas.draw()

    def on_scroll(self, event):
        """
        鼠标滚动callback
        """
        # 清空先前图像缓存
        self.ax.cla()

        if event.button == 'down' and event.step < -0.65:  # 下一张图
            # 更新图像数据
            self.idx += 1
        elif event.button == 'up' and event.step > 0.65:  # 前一张图
            if self.idx == 0:  # 对于第一张图, 不存在前一张图
                print('[Note]: idx 0 image has no previous image.')
                return
            self.idx -= 1

        # 更新绘图
        self.update_fig()

    def on_btn_press(self, event):
        """
        鼠标按下callback
        """
        # print('=> mouse btn press')
        self.x_0 = event.xdata
        self.y_0 = event.ydata
        self.is_btn_press = True

    def on_btn_release(self, event):
        """
        鼠标释放callback
        """
        # print('=> mouse btn release')
        if self.is_rect_ready:  # 如果是奇数次按下鼠标: 恢复鼠标未被按下的状态
            self.is_btn_press = False

            x_start = int(self.rect.get_x())
            x_end = int(self.rect.get_x() + self.rect.get_width())
            y_start = int(self.rect.get_y())
            y_end = int(self.rect.get_y() + self.rect.get_height())
            if x_start < x_end and y_start < y_end:
                self.ROI = Image.fromarray(
                    np.array(self.img)[y_start: y_end, x_start: x_end])
            elif x_start > x_end and y_start > y_end:
                self.ROI = Image.fromarray(
                    np.array(self.img)[y_end: y_start, x_end: x_start])

            if None != self.ROI:  # ROI是 PIL Image, 对ROI进行预测
                # car_color, car_direction, car_type = self.manager.predict(
                #     self.ROI)
                self.ROI.show()
                # print('=> predict:', car_color, car_direction, car_type)

        # 取反
        self.is_rect_ready = not self.is_rect_ready

    def on_mouse_motion(self, event):
        """
        鼠标移动callback
        """
        # print('=> mouse moving...')

        if self.is_btn_press:
            if None == event.xdata or None == event.ydata:
                self.is_btn_press = False
                return
            self.draw_rect(event)

    def on_key_release(self, event):
        """
        键盘按键释放callback
        """
        if event.key == 'c':  # clip and save to destination dir
            date_name = time.strftime(
                '_%Y_%m_%d_', time.localtime(time.time()))

            self.clip_id += 1
            write_name = self.dst_dir + '/' + \
                date_name + \
                str(self.idx) + \
                '_' + \
                str(self.clip_id) + \
                '.jpg'
            self.ROI.save(write_name)
            print('=> %s saved.' % write_name)

            # label = input('=> Enter label string:')  # 手动输入label
            # self.label_dict[write_name.split('/')[-1]] = label
            # print('=> label: ', label)

            # 现在并不自动跳到下一帧
            # self.idx += 1
            # self.update_fig()
        elif event.key == 'e':  # 退出程序
            self.exit()
        self.is_btn_press = False


# -----------------------------------------------------------

# 网络模型
# class Net(torch.nn.Module):
#     """
#     power-set车辆多标签分类
#     """

#     def __init__(self, num_cls, input_size):
#         """
#         网络定义
#         :param is_freeze:
#         """
#         torch.nn.Module.__init__(self)

#         # 输出通道数
#         self._num_cls = num_cls

#         # 输入图像尺寸
#         self.input_size = input_size

#         # 删除原有全连接, 得到特征提取层
#         self.features = torchvision.models.resnet18(pretrained=True)
#         del self.features.fc
#         # print('feature extractor:\n', self.features)

#         self.features = torch.nn.Sequential(
#             *list(self.features.children()))

#         # 重新定义全连接层
#         self.fc = torch.nn.Linear(512 ** 2, num_cls)  # 输出类别数
#         # print('=> fc layer:\n', self.fc)

#     def forward(self, X):
#         """
#         :param X:
#         :return:
#         """
#         N = X.size()[0]

#         X = self.features(X)  # extract features

#         X = X.view(N, 512, 1 ** 2)
#         X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear

#         X = X.view(N, 512 ** 2)
#         X = torch.sqrt(X + 1e-5)
#         X = torch.nn.functional.normalize(X)
#         X = self.fc(X)
#         assert X.size() == (N, self._num_cls)  # 输出类别数
#         return X


# 封装管理
# class Manager(object):
#     """
#     模型初始化等
#     """

#     def __init__(self,
#                  model_path,
#                  attrib_path):
#         """
#         加载模型并初始化
#         """

#         # 定义模型, 放入device, 加载权重
#         self.net = Net(num_cls=23,
#                        input_size=224).to(device)

#         # self.net = torch.nn.DataParallel(Net(num_cls=23, input_size=224),
#         #                                  device_ids=[0]).to(device)

#         self.net.load_state_dict(torch.load(model_path))
#         print('=> vehicle classifier loaded from %s' % model_path)

#         # 设置模型为测试模式
#         self.net.eval()

#         # 测试数据预处理方式
#         self.transforms = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(size=224),
#             torchvision.transforms.CenterCrop(size=224),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                              std=(0.229, 0.224, 0.225))
#         ])

#         # 加载attributes向量
#         self.attributes = pickle.load(open(attrib_path, 'rb'))
#         self.attributes = [str(x) for x in self.attributes]
#         # print('=> training attributes:\n', attributes)

#         # 将多标签分开
#         self.color_attrs = self.attributes[:11]
#         del self.color_attrs[5]
#         print('=> color_attrs:\n', self.color_attrs)

#         self.direction_attrs = self.attributes[11:14]
#         del self.direction_attrs[2]
#         print('=> direction attrs:\n', self.direction_attrs)

#         self.type_attrs = self.attributes[14:]
#         del self.type_attrs[6]
#         print('=> type_attrs:\n', self.type_attrs)

#     def get_predict_ce(self, output):
#         """
#         softmax归一化,然后统计每一个标签最大值索引
#         :param output:
#         :return:
#         """
#         # 计算预测值
#         output = output.cpu()  # 从GPU拷贝出到host端
#         pred_color = output[:, :11]
#         pred_direction = output[:, 11:14]
#         pred_type = output[:, 14:]

#         color_idx = pred_color.max(1, keepdim=True)[1]
#         direction_idx = pred_direction.max(1, keepdim=True)[1]
#         type_idx = pred_type.max(1, keepdim=True)[1]

#         # 连接pred
#         pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
#         return pred

#     def get_predict(self, output):
#         """
#         新输出向量(20维)的处理
#         """
#         # 计算预测值
#         output = output.cpu()  # 从GPU拷贝出到host端
#         pred_color = output[:, :10]
#         pred_direction = output[:, 10:12]
#         pred_type = output[:, 12:]

#         color_idx = pred_color.max(1, keepdim=True)[1]
#         direction_idx = pred_direction.max(1, keepdim=True)[1]
#         type_idx = pred_type.max(1, keepdim=True)[1]

#         # 连接pred
#         pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
#         return pred

#     def pre_process(self, image):
#         """
#         图像数据类型转换
#         :rtype: PIL.JpegImagePlugin.JpegImageFile
#         """
#         # 数据预处理
#         if type(image) == np.ndarray:
#             if image.shape[2] == 3:  # 3通道转换成RGB
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             elif image.shape[2] == 1:  # 单通道, 灰度转换成RGB
#                 image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#             # numpy.ndarray转换成PIL.Image
#             image = Image.fromarray(image)
#         elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
#             if image.mode == 'L' or image.mode == 'I':  # 8bit或32bit单通道灰度图转换成RGB
#                 image = image.convert('RGB')

#         return image

#     def predict(self, img):
#         """
#         预测属性: 输入图像通过PIL读入的
#         :return:返回预测的车辆颜色、车辆朝向、车辆类别
#         """
#         # 数据预处理
#         img = self.transforms(img)
#         img = img.view(1, 3, 224, 224)

#         # 图像数据放入device运行
#         img = img.to(device)

#         # 前向运算
#         output = self.net.forward(img)

#         # 获取预测结果
#         try:
#             pred = self.get_predict(output)  # self.get_predict_ce, 返回的pred在host端
#             color_name = self.color_attrs[pred[0][0]]
#             direction_name = self.direction_attrs[pred[0][1]]
#             type_name = self.type_attrs[pred[0][2]]
#         except Exception as e:
#             return None, None, None

#         return color_name, direction_name, type_name


def test(is_pil=True):
    """
    单元测试和可视化
    :return:
    """
    # 测试数据路径
    root = 'e:/pick_car_roi'
    model_path = 'e:/epoch_42.pth'
    attrib_path = 'e:/vehicle_attributes.pkl'

    # 模型初始化
    manager = Manager(model_path=model_path, attrib_path=attrib_path)

    for file in os.listdir(root):
        # 读取测试数据
        file_path = os.path.join(root, file)

        if is_pil:
            image = Image.open(file_path)  # 通过PIL读取图像
        else:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # 通过opencv读取图像

        # -------------------------------
        # 图像数据格式预处理
        image = manager.pre_process(image)

        # 预测
        car_color, car_direction, car_type = manager.predict(image)
        # -------------------------------

        # 可视化
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(car_color + ' ' + car_direction + ' ' + car_type)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.show()


class Car_DR():
    def __init__(self,
                 src_dir,
                 dst_dir,
                 car_cfg_path='./car.cfg',
                 car_det_weights_path='g:/Car_DR/car_360000.weights',
                 inp_dim=768,
                 prob_th=0.2,
                 nms_th=0.4,
                 num_classes=1):
        """
        模型初始化
        """
        # 超参数
        self.inp_dim = inp_dim
        self.prob_th = prob_th
        self.nms_th = nms_th
        self.num_classes = num_classes
        self.dst_dir = dst_dir

        # 清空dst_dir
        if os.path.exists(self.dst_dir):
            for x in os.listdir(self.dst_dir):
                if x.endswith('.jpg'):
                    os.remove(self.dst_dir + '/' + x)
        else:
            os.makedirs(self.dst_dir)

        # 初始化车辆检测模型及参数
        self.Net = Darknet(car_cfg_path)
        self.Net.load_weights(car_det_weights_path)
        self.Net.net_info['height'] = self.inp_dim  # 车辆检测输入分辨率
        self.Net.to(device)
        self.Net.eval()  # 测试模式
        print('=> car detection model initiated.')

        # 初始化车辆多标签分类管理器
        self.manager = Manager(model_path=model_path, attrib_path=attrib_path)

        # 统计src_dir文件
        self.imgs_path = [os.path.join(src_dir, x) for x in os.listdir(
            src_dir) if x.endswith('.jpg')]

    def cls_draw_bbox(self, output, orig_img):
        """
        orig_img是通过opencv读取的numpy array格式: 通道顺序BGR
        在bbox基础上预测车辆属性
        将bbox绘制到原图上
        """
        labels = []
        pt_1s = []
        pt_2s = []

        # 获取车辆属性labels
        for det in output:
            # rectangle points
            pt_1 = tuple(det[1:3].int())  # the left-up point
            pt_2 = tuple(det[3:5].int())  # the right down point
            pt_1s.append(pt_1)
            pt_2s.append(pt_2)

            # 调用分类器预测车辆属性: BGR => RGB
            ROI = Image.fromarray(
                orig_img[pt_1[1]: pt_2[1],
                         pt_1[0]: pt_2[0]][:, :, ::-1])
            # ROI.show()

            car_color, car_direction, car_type = self.manager.predict(ROI)
            label = str(car_color + ' ' + car_direction + ' ' + car_type)
            labels.append(label)
            print('=> predicted label: ', label)

        # 将bbox绘制到原图
        color = (0, 215, 255)
        for i, det in enumerate(output):
            pt_1 = pt_1s[i]
            pt_2 = pt_2s[i]

            # 绘制bounding box
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)

            # 获取文本大小
            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # 文字大小
            # pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] + txt_size[1] + 5
            pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] - txt_size[1] - 5

            # 绘制文本底色矩形
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=-1)  # text

            # 绘制文本
            cv2.putText(orig_img, labels[i], (pt_1[0], pt_1[1]),  # pt_1[1] + txt_size[1] + 4
                        cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)

    def cls_and_draw(self, output, orig_img):
        """
        orig_img是PIL Image图像格式
        在bbox基础上预测车辆属性
        将bbox绘制到原图上
        """
        labels = []
        x_ys = []
        w_hs = []

        # 获取车辆属性labels
        for det in output:
            # rectangle
            x_y = tuple(det[1:3].int())  # x, y
            w_h = tuple(det[3:5].int())  # w, h
            x_ys.append(x_y)
            w_hs.append(w_h)

            # 调用分类器预测车辆属性: BGR => RGB
            box = (int(x_y[0]), int(x_y[1]), int(x_y[0] + w_h[0]),
                   int(x_y[1] + w_h[1]))  # left, upper, right, lower
            ROI = orig_img.crop(box)

            car_color, car_direction, car_type = self.manager.predict(ROI)
            label = car_color + ' ' + car_direction + ' ' + car_type
            print('=> label: ', label)
            labels.append(label)

        # 将bbox绘制到原图
        for i, det in enumerate(output):
            x_y = x_ys[i]
            w_h = w_hs[i]

            color = (0, 215, 255)
            cv2.rectangle(np.asarray(orig_img), x_y, w_h, color,
                          thickness=2)  # bounding box

            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # 文字大小
            w_h = x_y[0] + txt_size[0] + 4, x_y[1] + txt_size[1] + 4
            cv2.rectangle(np.asarray(orig_img), x_y, w_h,
                          color, thickness=-1)  # text
            cv2.putText(np.asarray(orig_img), labels[i], (x_y[0], x_y[1] + txt_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)

    def predict(self):
        """
        批量检测和识别, 将检测, 识别结果输出到dst_dir
        """
        for x in self.imgs_path:
            # 读取图像数据
            img = Image.open(x)
            img2det = process_img(img, self.inp_dim)
            img2det = img2det.to(device)  # 图像数据放到device

            # 车辆检测
            prediction = self.Net.forward(img2det, CUDA=True)

            # 计算scaling factor
            orig_img_size = list(img.size)
            output = process_predict(prediction,
                                     self.prob_th,
                                     self.num_classes,
                                     self.nms_th,
                                     self.inp_dim,
                                     orig_img_size)

            orig_img = cv2.cvtColor(np.asarray(
                img), cv2.COLOR_RGB2BGR)  # RGB => BGR
            if type(output) != int:
                # 将检测框bbox绘制到原图上
                # draw_car_bbox(output, orig_img)
                self.cls_draw_bbox(output, orig_img)
                # self.cls_and_draw(output, img)
                dst_path = self.dst_dir + '/' + os.path.split(x)[1]
                if not os.path.exists(dst_path):
                    cv2.imwrite(dst_path, orig_img)

# -----------------------------------------------------------


def test_car_detect(car_cfg_path='./car.cfg',
                    car_det_weights_path='g:/Car_DR/car_360000.weights'):
    """
    imgs_path: 图像数据路径
    """
    inp_dim = 768
    prob_th = 0.2  # 车辆检测概率阈值
    nms_th = 0.4  # NMS阈值
    num_cls = 1  # 只检测车辆1类

    # 初始化车辆检测模型及参数
    Net = Darknet(car_cfg_path)
    Net.load_weights(car_det_weights_path)
    Net.net_info['height'] = inp_dim  # 车辆检测输入分辨率
    Net.to(device)
    Net.eval()  # 测试模式
    print('=> car detection model initiated.')

    # 读取图像数据
    img = Image.open(
        'f:/FaceRecognition_torch_0_4/imgs_21/det_2018_08_21_63_1.jpg')
    img2det = process_img(img, inp_dim)
    img2det = img2det.to(device)  # 图像数据放到device

    # 测试车辆检测
    prediction = Net.forward(img2det, CUDA=True)

    # 计算scaling factor
    orig_img_size = list(img.size)
    output = process_predict(prediction,
                             prob_th,
                             num_cls,
                             nms_th,
                             inp_dim,
                             orig_img_size)

    orig_img = np.asarray(img)
    if type(output) != int:
        # 将检测框bbox绘制到原图上
        draw_car_bbox(output, orig_img)

    cv2.imshow('test', orig_img)
    cv2.waitKey()


"""
    # prep_ret = prep_image('f:/FaceRecognition_torch_0_4/imgs_21/det_2018_08_21_63_1.jpg',
    #                       inp_dim)  # 返回一个Tensor
    # img2det = prep_ret[0].view(1, 3, inp_dim, inp_dim)
    # Net.load_state_dict(torch.load('./car_detect_model.pth'))
"""


def draw_car_bbox(output, orig_img):
    for det in output:
        label = 'car'  # 类型名称
        prob = '{:.3f}'.format(det[5].cpu().numpy())
        label += prob

        x_y = tuple(det[1:3].int())  # x, y
        w_h = tuple(det[3:5].int())  # w, h

        color = (0, 215, 255)
        cv2.rectangle(orig_img, x_y, w_h, color,
                      thickness=2)  # bounding box

        txt_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # 文字大小
        w_h = x_y[0] + txt_size[0] + 3, x_y[1] + txt_size[1] + 4
        cv2.rectangle(orig_img, x_y, w_h, color, thickness=-1)  # text
        cv2.putText(orig_img, label, (x_y[0], x_y[1] + txt_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)


def process_predict(prediction,
                    prob_th,
                    num_cls,
                    nms_th,
                    inp_dim,
                    orig_img_size):
    """
    处理预测结果
    """
    scaling_factor = min([inp_dim / float(x)
                          for x in orig_img_size])  # W, H缩放系数
    output = post_process(prediction,
                          prob_th,
                          num_cls,
                          nms=True,
                          nms_conf=nms_th,
                          CUDA=True)  # post-process such as nms

    if type(output) != int:
        output[:, [1, 3]] -= (inp_dim - scaling_factor *
                              orig_img_size[0]) / 2.0  # x, w
        output[:, [2, 4]] -= (inp_dim - scaling_factor *
                              orig_img_size[1]) / 2.0  # y, h
        output[:, 1:5] /= scaling_factor
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(
                output[i, [1, 3]], 0.0, orig_img_size[0])
            output[i, [2, 4]] = torch.clamp(
                output[i, [2, 4]], 0.0, orig_img_size[1])
    return output


def test_equal(f_path_1, f_path_2):
    """
    f_path_1: 第一个文件路径
    f_path_2: 第二个文件路径
    """
    arr_1 = np.load(f_path_1)['arr_0']
    arr_2 = np.load(f_path_2)['arr_0']

    # 判断两个数组是否逐元素相等
    print('=> the two array is equal:', (arr_1 == arr_2).all())


# --------------------------------将clipper处理的数据合并回vehicle_train
def process_clipped(src_root, dst_root):
    """
    将src_root中的数据按照label合并到dst_root对应子目录
    """
    # 加载label_dict
    # label_path = src_root + '/' + 'label_dict.pkl'
    try:
        label_dict = pickle.load(
            open('f:/FaceRecognition_torch_0_4/label_dict.pkl', 'rb'))
        # print(label_dict)
    except Exception as e:
        print(e)

    # 遍历src_root
    for x in os.listdir(src_root):
        if x.endswith('.jpg'):  # 只处理存在的jpg图
            if x in label_dict.keys():  # 只处理存在key的数据
                label = label_dict[x]
                # print('=> key: %s, value: %s' % (x, label))
                sub_dir_path = dst_root + '/' + label.replace(' ', '_')
                # print(sub_dir_path)

                # 如果src, dst文件存在才合并
                if os.path.isdir(sub_dir_path):
                    src_path = src_root + '/' + x
                    if os.path.exists(src_path):
                        dst_path = sub_dir_path + '/' + x
                        if not os.path.exists(dst_path):  # 如果已经存, 则不再拷贝
                            shutil.copy(src_path, sub_dir_path)
                            print('=> %s copied to %s' %
                                  (src_path, sub_dir_path))

# ----------------------------


def viz_err(err_path, root='f:/'):
    """
    可视化分类错误信息
    """
    err_dict = pickle.load(open(err_path, 'rb'))
    # print(err_dict)

    fig = plt.figure()  #

    for k, v in err_dict.items():
        img_path = root + k
        if os.path.isfile(img_path):
            img = Image.open(img_path)
            plt.gcf().set_size_inches(8, 8)
            plt.imshow(img)
            plt.title(img_path + '\n' + v)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.show()


if __name__ == '__main__':
    # ---------------------------- Clip roi, labeling and copy
    parser = argparse.ArgumentParser(description='Cropper parameters')
    parser.add_argument('-src',
                        type=str,
                        dest='s',
                        default=u'f:/LPVehicleID_1/',
                        help='dir path of JPEGImages')
    parser.add_argument('-dst',
                        type=str,
                        dest='d',
                        default=u'f:/LPVehicleID_pro/',
                        help='dir path of JPEGImages')
    parser.add_argument('-folder',
                        type=str,
                        dest='f',
                        default=u'桂A66K53',
                        help='dir path of JPEGImages')
    parser.add_argument('-r',
                        type=int,
                        default=0,
                        help='dir path of JPEGImages')
    args = parser.parse_args()

    cropper = Cropper(root=args.s + args.f,
                      dst_dir=args.d + args.f,
                      is_resume=args.r)

    # process_clipped(src_root=u'f:/LPVehicleID_1/川A1D695',
    #                 dst_root='f:/vehicle_train')

    # ----------------------------
    # test_car_detect()

    # ---------------------------- Car detect and classify
    # DR_model = Car_DR(src_dir='g:/car_0819',
    #                   dst_dir='f:/test_result')
    # DR_model.predict()

    # ----------------------------
    # test_equal('e:/prediction_1.npz', 'e:/prediction_2.npz'c)
    # test()

    # ----------------------------

    # viz_err('g:/err_dict.pkl')
    print('=> Test done.')
