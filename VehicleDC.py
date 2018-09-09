# coding: utf-8

import os
import sys
import re
import time
import pickle
import shutil
import random
import argparse

from darknet_util import *
from darknet import Darknet
from preprocess import prep_image, process_img, inp_to_image
from dataset import color_attrs, direction_attrs, type_attrs

import torch
import torchvision
import paramiko
import cv2
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.image import AxesImage
from scipy.spatial.distance import cityblock
from tqdm import tqdm

# -------------------------------------
# for matplotlib to displacy chinese characters correctly
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

use_cuda = True  # True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
print('=> device: ', device)

# root = 'e:/pick_car_roi'
local_model_path = './checkpoints/epoch_39.pth'
local_car_cfg_path = './car.cfg'
local_car_det_weights_path = './car_detect.weights'

local_color_attrs = ['Black', 'Blue', 'Brown', 'Gray',
                     'Green', 'Pink', 'Red', 'Violet', 'White', 'Yellow']
local_direction_attrs = ['Front', 'Rear']
local_type_attrs = ['passengerCar', 'saloonCar',
                    'shopTruck', 'suv', 'trailer', 'truck', 'van', 'waggon']


class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)  # 输出类别数
        # print('=> fc layer:\n', self.fc)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


# ------------------------------------- vehicle detection model
class Car_Classifier(object):
    """
    vehicle detection model mabager
    """

    def __init__(self,
                 num_cls,
                 model_path=local_model_path):
        """
        load model and initialize
        """

        # define model and load weights
        self.net = Cls_Net(num_cls=num_cls, input_size=224).to(device)
        # self.net = torch.nn.DataParallel(Net(num_cls=20, input_size=224),
        #                                  device_ids=[0]).to(device)
        self.net.load_state_dict(torch.load(model_path))
        print('=> vehicle classifier loaded from %s' % model_path)

        # set model to eval mode
        self.net.eval()

        # test data transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # split each label
        self.color_attrs = color_attrs
        print('=> color_attrs:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        print('=> direction attrs:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        print('=> type_attrs:\n', self.type_attrs)

    def get_predict(self, output):
        """
        get prediction from output
        """
        # get each label's prediction from output
        output = output.cpu()  # fetch data from gpu
        pred_color = output[:, :9]
        pred_direction = output[:, 9:11]
        pred_type = output[:, 11:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def pre_process(self, image):
        """
        image formatting
        :rtype: PIL.JpegImagePlugin.JpegImageFile
        """
        # image data formatting
        if type(image) == np.ndarray:
            if image.shape[2] == 3:  # turn all 3 channels to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 1:  # turn 1 channel to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # turn numpy.ndarray into PIL.Image
            image = Image.fromarray(image)
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            if image.mode == 'L' or image.mode == 'I':  # turn 8bits or 32bits into 3 channels RGB
                image = image.convert('RGB')

        return image

    def predict(self, img):
        """
        predict vehicle attributes by classifying
        :return: vehicle color, direction and type 
        """
        # image pre-processing
        img = self.transforms(img)
        img = img.view(1, 3, 224, 224)

        # put image data into device
        img = img.to(device)

        # calculating inference
        output = self.net.forward(img)

        # get result
        # self.get_predict_ce, return pred to host side(cpu)
        pred = self.get_predict(output)
        color_name = self.color_attrs[pred[0][0]]
        direction_name = self.direction_attrs[pred[0][1]]
        type_name = self.type_attrs[pred[0][2]]

        return color_name, direction_name, type_name


class Car_DC():
    def __init__(self,
                 src_dir,
                 dst_dir,
                 car_cfg_path=local_car_cfg_path,
                 car_det_weights_path=local_car_det_weights_path,
                 inp_dim=768,
                 prob_th=0.2,
                 nms_th=0.4,
                 num_classes=1):
        """
        model initialization
        """
        # super parameters
        self.inp_dim = inp_dim
        self.prob_th = prob_th
        self.nms_th = nms_th
        self.num_classes = num_classes
        self.dst_dir = dst_dir

        # clear dst_dir
        if os.path.exists(self.dst_dir):
            for x in os.listdir(self.dst_dir):
                if x.endswith('.jpg'):
                    os.remove(self.dst_dir + '/' + x)
        else:
            os.makedirs(self.dst_dir)

        # initialize vehicle detection model
        self.detector = Darknet(car_cfg_path)
        self.detector.load_weights(car_det_weights_path)
        # set input dimension of image
        self.detector.net_info['height'] = self.inp_dim
        self.detector.to(device)
        self.detector.eval()  # evaluation mode
        print('=> car detection model initiated.')

        # initiate multilabel classifier
        self.classifier = Car_Classifier(num_cls=19,
                                         model_path=local_model_path)

        # initiate imgs_path
        self.imgs_path = [os.path.join(src_dir, x) for x in os.listdir(
            src_dir) if x.endswith('.jpg')]

    def cls_draw_bbox(self, output, orig_img):
        """
        1. predict vehicle's attributes based on bbox of vehicle
        2. draw bbox to orig_img
        """
        labels = []
        pt_1s = []
        pt_2s = []

        # 1
        for det in output:
            # rectangle points
            pt_1 = tuple(det[1:3].int())  # the left-up point
            pt_2 = tuple(det[3:5].int())  # the right down point
            pt_1s.append(pt_1)
            pt_2s.append(pt_2)

            # turn BGR back to RGB
            ROI = Image.fromarray(
                orig_img[pt_1[1]: pt_2[1],
                         pt_1[0]: pt_2[0]][:, :, ::-1])
            # ROI.show()

            # call classifier to predict
            car_color, car_direction, car_type = self.classifier.predict(ROI)
            label = str(car_color + ' ' + car_direction + ' ' + car_type)
            labels.append(label)
            print('=> predicted label: ', label)

        # 2
        color = (0, 215, 255)
        for i, det in enumerate(output):
            pt_1 = pt_1s[i]
            pt_2 = pt_2s[i]

            # draw bounding box
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)

            # get str text size
            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            # pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] + txt_size[1] + 5
            pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] - txt_size[1] - 5

            # draw text background rect
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=-1)  # text

            # draw text
            cv2.putText(orig_img, labels[i], (pt_1[0], pt_1[1]),  # pt_1[1] + txt_size[1] + 4
                        cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)

    def process_predict(self,
                        prediction,
                        prob_th,
                        num_cls,
                        nms_th,
                        inp_dim,
                        orig_img_size):
        """
        processing detections
        """
        scaling_factor = min([inp_dim / float(x)
                              for x in orig_img_size])  # W, H scaling factor
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

    def detect_classify(self):
        """
        detect and classify
        """
        for x in self.imgs_path:
            # read image data
            img = Image.open(x)
            img2det = process_img(img, self.inp_dim)
            img2det = img2det.to(device)  # put image data to device

            # vehicle detection
            prediction = self.detector.forward(img2det, CUDA=True)

            # calculating scaling factor
            orig_img_size = list(img.size)
            output = self.process_predict(prediction,
                                          self.prob_th,
                                          self.num_classes,
                                          self.nms_th,
                                          self.inp_dim,
                                          orig_img_size)

            orig_img = cv2.cvtColor(np.asarray(
                img), cv2.COLOR_RGB2BGR)  # RGB => BGR
            if type(output) != int:
                self.cls_draw_bbox(output, orig_img)
                dst_path = self.dst_dir + '/' + os.path.split(x)[1]
                if not os.path.exists(dst_path):
                    cv2.imwrite(dst_path, orig_img)

# -----------------------------------------------------------


parser = argparse.ArgumentParser(description='Detect and classify cars.')
parser.add_argument('-src-dir',
                    type=str,
                    default='./test_imgs',
                    help='source directory of images')
parser.add_argument('-dst-dir',
                    type=str,
                    default='./test_result',
                    help='destination directory of images to store results.')

if __name__ == '__main__':
    # ---------------------------- Car detect and classify
    # DR_model = Car_DC(src_dir='./test_imgs',
    #                   dst_dir='./test_result')
    # DR_model.detect_classify()

    args = parser.parse_args()
    DR_model = Car_DC(src_dir=args.src_dir, dst_dir=args.dst_dir)
    DR_model.detect_classify()
