# coding: utf-8

import os
import re
import shutil
import pickle
import numpy as np
import scipy.io as scio
from torch import Tensor as Tensor
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image

color_attrs = ['Black', 'Blue', 'Brown',
                     'Gray', 'Green', 'Pink',
                     'Red', 'White', 'Yellow']
direction_attrs = ['Front', 'Rear']
type_attrs = ['passengerCar', 'saloonCar',
                    'shopTruck', 'suv', 'trailer', 'truck', 'van', 'waggon']


class Vehicle(data.Dataset):
    """
    属性向量多标签:配合cross entropy loss的使用
    使用处理过的数据: 去掉所有的unknown
    """

    def __init__(self,
                 root,
                 transform=None,
                 is_train=True):
        """
        :return:
        """
        if not os.path.exists(root):
            print('=> [Err]: root not exists.')
            return
        if is_train:
            print('=> train data root: ', root)
        else:
            print('=> test data root: ', root)

        # 统计非空子目录并按名称(类别名称)自然排序
        self.img_dirs = [os.path.join(root, x) for x in os.listdir(root) \
                         if os.path.isdir(os.path.join(root, x))]
        self.img_dirs = [x for x in self.img_dirs if len(os.listdir(x)) != 0]
        if len(self.img_dirs) == 0:
            print('=> [Err]: empty sub-dirs.')
            return
        self.img_dirs.sort()  # 默认自然排序, 从小到大
        # print('=> total {:d} classes for training'.format(len(self.img_dirs)))

        # 将多标签分开
        self.color_attrs = color_attrs
        self.direction_attrs = direction_attrs
        self.type_attrs = type_attrs

        # 按子目录(类名)的顺序排序文件路径
        self.imgs_path = []
        self.labels = []
        for x in self.img_dirs:
            match = re.match('([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z]+)', os.path.split(x)[1])
            color = match.group(1)  # 车身颜色
            direction = match.group(2)  # 车身方向
            type = match.group(3)  # 车身类型
            # print('=> color: %s, direction: %s, type: %s' % (color, direction, type))

            for y in os.listdir(x):
                # 添加文件路径
                self.imgs_path.append(os.path.join(x, y))

                # 添加label
                color_idx = int(np.where(self.color_attrs == np.array(color))[0])
                direction_idx = int(np.where(self.direction_attrs == np.array(direction))[0])
                type_idx = int(np.where(self.type_attrs == np.array(type))[0])
                label = np.array([color_idx, direction_idx, type_idx], dtype=int)

                label = torch.Tensor(label)  # torch.from_numpy(label)
                self.labels.append(label)  # Tensor(label)
                # print(label)

        if is_train:
            print('=> total {:d} samples for training.'.format(len(self.imgs_path)))
        else:
            print('=> total {:d} samples for testing.'.format(len(self.imgs_path)))

        # 加载数据变换
        if transform is not None:
            self.transform = transform
        else:  # default image transformation
            self.transform = T.Compose([
                T.Resize(448),
                T.CenterCrop(448),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # --------------------- serialize imgs_path to disk
        # root_parent = os.path.abspath(os.path.join(root, '..'))
        # print('=> parent dir: ', root_parent)
        # if is_train:
        #     imgs_path =  os.path.join(root_parent, 'train_imgs_path.pkl')
        # else:
        #     imgs_path = os.path.join(ropytorch docot_parent, 'test_imgs_path.pkl')
        # print('=> dump imgs path: ', imgs_path)
        # pickle.dump(self.imgs_path, open(imgs_path, 'wb'))

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        image = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if image.mode == 'L' or image.mode == 'I':  # 8bit或32bit灰度图
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        f_path = os.path.split(self.imgs_path[idx])[0].split('/')[-2] + \
                 '/' + os.path.split(self.imgs_path[idx])[0].split('/')[-1] + \
                 '/' + os.path.split(self.imgs_path[idx])[1]
        return image, label, f_path

    def __len__(self):
        """os.path.split(self.imgs_path[idx])[0].split('/')[-2]
        :return:
        """
        return len(self.imgs_path)
