# coding: utf-8

import os
import re
import shutil
import time
import pickle
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import dataset
from dataset import color_attrs, direction_attrs, type_attrs

from copy import deepcopy
from PIL import Image

from torchvision.datasets import ImageFolder

from copy import deepcopy
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

# print('=> torch version: ', torch.__version__)

is_remote = False
use_cuda = True  # True

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if is_remote:  # remote side
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # users can modify this according to needs and hardware
    device = torch.device(
        'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')
else:  # local side
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device(
        'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

# print('=> device: ', device)


class Classifier(torch.nn.Module):
    """
    vehicle multilabel-classifier
    """

    def __init__(self, num_cls, input_size, is_freeze=True):
        """
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        self._is_freeze = is_freeze
        print('=> is freeze: {}'.format(self._is_freeze))

        # delete origin FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)  # True
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)  # output channels
        # print('=> fc layer:\n', self.fc)

        # -----------whether to freeze
        if self._is_freeze:
            for param in self.features.parameters():
                param.requires_grad = False

            # init FC layer
            torch.nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        # assert X.size() == (N, 3, self.input_size, self.input_size)

        X = self.features(X)  # extract features

        # print('X.size: ', X.size())
        # assert X.size() == (N, 512, 1, 1)

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN for fine-grained classification

        # assert X.size() == (N, 512, 512)

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        assert X.size() == (N, self._num_cls)
        return X


class Manager(object):
    """
    train and test manager
    """
    def __init__(self, options, path):
        """
        model initialization
        """
        self.options = options
        self.path = path

        # get latest model checkpoint
        if self.options['is_resume']:
            if int(self.path['model_id']) == -1:
                checkpoints = os.listdir(self.path['net'])
                checkpoints.sort(key=lambda x: int(re.match('epoch_(\d+)\.pth', x).group(1)),
                                 reverse=True)
                if len(checkpoints) != 0:
                    self.LATEST_MODEL_ID = int(
                        re.match('epoch_(\d+)\.pth', checkpoints[0]).group(1))
            else:
                self.LATEST_MODEL_ID = int(self.path['model_id'])
        else:
            self.LATEST_MODEL_ID = 0
        print('=> latest net id: {}'.format(self.LATEST_MODEL_ID))

        # net config
        if is_remote:
            self.net = Classifier(num_cls=19,  # 19 = len(color_attrs) + len(direction_attrs) + len(type_attrs)
                           input_size=224,
                           is_freeze=self.options['is_freeze']).to(device)
        else:
            self.net = Classifier(num_cls=19,
                           input_size=224,
                           is_freeze=self.options['is_freeze']).to(device)

        # whether to resume from checkpoint
        if self.options['is_resume']:
            if int(self.path['model_id']) == -1:
                model_path = os.path.join(self.path['net'], checkpoints[0])
            else:
                model_path = self.path['net'] + '/' + \
                    'epoch_' + self.path['model_id'] + '.pth'
            self.net.load_state_dict(torch.load(model_path))
            print('=> net resume from {}'.format(model_path))
        else:
            print('=> net loaded from scratch.')

        # loss function
        self.loss_func = torch.nn.CrossEntropyLoss().to(device)

        # Solver
        if self.options['is_freeze']:
            print('=> fine-tune only the FC layer.')
            self.solver = torch.optim.SGD(self.net.fc.parameters(),
                                          lr=self.options['base_lr'],
                                          momentum=0.9,
                                          weight_decay=self.options['weight_decay'])
        else:
            print('=> fine-tune all layers.')
            self.solver = torch.optim.SGD(self.net.parameters(),
                                          lr=self.options['base_lr'],
                                          momentum=0.9,
                                          weight_decay=self.options['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.solver,
                                                                    mode='max',
                                                                    factor=0.1,
                                                                    patience=3,
                                                                    verbose=True,
                                                                    threshold=1e-4)

        # train data enhancement
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=self.net.input_size),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(
                size=self.net.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # test preprocess
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=self.net.input_size),
            torchvision.transforms.CenterCrop(size=self.net.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # load train and test data
        if is_remote:
            self.train_set = dataset.Vehicle(self.path['train_data'],
                                             transform=self.test_transforms,  # train_transforms
                                             is_train=True)
            self.test_set = dataset.Vehicle(self.path['test_data'],
                                            transform=self.test_transforms,
                                            is_train=False)
        else:
            self.train_set = dataset.Vehicle(self.path['train_data'],
                                             transform=self.test_transforms,  # train_transforms
                                             is_train=True)
            self.test_set = dataset.Vehicle(self.path['test_data'],
                                            transform=self.test_transforms,
                                            is_train=False)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.options['batch_size'],
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=1,  # one image each batch for testing
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True)

        # multilabels
        self.color_attrs = color_attrs
        print('=> color attributes:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        print('=> direction attributes:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        print('=> type_attributes:\n', self.type_attrs, '\n')

        # for storage and further analysis for err details
        self.err_dict = {}

    def train(self):
        """
        train the network
        """
        print('==> Training...')

        self.net.train()  # train mode

        best_acc = 0.0
        best_epoch = None

        print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self.options['epochs']):  # traverse each epoch
            epoch_loss = []
            num_correct = 0
            num_total = 0

            for data, label, _ in self.train_loader:  # traverse each batch in the epoch
                # put training data, label to device
                data, label = data.to(device), label.to(device)

                # clear the grad
                self.solver.zero_grad()

                # forword calculation
                output = self.net.forward(data)

                # calculate each attribute loss
                label = label.long()
                loss_color = self.loss_func(output[:, :9], label[:, 0])
                loss_direction = self.loss_func(output[:, 9:11], label[:, 1])
                loss_type = self.loss_func(output[:, 11:], label[:, 2])
                loss = loss_color + loss_direction + 2.0 * loss_type  # greater weight to type 

                # statistics of each epoch loss
                epoch_loss.append(loss.item())

                # statistics of sample number
                num_total += label.size(0)

                # statistics of accuracy
                pred = self.get_predict(output)
                label = label.cpu().long()
                num_correct += self.count_correct(pred, label)

                # backward calculation according to loss
                loss.backward()
                self.solver.step()

            # calculate training accuray
            train_acc = 100.0 * float(num_correct) / float(num_total)

            # calculate accuracy of test set
            test_acc = self.test_accuracy(self.test_loader, is_draw=False)

            # schedule the learning rate according to test acc
            self.scheduler.step(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1

                # dump model to disk
                model_save_name = 'epoch_' + \
                                  str(t + self.LATEST_MODEL_ID + 1) + '.pth'
                torch.save(self.net.state_dict(),
                           os.path.join(self.path['net'], model_save_name))
                print('<= {} saved.'.format(model_save_name))
            print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))

            # statistics of details of each epoch
            err_dict_path = './err_dict.pkl'
            pickle.dump(self.err_dict, open(err_dict_path, 'wb'))
            print('=> err_dict dumped @ %s' % err_dict_path)
            self.err_dict = {}  # reset err dict

        print('=> Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def test_accuracy(self, data_loader, is_draw=False):
        """
        multi-label test acc
        """
        self.net.eval()  # test mode

        num_correct = 0
        num_total = 0

        # counters
        num_color = 0
        num_direction = 0
        num_type = 0
        total_time = 0.0

        print('=> testing...')
        for data, label, f_name in data_loader:
            # place data in device
            if is_draw:
                img = data.cpu()[0]
                img = self.ivt_tensor_img(img)  # Tensor -> image
            data, label = data.to(device), label.to(device)

            # format label
            label = label.cpu().long()

            start = time.time()

            # forward calculation and processing output
            output = self.net.forward(data)
            pred = self.get_predict(output)  # return to cpu

            # time consuming
            end = time.time()
            total_time += float(end - start)
            if is_draw:
                print('=> classifying time: {:2.3f} ms'.format(
                    1000.0 * (end - start)))

            # count total number
            num_total += label.size(0)

            # count each attribute acc
            color_name = self.color_attrs[pred[0][0]]
            direction_name = self.direction_attrs[pred[0][1]]
            type_name = self.type_attrs[pred[0][2]]

            if is_draw:
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(img)
                plt.title(color_name + ' ' + direction_name + ' ' + type_name)
                plt.show()

            # num_correct += self.count_correct(pred, label)
            num_correct += self.statistics_result(pred, label, f_name)

            # calculate acc of each attribute
            num_color += self.count_attrib_correct(pred, label, 0)
            num_direction += self.count_attrib_correct(pred, label, 1)
            num_type += self.count_attrib_correct(pred, label, 2)

        # calculate time consuming of inference
        print('=> average inference time: {:2.3f} ms'.format(
            1000.0 * total_time / float(len(data_loader))))

        accuracy = 100.0 * float(num_correct) / float(num_total)
        color_acc = 100.0 * float(num_color) / float(num_total)
        direction_acc = 100.0 * float(num_direction) / float(num_total)
        type_acc = 100.0 * float(num_type) / float(num_total)

        print(
            '=> test accuracy: {:.3f}% | color acc: {:.3f}%, direction acc: {:.3f}%, type acc: {:.3f}%'.format(
                accuracy, color_acc, direction_acc, type_acc))
        return accuracy

    def get_predict(self, output):
        """
        processing output
        :param output:
        :return: prediction
        """
        # get prediction for each label
        output = output.cpu()  # get data back to cpu side
        pred_color = output[:, :9]
        pred_direction = output[:, 9:11]
        pred_type = output[:, 11:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def count_correct(self, pred, label):
        """
        :param pred:
        :param label:
        :return:
        """
        # label_cpu = label.cpu().long()  # 需要将label转化成long tensor
        assert pred.size(0) == label.size(0)
        correct_num = 0
        for one, two in zip(pred, label):
            if torch.equal(one, two):
                correct_num += 1
        return correct_num

    def statistics_result(self, pred, label, f_name):
        """
        statistics of correct and error
        :param pred:
        :param label:
        :param f_name:
        :return:
        """
        # label_cpu = label.cpu().long()
        assert pred.size(0) == label.size(0)
        correct_num = 0
        for name, one, two in zip(f_name, pred, label):
            if torch.equal(one, two):  # statistics of correct number
                correct_num += 1
            else:  # statistics of detailed error info
                pred_color = self.color_attrs[one[0]]
                pred_direction = self.direction_attrs[one[1]]
                pred_type = self.type_attrs[one[2]]

                label_color = self.color_attrs[two[0]]
                label_direction = self.direction_attrs[two[1]]
                label_type = self.type_attrs[two[2]]
                err_result = label_color + ' ' + label_direction + ' ' + label_type + \
                    ' => ' + \
                    pred_color + ' ' + pred_direction + ' ' + pred_type
                self.err_dict[name] = err_result
        return correct_num

    def count_attrib_correct(self, pred, label, idx):
        """
        :param pred:
        :param label:
        :param idx:
        :return:
        """
        assert pred.size(0) == label.size(0)
        correct_num = 0
        for one, two in zip(pred, label):
            if one[idx] == two[idx]:
                correct_num += 1
        return correct_num

    def ivt_tensor_img(self, inp, title=None):
        """
        Imshow for Tensor.
        """

        # turn channelsxWxH into WxHxchannels
        inp = inp.numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # de-standardization
        inp = std * inp + mean

        # clipping
        inp = np.clip(inp, 0, 1)

        # plt.imshow(inp)
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)  # pause a bit so that plots are updated
        return inp

    def recognize_pil(self, image):
        """
        classify a single image
        :param img: PIL Image
        :return:
        """
        img = deepcopy(image)
        if img.mode == 'L' or img.mode == 'I':  # turn 8bits or 32bits gray into RGB
            img = img.convert('RGB')
        img = self.test_transforms(img)
        img = img.view(1, 3, self.net.module.input_size,
                       self.net.module.input_size)

        # put data to device
        img = img.to(device)

        start = time.time()

        # inference calculation
        output = self.net.forward(img)

        # get prediction
        pred = self.get_predict(output)

        end = time.time()

        print('=> classifying time: {:2.3f} ms'.format(1000.0 * (end - start)))

        color_name = self.color_attrs[pred[0][0]]
        direction_name = self.direction_attrs[pred[0][1]]
        type_name = self.type_attrs[pred[0][2]]

        # fig = plt.figure(figsize=(6, 6))
        # plt.imshow(image)
        # plt.title(color_name + ' ' + direction_name + ' ' + type_name)
        # plt.show()

    def test_single(self):
        """
        test single image
        :return:
        """
        self.net.eval()

        root = '/mnt/diskc/even/Car_DR/test_set'
        for file in os.listdir(root):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            self.recognize_pil(image)

    def random_pick(self, src, dst, pick_num=20):
        """
        random pick from src to dst
        :param src:
        :param dst:
        :return:
        """
        if not os.path.exists(src) or not os.path.exists(dst):
            print('=> [Err]: invalid dir.')
            return

        if len(os.listdir(dst)) != 0:
            shutil.rmtree(dst)
            os.mkdir(dst)

        # recursive traversing, search for '.jpg'
        jpgs_path = []

        def find_jpgs(root, jpgs_path):
            """
            :param root:
            :param jpgs_path:
            :return:
            """
            for file in os.listdir(root):
                file_path = os.path.join(root, file)

                if os.path.isdir(file_path):  # if dir do recursion
                    find_jpgs(file_path, jpgs_path)
                else:  # if file, put to list
                    if os.path.isfile(file_path) and file_path.endswith('.jpg'):
                        jpgs_path.append(file_path)

        find_jpgs(src, jpgs_path)
        # print('=> all jpgs path:\n', jpgs_path)

        # no replace random pick
        pick_ids = np.random.choice(
            len(jpgs_path), size=pick_num, replace=False)
        for id in pick_ids:
            shutil.copy(jpgs_path[id], dst)


def run():
    """
    main loop function
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Train bi-linear CNN based vehicle multilabel classification.')
    parser.add_argument('--base_lr',
                        dest='base_lr',
                        type=float,
                        default=1.0,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        type=int,
                        default=64,  # 64
                        help='Batch size.')  # 用多卡可以设置的更大
    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default=100,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay',
                        dest='weight_decay',
                        type=float,
                        default=1e-8,
                        help='Weight decay.')
    # parser.add_argument('--use-cuda', type=bool, default=True,
    #                     help='whether to use GPU or not.')
    parser.add_argument('--is-freeze',
                        type=bool,
                        default=True,
                        help='whether to freeze all other layers except FC layer.')
    parser.add_argument('--is-resume',
                        type=bool,
                        default=False,
                        help='whether to resume from checkpoints')
    parser.add_argument('--pre-train',
                        type=bool,
                        default=True,
                        help='whether in pre training mode.')
    args = parser.parse_args()

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must > 0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must > 0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must > 0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must > 0.')

    if args.pre_train:
        options = {
            'base_lr': args.base_lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'is_freeze': True,
            'is_resume': False
        }
    else:
        options = {
            'base_lr': args.base_lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'is_freeze': False,
            'is_resume': True
        }

    # super parameters for fine-tuning
    if not options['is_freeze']:
        options['base_lr'] = 1e-3
        options['epochs'] = 100
        options['weight_decay'] = 1e-8  # 1e-8
    print('=> options:\n', options)

    parent_dir = os.path.realpath(
        os.path.join(os.getcwd(), '..')) + os.path.sep
    project_root = parent_dir
    print('=> project_root: ', project_root)

    if is_remote:  # local paths
        path = {
            'net': '/mnt/diskc/even/b_cnn/filter_test_model',
            'model_id': '-1',  # -1
            'train_data': '/mnt/diskc/even/vehicle_train',
            'test_data': '/mnt/diskc/even/vehicle_test'
        }
    else:  # remote paths
        path = {
            'net': './checkpoints',
            'model_id': '-1',
            'train_data': 'f:/vehicle_train',
            'test_data': 'f:/vehicle_test'
        }

    manager = Manager(options, path)
    manager.train()
    # manager.test_accuracy(manager.test_loader, is_draw=True)
    # manager.random_pick(src='/mnt/diskc/even/Car_DR/vehicle_test', dst='/mnt/diskc/even/Car_DR/test_set')
    # manager.test_single()


if __name__ == '__main__':
    run()

