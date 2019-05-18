# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.models as models
import os
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
EPOCH = 100
STEP = 1000
BATCH_SIZE = 100
LR = 0.0005


def test_read():
    position, x1, y1, z1 = read_data(size=4)
    for i in range(len(position)):
        print(position)
        rgb, dep = read_img_dep(position[i][:-4])
        plt.imshow(rgb)
        plt.show()
        plt.imshow(dep)
        plt.show()
        x1 = float(position[i][1:-11].split(" ")[0][:-1])
        y1 = float(position[i][1:-11].split(" ")[1])
        print x1, y1


def random_read_():
    tar, pos = data_prepare()
    index = np.random.randint(1, 9999)
    while int(pos[index].split(".")[2][:2]) > 27:
        index += 1
        index %= 9999
    while True:
        index += 1
        index %= 9999
        rgb, dep = read_img_dep(pos[index][:-4])
        # plt.imshow(self.rgb)
        # plt.show()
        xy = np.mean(np.argwhere(rgb[:, :, 2] == 0), axis=0).astype(np.float32)
        x = xy[0].astype(np.int32)
        y = xy[1].astype(np.int32)
        if np.isnan(np.max(xy)):
            continue
        xyd = np.concatenate((xy, np.array(dep[x][y][:1])), axis=0)
        break
    xyd = torch.from_numpy(xyd.reshape((1, 3)).astype(np.float32)).cuda()
    xyz = torch.from_numpy(tar[index].astype(np.float32)).cuda()
    return xyd, xyz


def read_img_dep(file_name):
    dep = np.load("/home/ljt/Desktop/images/dep/" + file_name + ".npy").reshape((224, 224, 1))
    dep[np.isnan(dep)] = 0
    dep = np.concatenate((dep, dep, dep), axis=2)
    return cv2.imread("/home/ljt/Desktop/images/rgb/" + file_name + ".png"), dep


def read_data(file_dir="/home/ljt/Desktop/images/dep/", size=1):
    position = []
    number = 0
    x1 = []
    y1 = []
    z1 = []
    for root_name, not_use, files_name in os.walk(file_dir):
        # print "load data from:{0},there are {1} picture".format(root_name, len(files_name))
        for i in files_name:
            position.append(i)
            x1.append(float(i[1:-11].split(" ")[0][:-1]))
            y1.append(float(i[1:-11].split(" ")[1]))
            z1.append(float(0.75))
            number += 1
            if number == size:
                break
    return position, x1, y1, z1


def data_prepare():
    position, x, y, z = read_data(size=10000)
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    z = np.array(z).reshape(-1, 1)
    return np.concatenate((x, y, z), axis=1), position


class MyDataset(Dataset):
    def __init__(self):
        self.tar, self.pos = data_prepare()
        self.y_data = torch.from_numpy(self.tar)
        self.len = self.tar.shape[0]

    def __getitem__(self, index):
        while int(self.pos[index].split(".")[2][:2]) > 27:
            index += 1
            index %= 9999
        while True:
            index += 1
            index %= 9999
            self.rgb, self.dep = read_img_dep(self.pos[index][:-4])
            # plt.imshow(self.rgb)
            # plt.show()
            self.xy = np.mean(np.argwhere(self.rgb[:, :, 2] == 0), axis=0).astype(np.float32)
            self.x = self.xy[0].astype(np.int32)
            self.y = self.xy[1].astype(np.int32)
            if np.isnan(np.max(self.xy)):
                continue
            self.xyd = np.concatenate((self.xy, np.array(self.dep[self.x][self.y][:1])), axis=0)
            break
        return self.xyd, self.y_data[index]

    def __len__(self):
        return self.len


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        print 'MyLoss'

    def forward(self, pred, truth):
        var = []
        for i in range(pred.shape[1]):
            var.append(torch.abs(pred[:, i] - truth[:, i]))
        return torch.mean(var[0]) + torch.mean(var[1]) + torch.mean(var[2])


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.normal = nn.BatchNorm2d(3)
        self.resnet18 = models.resnet18(True)

        self.convolution_net = nn.Sequential(OrderedDict([
            ('push-conv0', nn.Conv2d(3, 128, kernel_size=1, stride=1)),
            ('push-conv1', nn.Conv2d(128, 256, kernel_size=1, stride=1)),
            ('push-conv2', nn.Conv2d(256, 64, kernel_size=1, stride=1)),
            ('push-conv3', nn.Conv2d(64, 1, kernel_size=1, stride=1))
        ]))
        self.linear_net = nn.Sequential(OrderedDict([
            ('push-linear0', nn.Linear(3, 128)),
            ('push-linear1', nn.Linear(128, 256)),
            ('push-linear2', nn.Linear(256, 64)),
            ('push-linear3', nn.Linear(64, 16)),
            ('push-linear4', nn.Linear(16, 3))
        ]))

    def forward(self, xyd):
        xyz = self.linear_net(xyd)
        # xyz = self.convolution_net(xyz.unsqueeze(dim=2).unsqueeze(dim=2)).squeeze(dim=2).squeeze(dim=2)
        return xyz  #


def test():
    model = torch.load("xy2xyz.pkl")
    t_xyd, t_xyz = random_read_()
    t_out = model(t_xyd).data
    print test_xyz
    print test_out[0]
    ac = float(cal_acc(t_xyz.cpu().data.numpy(), t_out[0].cpu().data.numpy()))
    print "|test accuracy:%.4f" % ac


def cal_acc(a, b):
    sum1 = 0
    for i in range(len(a)):
            sum1 += abs(a[i] - b[i])
    return 1 - sum1


if __name__ == "__main__":
    print "-------{Building Network}-------"
    cnn = CNN().cuda()  # print cnn
    print "-------{Build finish!}-------"
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
    # loss_func = nn.L1Loss()
    loss_func = MyLoss()



    data = MyDataset()
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    print "-------{Start Trian}-------"
    for epoch in range(EPOCH):
        for batch_idx, (xyd, xyz) in enumerate(train_loader):
            # print batch_idx
            b_xyd = Variable(xyd).cuda()  # todo
            b_xyz = Variable(xyz).cuda()
            output = cnn(b_xyd).double()
            loss = loss_func(output, b_xyz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                test_xyd, test_xyz = random_read_()
                test_out = cnn(test_xyd).data
                print test_xyz
                print test_out[0]
                accuracy = float(cal_acc(test_xyz.cpu().data.numpy(), test_out[0].cpu().data.numpy()))
                print "Epoch:", epoch, "|train loss: %.4f" % loss.cpu().data.numpy(), "|test accuracy:%.4f" % accuracy
                print "\n"
        torch.save(cnn, "xy2xyz.pkl")
