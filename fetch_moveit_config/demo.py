from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torch.nn import Linear

from collect_img_goal import read
from torch.utils.data import TensorDataset
import numpy as np
# EPOCH = 1
# BATCH_SIZE = 10
# LR = 0.0005
#
# # print "-------{Loading data}-------"
# # x, y, data = read()
# # print x[0], y[0], data[0]
# # x = np.array(x).reshape(-1, 1)
# # y = np.array(y).reshape(-1, 1)
# # x_y = torch.from_numpy(np.concatenate((x, y), axis=1))
# # data = torch.from_numpy(np.array(data)).permute(0, 3, 1, 2)
# #
# # print x_y.size(), data.size()
# #
# # Train_DeepDataSet = TensorDataset(data[:100], x_y[:100])
# # Test_DeepDataSet = TensorDataset(data[21:23], x_y[21:23])
# # train_loader = Data.DataLoader(dataset=Train_DeepDataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
# #
# # test_x = Variable(torch.unsqueeze(data[21:23], dim=1)).type(torch.FloatTensor)  # .cuda todo
# # test_y = x_y[21:23]  # .cuda todo
# #
# # # print test_x
# # print "-------{Load data finish!!!}-------"
#
#
# class CNN(nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         print self.out
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=16,  # (16, 224, 224)
#                       kernel_size=5, stride=1, padding=2),
#             # padding=(kernel_size-1)/2=(5-1)/2
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # (16, 112, 112)
#         )
#         # self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,  # (16, 224, 224)
#         #               kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),  # (32, 112, 112)
#             # padding=(kernel_size-1)/2=(5-1)/2
#             nn.Dropout(0.5),
#             nn.ReLU(),  # (32, 112, 112)
#             nn.MaxPool2d(kernel_size=2),  # (32, 56, 56)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, 1, 2),  # (64, 56, 56)
#             # padding=(kernel_size-1)/2=(5-1)/2
#             nn.Dropout(0.5),
#             nn.ReLU(),  # (64, 56, 56)
#             nn.MaxPool2d(kernel_size=2),  # (64, 28, 28)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 64, 5, 1, 2),  # (64, 28, 28)
#             # padding=(kernel_size-1)/2=(5-1)/2
#             nn.Dropout(0.5),
#             nn.ReLU(),  # (64, 28, 28)
#             nn.MaxPool2d(kernel_size=2),  # (64, 14, 14)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(64, 128, 5, 1, 2),  # (128, 14, 14)
#             # padding=(kernel_size-1)/2=(5-1)/2
#             nn.Dropout(0.5),
#             nn.ReLU(),  # (128, 14, 14)
#             nn.MaxPool2d(kernel_size=2),  # (128, 7, 7)
#         )
#         self.out = nn.Linear(128 * 7 * 7, 1000)
#         self.final = nn.Linear(1000, 2)
#
#     def forward(self, x):
#         # x1 = self.conv1(x)
#         # x2 = self.conv2(x1)
#         # x3 = self.conv3(x2)
#         # x4 = self.conv4(x3)
#         # x5 = self.conv5(x4)
#         # print type(x5), x5.shape
#         # conv_out = x5.reshape(x5.shape[0], -1)
#         # # conv_out = x.view(x5.size(0), -1)  # flat (batch_size, 32*7*7)
#         # print type(conv_out)
#         # print "128*7*7:", conv_out.shape
#         x = self.out(x)
#         # print "500:", middle.shape
#         # final = self.final(middle)
#         # return final
#         return x
#
#
# x = torch.Tensor(1, 128*7*7)
# cnn = CNN()
#
# y = cnn(x)
# print y
#

entroy=nn.CrossEntropyLoss()
input=torch.Tensor(2, 3)
target = torch.Tensor([0,1]).long()
print(input)
print(target)
output = entroy(input, target)
# print output

input = torch.randn(2, 3, requires_grad=True)
target = torch.empty(2, dtype=torch.int64).random_(3)
print(input)
print(target)
output = entroy(input, target)

# print "-------{Building Network}-------"
# cnn = CNN()  # print cnn
# #  cnn.cuda() todo
# print "-------{Build finish!}-------"
#
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()
#
#
# def cal_acc(a, b):
#     sum1 = 0
#     for i in range(len(a)):
#         if a[i] == b[i]:
#             sum1 += 1
#     return sum1
#
#
# print "-------{Start Trian}-------"
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         print x.shape
#         b_x = Variable(x)  # .cuda() todo
#         b_y = Variable(y)  # .cuda() todo
#         output = cnn(b_x)
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             test_out = cnn(test_x)
#             pred_y = torch.max(test_out, 1)[1].data.squeeze()
#             # pred_y = torch1.max(test_out, 1)[1].cuda().data.squeeze() todo
#
#             accuracy = float(cal_acc(test_y, pred_y)) / test_y.size(0)
#             print "Epoch:", epoch, "|train loss: %.4f" % loss.data.numpy(), "|test accuracy:%.4f" % accuracy
#
# test_out = cnn(test_x[:10])
#
# pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
# # pred_y = torch1.max(test_out, 1)[1].cuda().data.numpy().squeeze() todo
# # pred_y.cpu() todo
#
# print pred_y
# print test_y[:10].numpy()
