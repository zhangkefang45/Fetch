#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 50      # 记忆库大小
N_ACTIONS = 7               # 机械臂能做的动作
N_STATES = 224*224*4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=6)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(48*4*4 + 7, 256)
        self.fc2 = nn.Linear(256, 48)
        self.fc3 = nn.Linear(48, 7)

    def forward(self, x, state):
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))     # 224*224 => 55*55
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))     # 55*55 => 13*13
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))     # 13*13 => 4*4
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.num_flat_features(x))
        x = torch.cat([x.float(), state.float()], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0     # 用于target更新计时
        self.memory_counter = 0         # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, (224*224*4+7)*2+8))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss  # 误差公式

    # 根据神经网络选取一个值
    def choose_action(self, x):
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform()<EPSILON:
            joint_view, image_view = x
            image_view = torch.from_numpy(np.array(image_view).reshape(-1, 224, 224, 4)).permute(0, 3, 1, 2).to(device)
            joint_view = torch.from_numpy(np.array(joint_view).reshape(-1, 7)).to(device)
            action = self.eval_net.forward(image_view, joint_view).cpu().detach().numpy()
            # action = np.array(self.eval_net.forward(image_view.to(device), joint_view.to(device)))
        else:
            action = np.random.uniform(low=-np.pi, high=np.pi, size=7)-0.5
        return action

    def store_transition(self, s, a, r, s_):
        s1, s2 = s
        s3, s4 = s_
        s1 = np.array(s1).reshape(-1, 7)
        s2 = np.array(s2).reshape(-1, 224*224*4)
        a = np.array(a).reshape(-1, 7)
        r = np.array(r).reshape(-1, 1)
        s3 = np.array(s3).reshape(-1, 7)
        s4 = np.array(s4).reshape(-1, 224*224*4)

        transition = np.hstack((s1, s2, a, r, s3, s4))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s1 = torch.FloatTensor((b_memory[:, :7]).reshape(-1, 7))
        b_s2 = torch.FloatTensor((b_memory[7:, :N_STATES+7]).reshape(224, 224, 4))
        b_s = b_s1, b_s2
        b_a = torch.LongTensor((b_memory[:, N_STATES+7:N_STATES + 14]).reshape(7).astype(float))
        b_r = torch.FloatTensor((b_memory[:, N_STATES + 14:N_STATES + 15]).reshape(1))
        b_s_ = torch.FloatTensor((b_memory[:, -N_STATES:]).reshape(224, 224, 4))

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s, b_a)              # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    net = DQN()

    input = torch.randn(7), torch.randn(4, 224, 224)
    out = net.choose_action(input)
    print(out)



