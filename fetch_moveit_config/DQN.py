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
MEMORY_CAPACITY = 2000      # 记忆库大小
N_ACTIONS = 7               # 机械臂能做的动作
N_STATES = 224*224*4

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 24, kernel_size=5, stride=2)
        self.conv1.weight.data.norm(0, 0.1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=6, stride=2)
        self.conv2.weight.data.norm(0, 0.1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=6)
        self.conv3.weight.data.norm(0, 0.1)

        self.fc1 = nn.Linear(48*3*3 + 7, 256)
        self.fc1.weight.data.norm(0, 0.1)
        self.fc2 = nn.Linear(256, 48)
        self.fc2.weight.data.norm(0, 0.1)
        self.fc3 = nn.Linear(48, 7)
        self.fc3.weight.data.norm(0, 0.1)

    def forward(self, x, state):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))     # 224*224 => 53*53
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))     # 53*53 => 12*12
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))     # 12*12 => 3*3

        x = torch.cat(x.view(-1, self.num_flat_features(x)), state)
        x = F.relu(self.fc1(x))
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
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0     # 用于target更新计时
        self.memory_counter = 0         # 记忆库计数
        self.memory = (MEMORY_CAPACITY, 224*224*4*2+8)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    # 根据神经网络选取一个值
    def choose_action(self, x):
        # x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform()<EPSILON:
            action = np.array(self.eval_net.forward(x))
        else:
            action = np.random.uniform(6)-0.5
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # target net 参数更新
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # 抽取记忆库中的批数据
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = self.memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :N_STATES+7])
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 7]).astype(float)
            b_r = torch.FloatTensor(b_memory[:, N_STATES + 7:N_STATES + 8])
            b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

            # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
            q_eval = self.eval_net(b_s)              # shape (batch, 1)
            q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
            q_target = b_r + GAMMA * q_next  # shape (batch, 1)
            loss = self.loss_func(q_eval, q_target)

            # 计算, 更新 eval net
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()





if __name__ == "__main__":
    net = Net()
    print(net)



