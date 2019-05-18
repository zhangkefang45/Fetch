#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
import numpy as np
import os
import math
import random
import time


#####################  hyper parameters  ####################

LR_A = 0.001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 640
N_STATES = 3
RENDER = False
EPSILON = 0.9
ENV_PARAMS = {'obs': 3, 'goal': 3, 'action': 3, 'action_max': 0.2}
###############################  DDPG  ####################################


class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet, self).__init__()
        self.max_action = ENV_PARAMS['action_max']
        self.state = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(ENV_PARAMS['obs'] + ENV_PARAMS['goal'], 256, 1, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(256, 256, 1, 1)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(256, 256, 1, 1)),
                ('relu3', nn.ReLU()),
                ('conv4', nn.Conv2d(256, ENV_PARAMS['action'], 1, 1)),
                # ('tanh1', nn.Tanh())
            ]))
        # self.fc1 = nn.Linear(6, 16)
        # self.fc2 = nn.Linear(16, 64)
        # self.fc3 = nn.Linear(64, 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, s):
        return self.state(s.unsqueeze(dim=2).unsqueeze(dim=2)).squeeze(dim=2).squeeze(dim=2) * self.max_action


class CNet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(CNet, self).__init__()
        self.state = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(ENV_PARAMS['obs'] + ENV_PARAMS['goal'] + ENV_PARAMS['action'], 256, 1, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(256, 256, 1, 1)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(256, 256, 1, 1)),
                ('relu3', nn.ReLU()),
                ('conv4', nn.Conv2d(256, 1, 1, 1)),
            ]))
        # self.fc1 = nn.Linear(9, 16)
        # self.fc2 = nn.Linear(16, 64)
        # self.fc3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        return self.state(x.unsqueeze(dim=2).unsqueeze(dim=2)).squeeze(dim=2).squeeze(dim=2)


class DDPG(object):
    def __init__(self):
        self.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.device = torch.device("cuda:0")
        self.memory = np.zeros((MEMORY_CAPACITY, 6+3+1+6))
        self.memory_counter = 0  # 记忆库计数
        self.Actor_eval = ANet().to(self.device)
        self.Actor_target = ANet().to(self.device)
        self.Critic_eval = CNet().to(self.device)
        self.Critic_target = CNet().to(self.device)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()
        self.f = 0

    def choose_action(self, s):
        state = torch.FloatTensor(np.array(s).reshape(1, -1)).to(self.device)
        return self.Actor_eval(state).cpu().data.numpy()

    def learn(self):
        self.f += 1
        self.f %= 100
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor((b_memory[:, :6]).reshape(-1, 6)).to(self.device)
        b_a = torch.FloatTensor((b_memory[:, 6:9]).reshape(-1, 3)).to(self.device)
        b_r = torch.FloatTensor((b_memory[:, 9:10]).reshape(-1, 1)).to(self.device)
        b_s_ = torch.FloatTensor((b_memory[:, 10:16]).reshape(-1, 6)).to(self.device)

        # Compute the target Q value
        target_Q = self.Critic_target(b_s_, self.Actor_target(b_s_))
        target_Q = b_r + (GAMMA* target_Q).detach()

        # Get current Q estimate
        current_Q = self.Critic_eval(b_s, b_a)

        # Compute critic loss
        critic_loss = self.loss_td(current_Q, target_Q)

        # Optimize the critic
        self.ctrain.zero_grad()
        critic_loss.backward()
        self.ctrain.step()

        # Compute actor loss
        actor_loss = -self.Critic_eval(b_s, self.Actor_eval(b_s)).mean()
        if self.f == 0:
            print(actor_loss)
        # Optimize the actor
        self.atrain.zero_grad()
        actor_loss.backward()
        self.atrain.step()



        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)

        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)

    def store_transition(self, s, a, r, s_):

        s = np.array(s).reshape(-1, 6)
        a = np.array(a).reshape(-1, 3)
        r = np.array(r).reshape(-1, 1)
        s_ = np.array(s_).reshape(-1, 6)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data*(1.0 - tau) + param.data*tau
            )

    def save_model(self):
        torch.save(self.Actor_eval, "ae1.pkl")
        torch.save(self.Actor_target, "at1.pkl")
        torch.save(self.Critic_eval, "ce1.pkl")
        torch.save(self.Critic_target, "ct1.pkl")

    def load_model(self):
        self.Actor_eval = torch.load("ae1.pkl")
        self.Actor_target = torch.load("at1.pkl")
        self.Critic_eval = torch.load("ce1.pkl")
        self.Critic_target = torch.load("ct1.pkl")


