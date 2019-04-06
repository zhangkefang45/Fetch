#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
from camera import RGBD
from DDPG import  DDPG
from DQN import  DQN
from Env import Robot, CubesManager
import copy
import rospy
MAX_EPISODES = 5000
MAX_EP_STEPS = 100
MEMORY_CAPACITY = 1000
if __name__ == "__main__":
    robot = Robot()
    s_dim = robot.state_dim
    a_dim = robot.action_dim
    a_bound = robot.action_bound
    cubm = CubesManager()
    rl = DQN()

    rl.eval_net = torch.load('eval_dqn.pkl')
    rl.target_net = torch.load('target_dqn.pkl')

    robot.reset()
    start_position = robot.gripper.get_current_pose("gripper_link").pose.position  # 初始的夹爪位置
    st = 0
    rw = 0
    for i in range(1, MAX_EPISODES):
        cubm.reset_cube(rand=True)
        Box_position = cubm.read_cube_pose("cube1")             # 获取物块位置
        # print "cube position:", Box_position
        robot.Box_position = copy.deepcopy(Box_position)
        now_dis = math.sqrt(math.pow(start_position.x - robot.Box_position[0], 2)
                            + math.pow(start_position.y - robot.Box_position[1], 2))
        # 存储夹爪距离木块的距离
        robot.dis = now_dis
                            # + math.pow(now_position.z - robot.Box_position[2], 2))
        # 存储end_goal
        robot.end_goal = [start_position.x, start_position.y, start_position.z]
        s = robot.get_state()
        # 分成末端坐标和rgbd
        endg, view_state = s
        for j in range(1, 5):
            st += 1
            a = rl.choose_action([endg, view_state])               # choose 时沿用之前的图像
            s_, r, done = robot.test_step(a)                       # 执行一步
            rl.store_transition(s, a, -r, [s_, view_state])       # 沿用之前的图像rgbd
            if rl.memory_counter>1000:
                if rl.memory_counter % 50 == 0:
                    print "learn....."
                rl.learn()
            rw += r
            if done:
                break
        if i%50 == 0:
            print("total reward:{0}, average reward:{1}".format(rw, rw*1.0/st))
            rw = 0
            st = 0
