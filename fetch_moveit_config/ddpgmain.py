#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
from SimpleDDPG import DDPG
from FakeRobot import Robot
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 30000

if __name__ == "__main__":
    robot = Robot()
    robot.random_box_position()
    Box_position = robot.read_box_position()
    rl = DDPG()
    # rl.load_model()
    print(Box_position)
    var = 1.0
    total_rewards = []
    step_sums = []
    # 主循环
    for i in range(1, MAX_EPISODES):
        # robot.random_box_position()
        # Box_position = robot.read_box_position()
        print "Box_Position:", Box_position
        recent_end_goal = [0.715976, 0.029221, 1.0]
        robot.end_goal = recent_end_goal    # 末端坐标位置
        if i % 50 == 0:
            print("\n------------------Episode:{0}------------------".format(i))
        st = 0
        rw = 0
        # print "cube position:", Box_position
        # 存储夹爪距离木块的距离
        now_dis = math.sqrt(math.pow(recent_end_goal[0] - Box_position[0], 2)
                            + math.pow(recent_end_goal[1] - Box_position[1], 2)
                            + math.pow(recent_end_goal[2] - Box_position[2], 2))
        robot.dis = now_dis
        # 读取end_goal
        state = robot.get_state()
        if i % 500 == 0:
            print("****************memory counter:{0}****************".format(rl.memory_counter))
        end = time.clock()
        begin = time.clock()
        # 分成末端坐标和 rgbd
        while True:
            st += 1
            # ------------ choose action ------------
            s = np.array(state).reshape(1, -1)
            goal = np.array(Box_position).reshape(1, -1)
            x = np.concatenate((s, goal), axis=1)
            action = rl.choose_action(x)
            if st == 1:
                print("action: ", action)
            # ------------ -------------- ------------
            next_state, r, done, success = robot.test_step(action, var)

            # ----------- store transition -----------
            s_next = np.array(next_state).reshape(1, -1)
            x1 = np.concatenate((s_next, goal), axis=1)
            rl.store_transition(x, action, r, x1)
            # ----------- ---------------- -----------
            if rl.memory_counter > MEMORY_CAPACITY:
                var *= .9995
                rl.learn()
                if st == 1:
                    print(".....................learn.....................")
            rw += r
            state = next_state
            if done or st >= MAX_EP_STEPS:
                # if rl.memory_counter < 10000 and i % 50 == 0:
                #     break
                print("Step:{0}, total reward:{1}, average reward:{2}, {3}".format(st, rw, rw*1.0/st, "sucess"if success else "-----"))
                total_rewards.append(rw)
                step_sums.append(st)
                break
        print("{0}\n".format(state))
        if rl.memory_counter > MEMORY_CAPACITY and i % 50 == 0:
            rl.save_model()
    file_ob = open('table.txt', 'w')
    for i in total_rewards:
        file_ob.write(str(i))
        file_ob.write('\n')
    file_ob.close()

    file_ob = open('step.txt', 'w')
    for i in step_sums:
        file_ob.write(str(i))
        file_ob.write('\n')
    file_ob.close()
