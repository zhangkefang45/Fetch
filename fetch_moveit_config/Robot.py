#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, sys
import numpy as np
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import PlanningScene, ObjectColor
from geometry_msgs.msg import PoseStamped, Pose
import Interface

class Robot():
    dt = 0.1  # 转动的速度和 dt 有关
    action_bound = [-1, 1]  # 转动的角度范围
    state_dim = 7  # 六个观测值
    action_dim = 7  # 六个动作

    def __init__(self):
        # 机械臂的允许误差值
        self.arm.set_goal_joint_tolerance(0.001)
        self.arm_goal = [0, 0, 0, 0, 0, 0, 0]
        # 初始化
        itf = Interface.MoveItIkDemo()

    #
    def step(self, action):
        done = False
        reawrd = 0

        action = np.clip(action, self.action_bound)
        # 转动一定的角度
        self.arm_goal + action*self.dt
        self.arm_goal %= 2*np.pi

        s = self.arm_goal

        # 如果和物体的距离小于一定值，则回合结束
        position = self.arm.get_current_pose('wrist_3_link').pose.position
        end_x, end_y, end_z = position.x, position.y, position.z
        # print(end_x, end_y, end_z)

    # 初始化
    def reset(self):
        self.arm_goal = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        self.arm.set_joint_value_target(self.arm_goal)
        self.arm.go()
        rospy.sleep(1)

    def sample(self):
        return np.random.rand(6)-0.5


if __name__ == '__main__':
    robot = Robot()
