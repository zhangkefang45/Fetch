import math
from camera import RGBD
from DQN import DQN
from Env import Robot, CubesManager
import copy
import rospy


robot = Robot()
cum = CubesManager()
cum.reset_cube(False)
robot.test1()
robot.reset()