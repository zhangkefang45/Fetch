#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, cv2, math, os, rospy, sys, threading, time
# from pprint import pprint
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
import actionlib
import copy
import math
import torch
import rospy, sys
import moveit_commander
import control_msgs.msg
import random
import cv2
import numpy as np
from gazebo_msgs.msg import ModelState, ModelStates
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface,
                           PickPlaceInterface)
from tf.transformations import quaternion_from_euler
from moveit_python.geometry import rotate_pose_msg_by_euler_angles
from control_msgs.msg import PointHeadAction, PointHeadGoal
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from moveit_msgs.msg import PlaceLocation, MoveItErrorCodes
from camera import RGBD
from geometry_msgs.msg import PoseStamped, Pose

from cv_bridge import CvBridge, CvBridgeError

# 超参数
CLOSED_POS = 0.0  # The position for a fully-closed gripper (meters).
OPENED_POS = 0.10  # The position for a fully-open gripper (meters).
ACTION_SERVER = 'gripper_controller/gripper_action'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move base using navigation stack
class MoveBaseClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.client.wait_for_server()

    def goto(self, x, y, theta, frame="map"):
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = x
        move_goal.target_pose.pose.position.y = y
        move_goal.target_pose.pose.orientation.z = math.sin(theta/2.0)
        move_goal.target_pose.pose.orientation.w = math.cos(theta/2.0)
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()

        # TODO wait for things to work
        self.client.send_goal(move_goal)
        self.client.wait_for_result()

class PointHeadClient(object):

    def __init__(self):
        self.client = actionlib.SimpleActionClient("head_controller/point_head", PointHeadAction)
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class GraspingClient(object):

    def __init__(self):
        self.scene = PlanningSceneInterface("base_link")
        self.pickplace = PickPlaceInterface("arm", "gripper", verbose=True)
        self.move_group = MoveGroupInterface("arm", "base_link")

        find_topic = "basic_grasping_perception/find_objects"
        rospy.loginfo("Waiting for %s..." % find_topic)
        self.find_client = actionlib.SimpleActionClient(find_topic, FindGraspableObjectsAction)
        self.find_client.wait_for_server()

    def updateScene(self):
        # find objects
        goal = FindGraspableObjectsGoal()
        goal.plan_grasps = True
        self.find_client.send_goal(goal)
        self.find_client.wait_for_result(rospy.Duration(5.0))
        find_result = self.find_client.get_result()

        # remove previous objects
        for name in self.scene.getKnownCollisionObjects():
            self.scene.removeCollisionObject(name, False)
        for name in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(name, False)
        self.scene.waitForSync()

        # insert objects to scene
        objects = list()
        idx = -1

        if not find_result or find_result.objects == []:
            print "find object false"
            return False

        for obj in find_result.objects:
            idx += 1
            obj.object.name = "object%d"%idx
            self.scene.addSolidPrimitive(obj.object.name,
                                         obj.object.primitives[0],
                                         obj.object.primitive_poses[0],
                                         wait = False)
            if obj.object.primitive_poses[0].position.x < 0.85:
                objects.append([obj, obj.object.primitive_poses[0].position.z])

        for obj in find_result.support_surfaces:
            # extend surface to floor, and make wider since we have narrow field of view
            height = obj.primitive_poses[0].position.z
            obj.primitives[0].dimensions = [obj.primitives[0].dimensions[0],
                                            1.5,  # wider
                                            obj.primitives[0].dimensions[2] + height]
            obj.primitive_poses[0].position.z += -height/2.0

            # add to scene
            self.scene.addSolidPrimitive(obj.name,
                                         obj.primitives[0],
                                         obj.primitive_poses[0],
                                         wait = False)

        self.scene.waitForSync()

        # store for grasping
        #self.objects = find_result.objects
        self.surfaces = find_result.support_surfaces

        # store graspable objects by Z
        objects.sort(key=lambda object: object[1])
        objects.reverse()
        self.objects = [object[0] for object in objects]
        #for object in objects:
        #    print(object[0].object.name, object[1])
        #exit(-1)
        return True

    def getGraspableObject(self):
        graspable = None
        for obj in self.objects:
            # need grasps
            if len(obj.grasps) < 1:
                continue
            # check size
            if obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25 or \
               obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25 or \
               obj.object.primitives[0].dimensions[0] < 0.03 or \
               obj.object.primitives[0].dimensions[0] > 0.25:
                continue
            # has to be on table
            if obj.object.primitive_poses[0].position.z < 0.5:
                continue
            print(obj.object.primitive_poses[0], obj.object.primitives[0])
            return obj.object, obj.grasps
        # nothing detected
        return None, None

    def getSupportSurface(self, name):
        for surface in self.support_surfaces:
            if surface.name == name:
                return surface
        return None

    def getPlaceLocation(self):
        pass

    def pick(self, block, grasps):
        success, pick_result = self.pickplace.pick_with_retry(block.name,
                                                              grasps,
                                                              support_name=block.support_surface,
                                                              scene=self.scene)
        self.pick_result = pick_result
        return success

    def place(self, block, pose_stamped):
        places = list()
        l = PlaceLocation()
        l.place_pose.pose = pose_stamped.pose
        l.place_pose.header.frame_id = pose_stamped.header.frame_id

        # copy the posture, approach and retreat from the grasp used
        l.post_place_posture = self.pick_result.grasp.pre_grasp_posture
        l.pre_place_approach = self.pick_result.grasp.pre_grasp_approach
        l.post_place_retreat = self.pick_result.grasp.post_grasp_retreat
        places.append(copy.deepcopy(l))
        # create another several places, rotate each by 360/m degrees in yaw direction
        m = 16 # number of possible place poses
        pi = 3.141592653589
        for i in range(0, m-1):
            l.place_pose.pose = rotate_pose_msg_by_euler_angles(l.place_pose.pose, 0, 0, 2 * pi / m)
            places.append(copy.deepcopy(l))

        success, place_result = self.pickplace.place_with_retry(block.name,
                                                                places,
                                                                scene=self.scene)
        return success

    def tuck(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return

    def stow(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return

    def intermediate_stow(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        pose = [0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0]
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return


class CubesManager(object):
    CubeMap = {'cube1': {'init': [0.8, 0.1, 0.75]}}

    def __init__(self):
        """
        :param cubes_name: a list of string type of all cubes
        """
        # rospy.init_node('cube_demo')
        self.cubes_pose = ModelState()
        self.cubes_state = dict()
        # pos publisher
        self.pose_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.pose_sub = rospy.Subscriber("/gazebo/cubes", ModelStates, callback=self.callback_state, queue_size=1)

    def reset_cube(self, rand=False):
        if not rand:
            for k, v in self.CubeMap.items():
                self.set_cube_pose(k, v['init'])
        else:
            for k, v in self.CubeMap.items():
                pose = [0.8, 0.1, v["init"][2]]
                pose[0] += - 0.2 + random.random() * 0.4
                pose[1] += - 0.2 + random.random() * 0.4
                self.set_cube_pose(k, pose)

    def callback_state(self, data):
        for idx, cube in enumerate(data.name):
            self.cubes_state.setdefault(cube, [0] * 3)
            pose = self.cubes_state[cube]
            cube_init = self.CubeMap[cube]["init"]
            pose[0] = data.pose[idx].position.x + cube_init[0]
            pose[1] = data.pose[idx].position.y + cube_init[1]
            pose[2] = data.pose[idx].position.z + cube_init[2]

    # def add_cube(self, name):
    #     p = PoseStamped()
    #     p.header.frame_id = ros_robot.get_planning_frame()
    #     p.header.stamp = rospy.Time.now()
    #
    #     # p.pose = self._arm.get_random_pose().pose
    #     p.pose.position.x = -0.18
    #     p.pose.position.y = 0
    #     p.pose.position.z = 0.046
    #
    #     q = quaternion_from_euler(0.0, 0.0, 0.0)
    #     p.pose.orientation = Quaternion(*q)
    #     ros_scene.add_box(name, p, (0.02, 0.02, 0.02))
    #
    # def remove_cube(self, name):
    #     ros_scene.remove_world_object(name)

    def read_cube_pose(self, name=None):
        if name is not None:
            x = self.cubes_pose.pose.position.x
            y = self.cubes_pose.pose.position.y
            z = self.cubes_pose.pose.position.z
            return [x, y, z]
        else:
            return None

    def set_cube_pose(self, name, pose, orient=None):
        """
        :param name: cube name, a string
        :param pose: cube position, a list of three float, [x, y, z]
        :param orient: cube orientation, a list of three float, [ix, iy, iz]
        :return:
        """
        self.cubes_pose.model_name = name
        p = self.cubes_pose.pose
        # cube_init = self.CubeMap[name]["init"]
        p.position.x = pose[0]
        p.position.y = pose[1]
        p.position.z = pose[2]
        if orient is None:
            orient = [0, 0, 0]
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        p.orientation = Quaternion(*q)
        self.pose_pub.publish(self.cubes_pose)


'''
机器人类：包含了一些对机器人的控制和初始化功能
函数：
    open: 控制夹爪的张开（范围0.03～0.13）
    close(widh, max_effort)：控制夹爪闭合 width:张开距离,
        max_effort:闭合力度
    step(action): 读入一组关节角，将机械臂的七个关节转动到相应的角度
    reset: 重置机械臂和木块到初始状态
    sample: 随机七个角度(范围-0.5~0.5）
'''


class Robot(object):
    MIN_EFFORT = 35  # Min grasp force, in Newtons
    MAX_EFFORT = 100  # Max grasp force, in Newtons
    dt = 0.005  # 转动的速度和 dt 有关
    action_bound = [-1, 1]  # 转动的角度范围
    state_dim = 7  # 7个观测值
    action_dim = 7  # 7个动作
    Robot = {'fetch': {'init': [0, 0, 0]}}
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)

        # 初始化ROS节点
        rospy.init_node('moveit_demo')
        self.camera = RGBD()
        # robot reset
        self.robot_pose = ModelState()
        self.robot_state = dict()
        self.pose_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        # 初始化需要使用move group控制的机械臂中的arm group
        self.arm = moveit_commander.MoveGroupCommander('arm')
        self.gripper = moveit_commander.MoveGroupCommander('gripper')
        self._client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
        self._client.wait_for_server(rospy.Duration(10))
        self.Box_position = [0.8, 0.1, 0.75]
        # 获取终端link的名称
        self.end_effector_link = self.arm.get_end_effector_link()

        # 设置目标位置所使用的参考坐标系
        self.reference_frame = 'base_link'

        self.arm.set_pose_reference_frame(self.reference_frame)
        # 获取场景中的物体
        self.head_action = PointHeadClient()
        self.grasping_client = GraspingClient()
        # self.move_base = MoveBaseClient()
        # 向下看
        self.head_action.look_at(1.2, 0.0, 0.0, "base_link")
        # 初始化机器人手臂位置
        self.arm_goal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.end_goal = [0.0, 0.0, 0.0]
        self.reset()
        # 初始化reward
        now_position = self.gripper.get_current_pose("gripper_link").pose.position
        now_dis = math.sqrt(math.pow(now_position.x - self.Box_position[0], 2)
                              + math.pow(now_position.y - self.Box_position[1], 2)
                              + math.pow(now_position.z - self.Box_position[2], 2))
        self.reward = math.exp(-now_dis)
        # 更新场景到rviz中

        # 设置目标位置所使用的参考坐标系
        reference_frame = 'base_link'
        self.arm.set_pose_reference_frame(reference_frame)

        # 当运动规划失败后，允许重新规划
        self.arm.allow_replanning(True)

        # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
        self.arm.set_goal_position_tolerance(0.01)
        self.arm.set_goal_orientation_tolerance(0.05)
        self.target_pose = PoseStamped()

    def open(self):
        """Opens the gripper.
        """
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = OPENED_POS
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

    def close(self, width=0.0, max_effort=MAX_EFFORT):
        """Closes the gripper.

        Args:
            width: The target gripper width, in meters. (Might need to tune to
                make sure the gripper won't damage itself or whatever it's
                gripping.)
            max_effort: The maximum effort, in Newtons, to use. Note that this
                should not be less than 35N, or else the gripper may not close.
        """
        assert CLOSED_POS <= width <= OPENED_POS
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width
        goal.command.max_effort = max_effort
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

    def set_end_pose(self, x, y, z):
        a = copy.deepcopy(x)
        b = copy.deepcopy(y)
        c = copy.deepcopy(z)
        self.target_pose.header.frame_id = self.reference_frame
        self.target_pose.header.stamp = rospy.Time.now()
        self.target_pose.pose.position.x = a # % 0.3 + 0.4  # 0.70
        self.target_pose.pose.position.y = b # % 0.7 - 0.35 # 0.0
        self.target_pose.pose.position.z = c
        self.target_pose.pose.orientation.x = -0.0000
        self.target_pose.pose.orientation.y = 0.681666289017
        self.target_pose.pose.orientation.z = 0
        self.target_pose.pose.orientation.w = 0.73166304586
        self.end_goal[0] = self.target_pose.pose.position.x
        self.end_goal[1] = self.target_pose.pose.position.y
        self.end_goal[2] = self.target_pose.pose.position.z
        # print self.target_pose.pose.position

    def go_end_pose(self):
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(self.target_pose, self.end_effector_link)
        traj = self.arm.plan()
        success = self.arm.execute(traj)
        rospy.sleep(1)
        return success

    def step(self, action):
        done = False
        state = None
        success = True
        # 转动一定的角度(执行动作)
        # self.arm_goal[0] = self.arm_goal[0] % (np.pi/4)
        # self.arm_goal = [(-0.6~0.6), (0~-0.8), (0), (0~1.25), 0, 1.7, 0]
        limit = [0.6, -0.8, 10, 1.25, 3.14, 2.16, 3.14]
        # self.arm_goal = action[0] % (np.pi/4)
        # if action.shape[0] == 1:
        self.end_goal += action[0]
        temporarity = copy.deepcopy(self.end_goal)

        you_want_to_pick_now = True
        if you_want_to_pick_now:  # todo
            temporarity = copy.deepcopy(self.Box_position)
        # if action.shape[0] == 7:
        #     self.arm_goal += action
        # for i in range(7):
        #     self.arm_goal[i] = self.arm_goal[i] % (limit[i] * 2)  # change the limit for each joint todo
        #     self.arm_goal[i] -= limit[i]
        # self.arm_goal[1] = -math.fabs(self.arm_goal[1])
        # self.arm_goal[2] = 0
        # self.arm_goal[3] = math.fabs(self.arm_goal[3])
        # [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        # self.arm_goal[0] = 1
        # self.arm_goal[1] = 0
        # self.arm_goal[2] = 2
        # self.arm_goal[3] = 0
        # self.arm_goal[4] = 0
        # self.arm_goal[5] = 1
        # self.arm_goal[6] = 0
        # print "action: ", action  # todo
        # self.arm_goal = np.clip(self.arm_goal, *self.action_bound)
        # print("GOAL_ARM:", self.arm_goal)

        print "---frist---"
        self.set_end_pose(temporarity[0]-0.17, temporarity[1], 0.980521666929)
        print temporarity
        try:
            success = self.go_end_pose()
            print "---second---"
            self.set_end_pose(temporarity[0] - 0.17, temporarity[1], 0.91)
            print temporarity
            success = self.go_end_pose()
            # self.arm.set_joint_value_target(self.arm_goal.tolist())
            # success = self.arm.go()
            # rospy.sleep(1)
        except Exception as e:
            print e.message
            done = True


        print(success)
        if not success or done:    # 规划失败，发生碰撞
            reward = -10
            done = True
        else:                   # 规划成功，开始运动
            # 计算距离物块的距离，来返回reward
            af_position = self.gripper.get_current_pose("gripper_link").pose.position
            af_dis = math.sqrt(math.pow(af_position.x - self.Box_position[0], 2)
                                + math.pow(af_position.y - self.Box_position[1], 2)
                                + math.pow(af_position.z - self.Box_position[2], 2))
            reward = (-af_dis)*10 - self.reward
            self.reward = (-af_dis)*10
            # 尝试抓取
            self.close()
            # 抓取成功和碰撞到环境均为结束
            l = self.gripper.get_current_pose("l_gripper_finger_link").pose.position
            r = self.gripper.get_current_pose("r_gripper_finger_link").pose.position
            # 获取夹爪的距离(范围：0.03 ~ 0.13)
            dis = math.sqrt(pow(l.x-r.x, 2)+pow(l.y-r.y, 2)+pow(l.z-r.z, 2))
            if dis > 0.031:  # 抓取成功
                self.pick_up()
                rospy.sleep(1)
                l = self.gripper.get_current_pose("l_gripper_finger_link").pose.position
                r = self.gripper.get_current_pose("r_gripper_finger_link").pose.position
                # 再次获取夹爪的距离(范围：0.3 ~ 0.13)
                dis = math.sqrt(pow(l.x - r.x, 2) + pow(l.y - r.y, 2) + pow(l.z - r.z, 2))
                if dis > 0.031:
                    reward += 100
                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                done = True
            else:
                self.open()
            # 获取状态
        state = self.get_state()
        return state, reward, done
        # print(end_x, end_y, end_z)

    def read_depth_data(self):
        # 获取深度摄像头信息
        rospy.sleep(2)
        rgb = self.camera.read_color_data()
        dep = self.camera.read_depth_data()
        dep[np.transpose(np.argwhere(np.isnan(dep)))[0], np.transpose(np.argwhere(np.isnan(dep)))[1]]=0
        # if the depth value is nan to 0 todo
        rgb = np.array(rgb)
        dep = np.array(dep)
        dep = dep[:, :, np.newaxis]
        rgbd = np.concatenate((rgb, dep), axis=2)
        new_rgbd = cv2.resize(rgbd, (224, 224))
        return new_rgbd

    def get_state(self):
        # self.camera = RGBD()
        # return self.arm_goal, self.read_depth_data()
        return self.end_goal, self.read_depth_data()

    # 初始化机器人手臂和物块位置以及RViz中的场景
    def reset(self):
        # self.arm_goal = [(-0.6~0.6), (0~-0.8), (0), (0~1.25), 0, 1.7, 0]
        # [0.0, -0.8, 0, 1.25, 0, 0, 0]

        self.arm_goal = [0.0, -0.8, 0, 0.7, 0, 1.7, 0]
        self.arm_goal = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        print "reset:", type(self.arm_goal[0])
        # self.arm_goal = [0, 0, 0, 0, 0, 0, 0]
        self.arm.set_joint_value_target(self.arm_goal)
        self.arm.go()
        rospy.sleep(2)
        self.reset_robot()
        self.open()
        rospy.sleep(1)
        while not self.grasping_client.updateScene():
            print "-----update scence fail-----"
            self.reset_robot()
            cubmanager = CubesManager()
            cubmanager.reset_cube(True)
        rospy.sleep(3)
        # print "--------------test--------------"
        # self.test()

    def pick_up(self):
        self.arm_goal = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        print "reset:", type(self.arm_goal[0])
        self.arm.set_joint_value_target(self.arm_goal)
        self.arm.go()
        rospy.sleep(2)

    def reset_robot(self):
        for k, v in self.Robot.items():
            self.set_robot_pose(k, v['init'])

    def set_robot_pose(self, name, pose, orient=None):
        """
        :param name: cube name, a string
        :param pose: cube position, a list of three float, [x, y, z]
        :param orient: cube orientation, a list of three float, [ix, iy, iz]
        :return:
        """
        self.robot_pose.model_name = name
        p = self.robot_pose.pose
        # cube_init = self.CubeMap[name]["init"]
        p.position.x = pose[0] + 0.15
        p.position.y = pose[1]
        p.position.z = pose[2]
        if orient is None:
            orient = [0, 0, 0]
        q = quaternion_from_euler(orient[0], orient[1], orient[2])
        p.orientation = Quaternion(*q)
        self.pose_pub.publish(self.robot_pose)

    def sample(self):
        # -0.5 ~ 0.5
        return (2*np.pi*np.random.rand(7)-np.pi).to(device)

    def test(self):
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.reference_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = 0.70
        target_pose.pose.position.y = 0.0
        target_pose.pose.position.z = 0.9
        target_pose.pose.orientation.x = -0.0000
        target_pose.pose.orientation.y = 0.681666289017
        target_pose.pose.orientation.z = 0
        target_pose.pose.orientation.w = 0.73166304586
        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)
        traj = self.arm.plan()
        # print "it :", traj
        self.arm.execute(traj)
        rospy.sleep(1)

    def test1(self):
        self.set_end_pose(0.8 - 0.17, 0.1, 1)
        self.go_end_pose()
        rospy.sleep(1)
        self.set_end_pose(0.8 - 0.17, 0.1, 0.91)
        self.go_end_pose()
        rospy.sleep(1)
        self.close()
        rospy.sleep(1)
        self.set_end_pose(0.8 - 0.17, 0.1, 1)
        self.go_end_pose()
        rospy.sleep(1)
        self.set_end_pose(0.8 - 0.17, -0.3, 1)
        self.go_end_pose()
        rospy.sleep(1)
        self.set_end_pose(0.8 - 0.17, -0.3, 0.91)
        self.go_end_pose()
        rospy.sleep(1)
        self.open()


if __name__ == '__main__':
    # cube_manager = CubesManager()
    robot = Robot()
    robot.move()
    # while True:
    #     robot.reset()
    #     cube_manager.reset_cube(rand=True)
    #     Box_position = cube_manager.read_cube_pose("demo_cube")
    #     Box_position[0] -= 0.2
    #     Box_position[2] -= 0.1
    #     robot.Box_position = Box_position
    #     print(cube_manager.read_cube_pose("demo_cube"))
    #     print(robot.Box_position)
    #     rospy.sleep(3)
